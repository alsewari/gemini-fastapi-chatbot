import asyncio
import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import errors
from pydantic import BaseModel, Field

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
MAX_FILES = int(os.getenv("MAX_FILES", "5"))
MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "25"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

app = FastAPI(
    title="Gemini Chat API",
    description="A FastAPI chatbot powered by Gemini with file-upload support.",
    version="2.0.0",
)


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=10_000,
        examples=["Explain data-driven decision-making in simple terms."],
    )
    previous_interaction_id: str | None = Field(
        default=None,
        description="Return the previous interaction ID to continue the same conversation.",
    )


class ChatResponse(BaseModel):
    reply: str
    interaction_id: str
    model: str
    uploaded_files: list[str] = Field(default_factory=list)


def get_gemini_client() -> Any:
    if not API_KEY:
        raise HTTPException(
            status_code=503,
            detail="GEMINI_API_KEY is not configured. Add it to your .env file first.",
        )
    return genai.Client(api_key=API_KEY)


def detect_mime_type(file: UploadFile) -> str:
    """Return the best available MIME type for an uploaded file."""
    filename = Path(file.filename or "upload").name
    mime_type = file.content_type or mimetypes.guess_type(filename)[0]

    text_extensions = {
        ".txt",
        ".md",
        ".csv",
        ".json",
        ".xml",
        ".html",
        ".htm",
        ".py",
        ".js",
        ".ts",
        ".java",
        ".cs",
        ".sql",
        ".yaml",
        ".yml",
    }

    if not mime_type or mime_type == "application/octet-stream":
        if Path(filename).suffix.lower() in text_extensions:
            return "text/plain"
        return "application/octet-stream"

    return mime_type


def interaction_input_type(mime_type: str) -> str:
    """Map a MIME type to an Interactions API input type."""
    if mime_type.startswith("image/"):
        return "image"
    if mime_type.startswith("audio/"):
        return "audio"
    if mime_type.startswith("video/"):
        return "video"
    return "document"


def validate_mime_type(mime_type: str, filename: str) -> None:
    """Reject file types that Gemini is unlikely to process correctly."""
    allowed_application_types = {
        "application/pdf",
        "application/json",
        "application/xml",
        "application/xhtml+xml",
    }

    if (
        mime_type.startswith("text/")
        or mime_type.startswith("image/")
        or mime_type.startswith("audio/")
        or mime_type.startswith("video/")
        or mime_type in allowed_application_types
    ):
        return

    raise HTTPException(
        status_code=415,
        detail=(
            f"Unsupported file type for '{filename}': {mime_type}. "
            "Use PDF, text, CSV, JSON, Markdown, HTML, XML, image, audio, or video. "
            "Convert Word, PowerPoint, or Excel files to PDF, text, or CSV first."
        ),
    )


async def save_upload_temporarily(upload: UploadFile) -> Path:
    """Stream an uploaded file to a temporary path while enforcing a size limit."""
    filename = Path(upload.filename or "upload").name
    suffix = Path(filename).suffix

    temp_file = tempfile.NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        delete=False,
    )
    temp_path = Path(temp_file.name)
    total_size = 0

    try:
        while True:
            chunk = await upload.read(1024 * 1024)
            if not chunk:
                break

            total_size += len(chunk)
            if total_size > MAX_FILE_SIZE_BYTES:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"'{filename}' is larger than the application limit "
                        f"of {MAX_FILE_SIZE_MB} MB."
                    ),
                )

            temp_file.write(chunk)

        temp_file.close()

        if total_size == 0:
            raise HTTPException(
                status_code=400,
                detail=f"'{filename}' is empty.",
            )

        return temp_path

    except Exception:
        temp_file.close()
        temp_path.unlink(missing_ok=True)
        raise

    finally:
        await upload.close()


async def wait_until_file_ready(file_resource: Any) -> Any:
    """Wait until Gemini finishes processing an uploaded file."""
    client = get_gemini_client()
    for _ in range(60):
        state = str(getattr(file_resource, "state", "") or "").upper()

        if "FAILED" in state:
            raise HTTPException(
                status_code=502,
                detail="Gemini could not process one of the uploaded files.",
            )

        if "PROCESSING" not in state:
            return file_resource

        await asyncio.sleep(2)
        file_resource = await client.aio.files.get(name=file_resource.name)

    raise HTTPException(
        status_code=504,
        detail="The uploaded file took too long to process.",
    )


async def upload_file_to_gemini(upload: UploadFile) -> tuple[dict[str, str], str]:
    """Upload one FastAPI file to Gemini and return its interaction input block."""
    client = get_gemini_client()
    filename = Path(upload.filename or "upload").name
    mime_type = detect_mime_type(upload)
    validate_mime_type(mime_type, filename)

    temp_path = await save_upload_temporarily(upload)

    try:
        gemini_file = await client.aio.files.upload(
            file=str(temp_path),
            config={
                "mime_type": mime_type,
                "display_name": filename,
            },
        )
        gemini_file = await wait_until_file_ready(gemini_file)

        file_uri = getattr(gemini_file, "uri", None)
        if not file_uri:
            raise HTTPException(
                status_code=502,
                detail=f"Gemini did not return a file URI for '{filename}'.",
            )

        final_mime_type = getattr(gemini_file, "mime_type", None) or mime_type

        return (
            {
                "type": interaction_input_type(final_mime_type),
                "uri": file_uri,
                "mime_type": final_mime_type,
            },
            filename,
        )

    finally:
        temp_path.unlink(missing_ok=True)


async def create_gemini_response(
    *,
    input_data: str | list[dict[str, str]],
    previous_interaction_id: str | None,
    uploaded_files: list[str] | None = None,
) -> ChatResponse:
    client = get_gemini_client()
    create_args: dict[str, Any] = {
        "model": MODEL,
        "input": input_data,
    }

    if previous_interaction_id:
        create_args["previous_interaction_id"] = previous_interaction_id

    try:
        interaction = await client.aio.interactions.create(**create_args)

        reply = interaction.output_text
        if not reply:
            raise HTTPException(
                status_code=502,
                detail="Gemini returned no text response.",
            )

        return ChatResponse(
            reply=reply,
            interaction_id=interaction.id,
            model=MODEL,
            uploaded_files=uploaded_files or [],
        )

    except errors.ServerError as exc:
        raise HTTPException(
            status_code=503,
            detail=(
                "Gemini is temporarily unavailable or under high demand. "
                "Please wait briefly and try again."
            ),
        ) from exc

    except errors.ClientError as exc:
        status_code = getattr(exc, "status_code", 400)
        if not isinstance(status_code, int) or not 400 <= status_code < 500:
            status_code = 400

        raise HTTPException(
            status_code=status_code,
            detail=(
                "Gemini rejected the request. Check the API key, model name, "
                "prompt, or uploaded file type."
            ),
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        ) from exc


@app.get("/health")
def health() -> dict[str, str | int]:
    return {
        "status": "ok",
        "model": MODEL,
        "max_files": MAX_FILES,
        "max_file_size_mb": MAX_FILE_SIZE_MB,
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    """Text-only endpoint. Existing clients can continue using JSON."""
    return await create_gemini_response(
        input_data=request.message.strip(),
        previous_interaction_id=request.previous_interaction_id,
    )


@app.post("/api/chat/files", response_model=ChatResponse)
async def chat_with_files(
    message: str = Form(default=""),
    previous_interaction_id: str | None = Form(default=None),
    files: list[UploadFile] | None = File(default=None),
) -> ChatResponse:
    """Multipart endpoint for a message plus one or more uploaded files."""
    uploaded = files or []

    if not message.strip() and not uploaded:
        raise HTTPException(
            status_code=400,
            detail="Enter a message or select at least one file.",
        )

    if len(uploaded) > MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"You can upload a maximum of {MAX_FILES} files per message.",
        )

    input_blocks: list[dict[str, str]] = []
    uploaded_names: list[str] = []

    for upload in uploaded:
        file_block, filename = await upload_file_to_gemini(upload)
        input_blocks.append(file_block)
        uploaded_names.append(filename)

    prompt = message.strip() or "Please analyse and summarise the uploaded file(s)."
    input_blocks.append({"type": "text", "text": prompt})

    return await create_gemini_response(
        input_data=input_blocks,
        previous_interaction_id=previous_interaction_id,
        uploaded_files=uploaded_names,
    )


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Sewari ai chatbot using Gemini</title>
    <style>
        * {{ box-sizing: border-box; }}

        body {{
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #1f2937;
        }}

        .app {{
            width: min(920px, 94vw);
            margin: 32px auto;
            background: white;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            overflow: hidden;
        }}

        header {{
            padding: 20px 24px;
            border-bottom: 1px solid #e5e7eb;
        }}

        header h1 {{
            margin: 0 0 6px;
            font-size: 24px;
        }}

        header p {{
            margin: 0;
            color: #6b7280;
        }}

        #messages {{
            height: 500px;
            overflow-y: auto;
            padding: 22px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}

        .message {{
            max-width: 82%;
            padding: 12px 14px;
            border-radius: 12px;
            white-space: pre-wrap;
            overflow-wrap: anywhere;
            line-height: 1.45;
        }}

        .user {{
            align-self: flex-end;
            background: #2563eb;
            color: white;
        }}

        .bot {{
            align-self: flex-start;
            background: #eef2f7;
        }}

        .error {{
            align-self: flex-start;
            background: #fee2e2;
            color: #991b1b;
        }}

        form {{
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 10px;
            padding: 18px;
            border-top: 1px solid #e5e7eb;
            align-items: stretch;
        }}

        .composer {{
            display: flex;
            flex-direction: column;
            gap: 9px;
        }}

        textarea {{
            width: 100%;
            min-height: 52px;
            max-height: 150px;
            resize: vertical;
            padding: 12px;
            border: 1px solid #cbd5e1;
            border-radius: 9px;
            font-family: inherit;
            font-size: 16px;
        }}

        .file-row {{
            display: flex;
            align-items: center;
            gap: 10px;
            min-width: 0;
        }}

        #fileInput {{
            display: none;
        }}

        .file-button {{
            display: inline-block;
            padding: 8px 12px;
            border-radius: 8px;
            background: #eef2ff;
            color: #1d4ed8;
            cursor: pointer;
            font-weight: 700;
            white-space: nowrap;
        }}

        #fileStatus {{
            color: #6b7280;
            font-size: 13px;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}

        button {{
            border: 0;
            border-radius: 9px;
            padding: 0 18px;
            cursor: pointer;
            font-weight: 700;
            min-height: 52px;
        }}

        #sendButton {{
            background: #2563eb;
            color: white;
        }}

        #resetButton {{
            background: #e5e7eb;
            color: #111827;
        }}

        button:disabled {{
            opacity: 0.6;
            cursor: not-allowed;
        }}

        .small-note {{
            padding: 0 18px 15px;
            color: #6b7280;
            font-size: 12px;
        }}

        @media (max-width: 700px) {{
            form {{
                grid-template-columns: 1fr 1fr;
            }}

            .composer {{
                grid-column: 1 / -1;
            }}
        }}
    </style>
</head>
<body>
    <main class="app">
        <header>
            <h1>Sewari API Chat</h1>
            <p>Ask a question or upload files for Gemini to analyse.</p>
        </header>

        <section id="messages">
            <div class="message bot">
Hello! Ask me a question, or attach a PDF, text file, image, audio, or video.
            </div>
        </section>

        <form id="chatForm">
            <div class="composer">
                <textarea
                    id="messageInput"
                    placeholder="Ask a question about your file..."
                    autocomplete="off"
                ></textarea>

                <div class="file-row">
                    <label class="file-button" for="fileInput">📎 Add files</label>
                    <input
                        id="fileInput"
                        type="file"
                        multiple
                        accept=".pdf,.txt,.md,.csv,.json,.xml,.html,.htm,.py,.js,.ts,.java,.cs,.sql,.yaml,.yml,image/*,audio/*,video/*"
                    />
                    <span id="fileStatus">No files selected</span>
                </div>
            </div>

            <button id="sendButton" type="submit">Send</button>
            <button id="resetButton" type="button">New chat</button>
        </form>

        <div class="small-note">
            Maximum: {MAX_FILES} files per message, {MAX_FILE_SIZE_MB} MB per file.
        </div>
    </main>

    <script>
        const form = document.getElementById("chatForm");
        const input = document.getElementById("messageInput");
        const fileInput = document.getElementById("fileInput");
        const fileStatus = document.getElementById("fileStatus");
        const messages = document.getElementById("messages");
        const sendButton = document.getElementById("sendButton");
        const resetButton = document.getElementById("resetButton");

        let previousInteractionId = null;

        function addMessage(text, type) {{
            const element = document.createElement("div");
            element.className = `message ${{type}}`;
            element.textContent = text;
            messages.appendChild(element);
            messages.scrollTop = messages.scrollHeight;
        }}

        function updateFileStatus() {{
            const selected = Array.from(fileInput.files);

            if (selected.length === 0) {{
                fileStatus.textContent = "No files selected";
                return;
            }}

            fileStatus.textContent = selected.map(file => file.name).join(", ");
        }}

        fileInput.addEventListener("change", updateFileStatus);

        form.addEventListener("submit", async (event) => {{
            event.preventDefault();

            const message = input.value.trim();
            const selectedFiles = Array.from(fileInput.files);

            if (!message && selectedFiles.length === 0) {{
                addMessage("Enter a message or choose a file.", "error");
                return;
            }}

            let userDisplay = message;

            if (selectedFiles.length > 0) {{
                const names = selectedFiles.map(file => file.name).join(", ");
                userDisplay = message
                    ? `${{message}}\n\nAttached: ${{names}}`
                    : `Attached: ${{names}}`;
            }}

            addMessage(userDisplay, "user");
            input.value = "";
            sendButton.disabled = true;
            resetButton.disabled = true;
            fileInput.disabled = true;

            try {{
                let response;

                if (selectedFiles.length > 0) {{
                    const body = new FormData();
                    body.append("message", message);

                    if (previousInteractionId) {{
                        body.append(
                            "previous_interaction_id",
                            previousInteractionId
                        );
                    }}

                    selectedFiles.forEach(file => body.append("files", file));

                    response = await fetch("/api/chat/files", {{
                        method: "POST",
                        body,
                    }});
                }} else {{
                    response = await fetch("/api/chat", {{
                        method: "POST",
                        headers: {{ "Content-Type": "application/json" }},
                        body: JSON.stringify({{
                            message,
                            previous_interaction_id: previousInteractionId,
                        }}),
                    }});
                }}

                const data = await response.json();

                if (!response.ok) {{
                    throw new Error(data.detail || "The request failed.");
                }}

                previousInteractionId = data.interaction_id;
                addMessage(data.reply, "bot");

            }} catch (error) {{
                addMessage(error.message || "The request failed.", "error");

            }} finally {{
                fileInput.value = "";
                updateFileStatus();
                fileInput.disabled = false;
                sendButton.disabled = false;
                resetButton.disabled = false;
                input.focus();
            }}
        }});

        resetButton.addEventListener("click", () => {{
            previousInteractionId = null;
            input.value = "";
            fileInput.value = "";
            updateFileStatus();
            messages.innerHTML =
                '<div class="message bot">New conversation started.</div>';
            input.focus();
        }});
    </script>
</body>
</html>
"""


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        reload=True,
    )
