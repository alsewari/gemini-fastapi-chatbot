import os
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from google import genai
from google.genai import errors
from pydantic import BaseModel, Field

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is missing. Add it to your .env file before starting the API."
    )

client = genai.Client(api_key=API_KEY)

app = FastAPI(
    title="Gemini Chat API",
    description="A FastAPI chatbot powered by the Google Gemini API.",
    version="1.0.0",
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL}


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    create_args: dict[str, Any] = {
        "model": MODEL,
        "input": request.message.strip(),
    }

    if request.previous_interaction_id:
        create_args["previous_interaction_id"] = request.previous_interaction_id

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
        )

    except errors.ServerError as exc:
        # Typical example: 503 UNAVAILABLE when the model is under high demand.
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
            detail="Gemini rejected the request. Check the API key, model name, or request.",
        ) from exc

    except HTTPException:
        raise

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="An unexpected server error occurred.",
        ) from exc


@app.get("/", response_class=HTMLResponse)
def home() -> str:
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0" />
    <title>Gemini FastAPI Chat</title>
    <style>
        * { box-sizing: border-box; }
        body {
            margin: 0;
            font-family: Arial, sans-serif;
            background: #f4f6f8;
            color: #1f2937;
        }
        .app {
            width: min(900px, 94vw);
            margin: 32px auto;
            background: white;
            border-radius: 14px;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
            overflow: hidden;
        }
        header {
            padding: 20px 24px;
            border-bottom: 1px solid #e5e7eb;
        }
        header h1 { margin: 0 0 6px; font-size: 24px; }
        header p { margin: 0; color: #6b7280; }
        #messages {
            height: 500px;
            overflow-y: auto;
            padding: 22px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }
        .message {
            max-width: 80%;
            padding: 12px 14px;
            border-radius: 12px;
            white-space: pre-wrap;
            line-height: 1.45;
        }
        .user {
            align-self: flex-end;
            background: #2563eb;
            color: white;
        }
        .bot {
            align-self: flex-start;
            background: #eef2f7;
        }
        .error {
            align-self: flex-start;
            background: #fee2e2;
            color: #991b1b;
        }
        form {
            display: grid;
            grid-template-columns: 1fr auto auto;
            gap: 10px;
            padding: 18px;
            border-top: 1px solid #e5e7eb;
        }
        input {
            width: 100%;
            padding: 12px;
            border: 1px solid #cbd5e1;
            border-radius: 9px;
            font-size: 16px;
        }
        button {
            border: 0;
            border-radius: 9px;
            padding: 0 18px;
            cursor: pointer;
            font-weight: 700;
        }
        #sendButton { background: #2563eb; color: white; }
        #resetButton { background: #e5e7eb; color: #111827; }
        button:disabled { opacity: 0.6; cursor: not-allowed; }
    </style>
</head>
<body>
    <main class="app">
        <header>
            <h1>Gemini FastAPI Chat</h1>
            <p>Messages are sent to your FastAPI backend, not directly from the browser to Gemini.</p>
        </header>

        <section id="messages">
            <div class="message bot">Hello! Ask me a question.</div>
        </section>

        <form id="chatForm">
            <input id="messageInput" placeholder="Type your message..." autocomplete="off" required />
            <button id="sendButton" type="submit">Send</button>
            <button id="resetButton" type="button">New chat</button>
        </form>
    </main>

    <script>
        const form = document.getElementById("chatForm");
        const input = document.getElementById("messageInput");
        const messages = document.getElementById("messages");
        const sendButton = document.getElementById("sendButton");
        const resetButton = document.getElementById("resetButton");

        let previousInteractionId = null;

        function addMessage(text, type) {
            const element = document.createElement("div");
            element.className = `message ${type}`;
            element.textContent = text;
            messages.appendChild(element);
            messages.scrollTop = messages.scrollHeight;
        }

        form.addEventListener("submit", async (event) => {
            event.preventDefault();

            const message = input.value.trim();
            if (!message) return;

            addMessage(message, "user");
            input.value = "";
            sendButton.disabled = true;

            try {
                const response = await fetch("/api/chat", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        message,
                        previous_interaction_id: previousInteractionId,
                    }),
                });

                const data = await response.json();

                if (!response.ok) {
                    throw new Error(data.detail || "The request failed.");
                }

                previousInteractionId = data.interaction_id;
                addMessage(data.reply, "bot");
            } catch (error) {
                addMessage(error.message, "error");
            } finally {
                sendButton.disabled = false;
                input.focus();
            }
        });

        resetButton.addEventListener("click", () => {
            previousInteractionId = null;
            messages.innerHTML = '<div class="message bot">New conversation started.</div>';
            input.focus();
        });
    </script>
</body>
</html>
"""
