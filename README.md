# Gemini FastAPI Chat

A small FastAPI application that sends messages to the Google Gemini API and preserves multi-turn context using `previous_interaction_id`.

## 1. Install packages

```cmd
"C:\Program Files\Python313\python.exe" -m pip install -r requirements.txt
```

## 2. Create `.env`

Copy `.env.example` and rename the copy to `.env`.

```env
GEMINI_API_KEY=YOUR_REAL_API_KEY
GEMINI_MODEL=gemini-3.5-flash-lite
```

Do not commit `.env` to GitHub.

## 3. Run the API

```cmd
"C:\Program Files\Python313\python.exe" -m uvicorn main:app --reload --port 8000
```

## 4. Open the application

- Chat interface: http://127.0.0.1:8000
- Swagger UI: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## Example Swagger request

```json
{
  "message": "Explain AI in simple words.",
  "previous_interaction_id": null
}
```

Use the returned `interaction_id` as `previous_interaction_id` in the next request to continue the same conversation.
