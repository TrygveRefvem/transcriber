## Meeting Transcriber (ElevenLabs STT + Azure OpenAI + Outlook)

A minimal meeting transcriber you can run locally:
- Upload or record audio/video; transcribe with ElevenLabs Scribe v1
- Get an AI summary using your Azure OpenAI deployment
- Sign in with Microsoft, list your calendar events, and save notes back to an event

Recent additions:
- “Summarize” button to (re)generate the AI summary after you enter speaker names
- No prefilled speaker names between runs (each transcription starts blank)
- Markdown-style rendering of summary headings, bullets and bold
- Meeting times shown in your local timezone

### Features
- FastAPI backend at `http://localhost:8010`
- Frontend served from `static/index.html`
- Endpoints:
  - `GET /` static UI
  - `POST /api/transcribe` multipart upload → ElevenLabs Scribe v1
  - `POST /api/summarize` AI summary via Azure OpenAI Chat Completions
  - `GET /config/ms` MSAL config (driven by env)
  - `GET /debug/key` last 6 chars of ElevenLabs key (for diagnostics)

## Prerequisites
- macOS/Linux/Windows
- Python 3.9+ with `venv`
- ElevenLabs account with Speech to Text enabled
- Azure OpenAI deployment (Chat Completions) and key
- Microsoft Entra ID (Azure AD) app registration for calendar integration

## Setup

### 1) Create and activate a virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -V
```

### 2) Install dependencies
```bash
pip install --upgrade pip wheel setuptools
pip install fastapi "uvicorn[standard]" python-multipart pydantic-settings elevenlabs httpx
```

### 3) Environment variables
Create a `.env` file in the project root:
```bash
ELEVENLABS_API_KEY=YOUR_ELEVENLABS_API_KEY

# Azure OpenAI (Chat Completions)
AZURE_OPENAI_ENDPOINT=https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT/chat/completions?api-version=2025-01-01-preview
AZURE_OPENAI_API_KEY=YOUR_AZURE_OPENAI_KEY
AZURE_OPENAI_DEPLOYMENT=YOUR_DEPLOYMENT_NAME

# Microsoft (optional, for Outlook meeting integration)
MS_CLIENT_ID=YOUR_APP_CLIENT_ID
MS_TENANT_ID=common
MS_REDIRECT_URI=http://localhost:8010/
```
Notes:
- Do not commit `.env` to git; `.gitignore` excludes it.
- If you change `.env`, restart the server to load new values.

## Run locally
```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8010 --reload
```
Open `http://localhost:8010/` in your browser.

### Typical workflow in the UI
1) Upload a file or click Start Recording → Stop
2) Enter speaker names under Speakers → click Apply names
3) Click Summarize to generate an AI summary that uses those speaker names
4) Optional: Microsoft 365 → Load my meetings → pick an event → Save notes to event

## ElevenLabs Speech-to-Text
- Uses the official Python SDK and `model_id="scribe_v1"`
- Accepts audio/video (MP3, WAV, M4A, MP4, WebM, etc.)
- Optional form fields in `/api/transcribe`:
  - `diarize` (default true)
  - `tag_audio_events` (default true)
  - `language_code` (omit for auto-detect)

Diagnostics:
```bash
curl -s http://localhost:8010/debug/key
# {"length":51,"suffix":"......"}
```
If you see 401 with `missing_permissions: speech_to_text`, ensure your ElevenLabs key/plan has STT enabled.

## Azure OpenAI (Summary)
- Backend calls your Azure OpenAI Chat Completions endpoint
- The server reads `text`, `words`, `language_code`, and optional speaker name map and returns `summary_text`
- The prompt enforces the detected transcript language (e.g., Norwegian if `nor`) and structured headings
- Token limit is set to `max_tokens: 1000` by default (see `main.py`)

Example test:
```bash
python - << 'PY'
import requests
payload={"text":"Short meeting text","words":[],"language_code":"eng"}
print(requests.post('http://localhost:8010/api/summarize', json=payload).json())
PY
```

## Microsoft 365 (Optional)

### Create an Entra ID app registration
1. Azure Portal → Entra ID → App registrations → New registration
   - Name: Transcriber (local)
   - Supported account types: per your needs (e.g., multiple organizations)
   - Redirect URI: Single-page application → `http://localhost:8010/`
2. Authentication
   - Ensure SPA with `http://localhost:8010/` is listed
3. API permissions → Add a permission → Microsoft Graph → Delegated permissions
   - Add: `User.Read` and `Calendars.ReadWrite`
   - Grant admin consent if available (otherwise consent during sign-in)
4. Copy Application (client) ID → set `MS_CLIENT_ID` in `.env`

### Use in the app
- Ensure `.env` has `MS_CLIENT_ID`, `MS_TENANT_ID=common`, `MS_REDIRECT_URI=http://localhost:8010/`
- Restart the server
- In the UI (Microsoft 365 box): Sign in → Load my meetings → pick event → Save notes to event

Notes:
- The UI uses `calendarView` in a window (-1 day, +30 days)
- Times are rendered in your local timezone

## Troubleshooting

### Transcribe returns 401 missing_permissions
- Your ElevenLabs API key does not have Speech to Text permission or your plan doesn’t include it
- Create a new key with STT enabled and update `.env`, then restart the server

### POST /api/transcribe was Method Not Allowed (405)
- Static routes must not intercept the API routes. In this app, the static mount is at the end of `main.py` so API routes match first

### The server picked up the wrong API key after reload
- The dev reloader may read old env values. Confirm via `GET /debug/key`
- Restart with the correct `.env` values loaded

### Meeting list shows wrong times or UTC
- The frontend now parses `start.dateTime` with `start.timeZone` and renders in your local timezone. Refresh and click “Load my meetings”.

### Azure summary language seems wrong
- The prompt forces output to the detected `language_code` when present. If needed, pass a preferred language code in the transcribe UI, or adjust the prompt in `main.py`.

### Summary ignores edited speaker names
- After transcription, enter names under Speakers → click Apply names → click Summarize to re-run the LLM with your current names.

## Project structure
```
main.py               # FastAPI app (STT + summarizer + MS config)
static/index.html     # UI: upload/record, speakers, summary, Outlook integration
.gitignore            # excludes .venv/.env/etc
README.md             # this file
```

## Security & production notes
- Keep `.env` out of version control
- Use HTTPS and proper auth in any deployed environment
- Consider rate limiting, logging, and monitoring
- For production, run `uvicorn` without `--reload`, fronted by a reverse proxy

## License
MIT (or your preference)


