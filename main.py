import io
import json
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings
from elevenlabs.client import ElevenLabs
try:
    import httpx  # type: ignore
except Exception:  # pragma: no cover
    httpx = None  # Fallback if not present


class Settings(BaseSettings):
    ELEVENLABS_API_KEY: str
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_DEPLOYMENT: Optional[str] = None
    MS_CLIENT_ID: Optional[str] = None
    MS_TENANT_ID: Optional[str] = "common"
    MS_REDIRECT_URI: Optional[str] = None

    class Config:
        env_file = ".env"


settings = Settings()

app = FastAPI(title="Meeting Transcriber", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"ok": True}


@app.get("/debug/key")
def debug_key():
    key = settings.ELEVENLABS_API_KEY or ""
    return {"length": len(key), "suffix": key[-6:] if key else None}


@app.get("/config/ms")
def ms_config():
    # Minimal config for msal-browser on localhost
    client_id = settings.MS_CLIENT_ID or ""
    tenant = (settings.MS_TENANT_ID or "common").strip()
    redirect_uri = settings.MS_REDIRECT_URI or "http://localhost:8010/"
    authority = f"https://login.microsoftonline.com/{tenant}"
    scopes = ["User.Read", "Calendars.ReadWrite", "offline_access"]
    enabled = bool(client_id)
    return {
        "enabled": enabled,
        "clientId": client_id,
        "authority": authority,
        "redirectUri": redirect_uri,
        "scopes": scopes,
    }


@app.post("/api/transcribe")
async def transcribe(
    file: UploadFile = File(...),
    diarize: bool = Form(True),
    tag_audio_events: bool = Form(True),
    language_code: Optional[str] = Form(None),
):
    if not settings.ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="Missing ElevenLabs API key")

    try:
        # Read the uploaded file into memory
        data: bytes = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        # ElevenLabs Python SDK expects a file-like object
        file_like = io.BytesIO(data)
        file_like.name = file.filename or "audio.wav"  # provide a name hint

        client = ElevenLabs(api_key=settings.ELEVENLABS_API_KEY)

        kwargs = {
            "file": file_like,
            "model_id": "scribe_v1",
            "diarize": diarize,
            "tag_audio_events": tag_audio_events,
        }
        if language_code:
            kwargs["language_code"] = language_code

        result = client.speech_to_text.convert(**kwargs)

        # Convert SDK model to JSON-serializable dict
        if isinstance(result, dict):
            payload = result
        elif hasattr(result, "model_dump"):
            payload = result.model_dump()  # pydantic v2
        elif hasattr(result, "to_dict"):
            payload = result.to_dict()
        elif hasattr(result, "json"):
            try:
                payload = json.loads(result.json())
            except Exception:
                payload = json.loads(str(result))
        else:
            payload = json.loads(json.dumps(result, default=lambda o: getattr(o, "__dict__", str(o))))

        return JSONResponse(content=payload)

    except HTTPException:
        raise
    except Exception as exc:  # return clean error to client with upstream status when possible
        # If the ElevenLabs SDK raised an HTTP error (via httpx), preserve status and JSON body
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", 500)
            try:
                detail = response.json()
            except Exception:
                try:
                    detail = response.text
                except Exception:
                    detail = str(exc)
            raise HTTPException(status_code=status_code, detail=detail)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/summarize")
async def summarize(payload: dict):
    """Summarize a transcript using Azure OpenAI Chat Completions.

    Expected payload shape: { text: str, words: list, language_code?: str, speaker_names?: {id: name} }
    Returns: { summary_text: str }
    """
    if not settings.AZURE_OPENAI_ENDPOINT or not settings.AZURE_OPENAI_API_KEY:
        raise HTTPException(status_code=501, detail="Summarization not configured")

    if httpx is None:
        raise HTTPException(status_code=500, detail="httpx not available on server")

    text = payload.get("text") or ""
    words = payload.get("words") or []
    language_code = payload.get("language_code") or ""
    speaker_names = payload.get("speaker_names") or {}

    if not text and not words:
        raise HTTPException(status_code=400, detail="Missing transcript content")

    # Build a brief, deterministic prompt with strict language control
    system_prompt = (
        "You are an assistant that produces high-quality, factual meeting summaries. "
        "If a target language is specified via language_code, you MUST write the ENTIRE output in that target language. "
        "For 'nor' use Norwegian Bokmål. Do not use Swedish unless the transcript is Swedish. "
        "If no language_code is given, use the transcript language.\n\n"
        "Output structure (use headings appropriate to the target language):\n"
        "- Summary: 4–8 sentences that cover purpose, key points, and outcomes\n"
        "- Action items: 3–7 bullets (assign owners if inferable)\n"
        "- Decisions: 2–5 bullets (if any)\n\n"
        "Guidelines: Be clear and faithful to the transcript. Ignore beeps and filler words. Do not invent facts."
    )

    # Compact speaker map for context
    if speaker_names and isinstance(speaker_names, dict):
        name_lines = [f"{sid}: {speaker_names[sid]}" for sid in sorted(speaker_names.keys()) if speaker_names[sid]]
        speaker_block = "\n".join(name_lines)
    else:
        speaker_block = ""

    # If text missing, reconstruct from words
    if not text and isinstance(words, list):
        try:
            text = "".join([w.get("text", " ") for w in words])
        except Exception:
            text = ""

    language_hint = (
        f"Target language (ISO-639 code): {language_code}. Write the ENTIRE output in this target language."
        if language_code else ""
    )
    user_prompt = (
        f"{language_hint}\n" \
        f"Speaker names (if provided):\n{speaker_block}\n\n" \
        f"Transcript:\n{text}"
    ).strip()

    # Helper: call Azure chat completions
    async def azure_chat(messages: list[dict]) -> str:
        headers = {"Content-Type": "application/json", "api-key": settings.AZURE_OPENAI_API_KEY}
        body = {"messages": messages, "temperature": 0.2, "max_tokens": 1000, "top_p": 1}
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(settings.AZURE_OPENAI_ENDPOINT, headers=headers, json=body)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)
        data = resp.json()
        return ((data.get("choices") or [{}])[0].get("message", {}).get("content", ""))

    # If text missing, reconstruct from words
    if not text and isinstance(words, list):
        try:
            text = "".join([w.get("text", " ") for w in words])
        except Exception:
            text = ""

    # Chunk if too long (heuristic by characters)
    def split_into_chunks(source: str, target_size: int = 7000) -> list[str]:
        if not source or len(source) <= target_size:
            return [source] if source else []
        # Prefer to split on sentence boundaries
        import re
        sentences = re.split(r"(?<=[.!?])\s+", source)
        chunks: list[str] = []
        current = []
        current_len = 0
        for s in sentences:
            sl = len(s) + 1
            if current_len + sl > target_size and current:
                chunks.append(" ".join(current).strip())
                current = [s]
                current_len = sl
            else:
                current.append(s)
                current_len += sl
        if current:
            chunks.append(" ".join(current).strip())
        return chunks

    try:
        chunks = split_into_chunks(text, 7000)

        if len(chunks) <= 1:
            # Single-shot
            messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
            content = await azure_chat(messages)
            return {"summary_text": content}

        # Map-Reduce: summarize each chunk, then synthesize
        map_summaries: list[str] = []
        total = len(chunks)
        for idx, chunk in enumerate(chunks, start=1):
            map_user = (
                f"{language_hint}\n"
                f"Speaker names (if provided):\n{speaker_block}\n\n"
                f"You will summarize chunk {idx} of {total}.\n"
                f"Chunk transcript:\n{chunk}"
            ).strip()
            map_messages = [
                {"role": "system", "content": system_prompt + "\nSummarize ONLY this chunk."},
                {"role": "user", "content": map_user},
            ]
            map_summaries.append(await azure_chat(map_messages))

        reduce_user = (
            f"{language_hint}\nSpeaker names (if provided):\n{speaker_block}\n\n"
            f"Synthesize a final meeting summary from these chunk summaries (ordered):\n\n"
            + "\n\n".join([f"Chunk {i+1} of {total}:\n{ms}" for i, ms in enumerate(map_summaries)])
        )
        reduce_messages = [
            {"role": "system", "content": system_prompt + "\nCreate ONE cohesive summary for the full meeting."},
            {"role": "user", "content": reduce_user},
        ]
        final_summary = await azure_chat(reduce_messages)
        return {"summary_text": final_summary}
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# Serve a simple static UI from / (index.html lives in ./static)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
