import io
import json
import os
import re
import logging
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic_settings import BaseSettings, SettingsConfigDict
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
    NOTION_API_KEY: Optional[str] = None
    NOTION_DATABASE_ID: Optional[str] = None

    # Ensure .env is loaded in pydantic-settings v2
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")


settings = Settings()

# Logger setup
logger = logging.getLogger("transcriber")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
logger.setLevel(logging.INFO)

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


@app.get("/config/notion")
def notion_config():
    # Robustly detect Notion config from settings, env vars, or .env
    api_key, db_id = _get_notion_credentials()
    enabled = bool(api_key and db_id)
    return {
        "enabled": enabled,
        "hasApiKey": bool(api_key),
        "hasDatabaseId": bool(db_id),
    }


@app.get("/debug/notion")
def debug_notion():
    api_key, db_id = _get_notion_credentials()
    norm_id = normalize_notion_id(db_id or "") if db_id else None
    return {
        "api_key_length": len(api_key or ""),
        "api_key_suffix": (api_key[-6:] if api_key else None),
        "database_id_length": len(db_id or ""),
        "database_id_suffix": (db_id[-6:] if db_id else None),
        "normalized_database_id": norm_id,
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
    calendar_context = payload.get("calendar_context") or ""

    if not text and not words:
        raise HTTPException(status_code=400, detail="Missing transcript content")

    # Build a brief, deterministic prompt with strict language control
    system_prompt = (
        "You are an assistant that produces high-quality, factual meeting summaries. "
        "If a target language is specified via language_code, you MUST write the ENTIRE output in that target language. "
        "For 'nor' use Norwegian Bokmål. Do not use Swedish unless the transcript is Swedish. "
        "Do not use Danish unless language_code is 'dan'. "
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
    calendar_block = (
        f"Meeting details from calendar (agenda/topic/description if present):\n{calendar_context}\n\n"
        if calendar_context else ""
    )
    user_prompt = (
        f"{language_hint}\n" \
        f"Speaker names (if provided):\n{speaker_block}\n\n" \
        f"{calendar_block}" \
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


@app.post("/api/notion/export")
async def notion_export(payload: dict):
    """Create a Notion page with the summary and formatted transcript.

    Expected payload: { title?: str, audio_url?: str, summary_text: str, formatted_html?: str }
    """
    logger.info("Notion export: request received")
    api_key, db_id = _get_notion_credentials()
    if not api_key or not db_id:
        logger.warning("Notion export: Not configured (api_key_len=%s, db_id_len=%s)", len(api_key or ""), len(db_id or ""))
        raise HTTPException(status_code=501, detail="Notion not configured")
    if httpx is None:
        raise HTTPException(status_code=500, detail="httpx not available on server")

    title = (payload.get("title") or "Audio Summary").strip() or "Audio Summary"
    audio_url = payload.get("audio_url") or ""
    summary_text = payload.get("summary_text") or ""
    formatted_html = payload.get("formatted_html") or ""

    if not summary_text:
        raise HTTPException(status_code=400, detail="Missing summary_text")

    logger.info(
        "Notion export: payload meta title_len=%s summary_len=%s formatted_len=%s audio_url_set=%s",
        len(title), len(summary_text), len(formatted_html), bool(audio_url),
    )

    # Build Notion page create payload
    # Properties: Only Name (Title) to avoid schema coupling
    properties = {
        "Name": {
            "title": [{"text": {"content": title[:200]}}]
        }
    }

    children = [
        {
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "AI Summary"}}]}
        },
        {
            "object": "block",
            "type": "paragraph",
            "paragraph": {
                "rich_text": [{"type": "text", "text": {"content": summary_text[:19500]}}]
            }
        }
    ]
    if formatted_html:
        # Put transcript as a code block (HTML) for fidelity; Notion markdown/html import is limited via API.
        children.append({
            "object": "block",
            "type": "heading_2",
            "heading_2": {"rich_text": [{"type": "text", "text": {"content": "Transcript"}}]}
        })
        children.append({
            "object": "block",
            "type": "code",
            "code": {
                "rich_text": [{"type": "text", "text": {"content": formatted_html[:19500]}}],
                "language": "html"
            }
        })

    normalized_db_id = normalize_notion_id(db_id)
    logger.info("Notion export: using database_id raw=%s normalized=%s", db_id, normalized_db_id)
    body = {
        "parent": {"type": "database_id", "database_id": normalized_db_id},
        "properties": properties,
        "children": children,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Notion-Version": "2022-06-28",
        "Content-Type": "application/json",
    }

    try:
        async with httpx.AsyncClient(timeout=60) as client:
            resp = await client.post("https://api.notion.com/v1/pages", headers=headers, json=body)
        logger.info("Notion export: response status=%s", resp.status_code)
        if resp.status_code >= 400:
            try:
                detail = resp.json()
            except Exception:
                detail = resp.text
            # Truncate to avoid huge logs
            log_excerpt = (detail if isinstance(detail, str) else json.dumps(detail))[:500]
            logger.warning("Notion export: error body=%s", log_excerpt)
            raise HTTPException(status_code=resp.status_code, detail=detail)
        j = resp.json()
        logger.info("Notion export: success page_id=%s", (j.get("id") if isinstance(j, dict) else None))
        return j
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Notion export: unexpected error")
        raise HTTPException(status_code=500, detail=str(exc))


def _get_notion_credentials() -> tuple[Optional[str], Optional[str]]:
    """Resolve Notion API key and Database ID from settings, env, or .env as fallback.

    Returns (api_key, database_id) where either can be None if unresolved.
    """
    api_key = settings.NOTION_API_KEY or os.getenv("NOTION_API_KEY")
    db_id = settings.NOTION_DATABASE_ID or os.getenv("NOTION_DATABASE_ID")
    if api_key and db_id:
        return api_key.strip() or None, db_id.strip() or None

    # Fallback: read .env manually
    try:
        env_path = ".env"
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8-sig") as fh:
                for raw in fh:
                    line = raw.rstrip("\r\n")
                    if not line or line.lstrip().startswith("#"):
                        continue
                    m1 = re.match(r"^\s*NOTION_API_KEY\s*=\s*(.*)$", line)
                    if m1 and not api_key:
                        api_key = m1.group(1).strip()
                        continue
                    m2 = re.match(r"^\s*NOTION_DATABASE_ID\s*=\s*(.*)$", line)
                    if m2 and not db_id:
                        db_id = m2.group(1).strip()
    except Exception:
        # Ignore .env read errors; just return what we have
        pass

    api_key = (api_key or "").strip() or None
    db_id = (db_id or "").strip() or None
    return api_key, db_id


def normalize_notion_id(raw: str) -> str:
    """Return database ID in dashed UUID format if possible.

    Accepts dashed or undashed; strips non-hex, enforces 32 hex chars,
    then inserts dashes as 8-4-4-4-12.
    """
    if not raw:
        return raw
    hex_only = re.sub(r"[^0-9a-fA-F]", "", raw).lower()
    if len(hex_only) != 32:
        return raw
    return f"{hex_only[0:8]}-{hex_only[8:12]}-{hex_only[12:16]}-{hex_only[16:20]}-{hex_only[20:32]}"


# Serve a simple static UI from / (index.html lives in ./static)
app.mount("/", StaticFiles(directory="static", html=True), name="static")
