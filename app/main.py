"""
FastAPI application — HTTP transport layer for the PR Validation LangGraph agent.

Routes:
  GET  /           → serves static/index.html
  POST /validate   → runs the LangGraph agent and returns validation results
  GET  /health     → health check
"""
import logging
import mimetypes
import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.graph import graph
from app.state import PRValidationState, UploadedFile

logger = logging.getLogger(__name__)

app = FastAPI(title="PR Attachment Validation Agent", version="1.0.0")

# Serve static files
STATIC_DIR = Path(__file__).parent.parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the frontend form."""
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>PR Validation Agent</h1><p>Frontend not found.</p>")


@app.get("/health")
async def health():
    settings = get_settings()
    return {
        "status": "ok",
        "gemini_enabled": settings.gemini_enabled,
        "drive_enabled": settings.drive_enabled,
        "langfuse_enabled": settings.langfuse_enabled,
    }


@app.post("/validate")
async def validate(
    pr_type: str = Form(...),
    pr_number: str = Form(...),
    item_long_text: str = Form(""),
    pr_description: str = Form(""),
    files: Optional[List[UploadFile]] = File(default=None),
    doc_types: Optional[List[str]] = Form(default=None),
):
    """
    Run the PR Validation LangGraph agent.

    Accepts multipart/form-data with:
      - pr_type, pr_number, item_long_text, pr_description (form fields)
      - files: one or more file uploads
      - doc_types: parallel list of document type labels for each uploaded file
    """
    settings = get_settings()

    # Build UploadedFile list
    uploaded: List[UploadedFile] = []
    if files:
        for i, f in enumerate(files):
            if f.filename:
                content = await f.read()
                mime = f.content_type or _guess_mime(f.filename)
                doc_type = ""
                if doc_types and i < len(doc_types):
                    doc_type = doc_types[i]
                uploaded.append({
                    "name": f.filename,
                    "content": content,
                    "mime_type": mime,
                    "doc_type": doc_type,
                })

    # Build initial LangGraph state
    initial_state: PRValidationState = {
        "pr_type": pr_type,
        "pr_number": pr_number,
        "item_long_text": item_long_text,
        "pr_description": pr_description,
        "uploaded_files": uploaded,
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }

    # Build run config — attach Langfuse if configured
    run_config = {
        "run_name": f"pr-validation-{pr_number}",
        "metadata": {"pr_type": pr_type, "env": settings.app_env},
    }

    langfuse_handler = None
    if settings.langfuse_enabled:
        try:
            import logging as _logging
            # Suppress noisy Langfuse SDK warnings (e.g. proxy/SSL connectivity errors)
            _logging.getLogger("langfuse").setLevel(_logging.ERROR)
            from langfuse.callback import CallbackHandler
            langfuse_handler = CallbackHandler(
                public_key=settings.langfuse_public_key,
                secret_key=settings.langfuse_secret_key,
                host=settings.langfuse_host,
            )
            run_config["callbacks"] = [langfuse_handler]
        except Exception as exc:
            logger.warning(f"Failed to initialise Langfuse: {exc}")

    # Run the LangGraph agent
    try:
        result_state = graph.invoke(initial_state, config=run_config)
    except Exception as exc:
        logger.exception(f"Graph invocation failed: {exc}")
        return JSONResponse(
            status_code=500,
            content={
                "messages": [{"icon": "✗", "text": f"Internal error during validation: {exc}"}],
                "overall_status": "error",
                "blocker_count": 0,
            },
        )
    finally:
        if langfuse_handler:
            try:
                langfuse_handler.flush()
            except Exception:
                pass

    # Build response
    messages = result_state.get("validation_messages", [])
    blocker_count = result_state.get("blocker_count", 0)
    overall_status = result_state.get("overall_status", "ready")

    return {
        "messages": messages,
        "overall_status": overall_status,
        "blocker_count": blocker_count,
    }


def _guess_mime(filename: str) -> str:
    mime, _ = mimetypes.guess_type(filename)
    return mime or "application/octet-stream"
