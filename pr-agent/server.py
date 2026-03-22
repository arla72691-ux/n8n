"""FastAPI PR Attachment Validation Agent — with LangGraph routing + Langfuse observability."""
import os
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

load_dotenv()

# ── Langfuse + OpenTelemetry setup (must happen before any tracer.get_tracer) ─
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

_langfuse_enabled = bool(
    os.environ.get("LANGFUSE_SECRET_KEY") and os.environ.get("LANGFUSE_PUBLIC_KEY")
)

if _langfuse_enabled:
    from langfuse.opentelemetry import LangfuseExporter
    _provider = TracerProvider()
    _provider.add_span_processor(BatchSpanProcessor(LangfuseExporter()))
    trace.set_tracer_provider(_provider)

tracer = trace.get_tracer(__name__)

# ── LangGraph graph import (comes after OTel init so graph nodes share provider) ─
from graph import pr_validation_graph

app = FastAPI(title="PR Validation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FORM_HTML = Path(__file__).parent / "form.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    return FORM_HTML.read_text()


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "langfuse": "enabled" if _langfuse_enabled else "disabled (set LANGFUSE_SECRET_KEY + LANGFUSE_PUBLIC_KEY)",
    }


@app.post("/validate")
async def validate(
    prType: Annotated[str, Form()],
    prNumber: Annotated[str, Form()],
    itemLongText: Annotated[Optional[str], Form()] = "",
    docType0: Annotated[Optional[str], Form()] = None,
    docType1: Annotated[Optional[str], Form()] = None,
    docType2: Annotated[Optional[str], Form()] = None,
    file0: Annotated[Optional[UploadFile], File()] = None,
    file1: Annotated[Optional[UploadFile], File()] = None,
    file2: Annotated[Optional[UploadFile], File()] = None,
):
    # Collect uploaded files as (docType, bytes, mimeType) tuples
    files: list[tuple[str, bytes, str]] = []
    for i, (upload, doc_type) in enumerate(
        [(file0, docType0), (file1, docType1), (file2, docType2)]
    ):
        if upload and upload.filename:
            content = await upload.read()
            mime = upload.content_type or "application/octet-stream"
            if upload.filename.lower().endswith(".pdf") and mime == "application/octet-stream":
                mime = "application/pdf"
            dt = doc_type or f"file{i}"
            files.append((dt, content, mime))

    pr_type = prType.strip()

    with tracer.start_as_current_span("pr_validation") as span:
        span.set_attribute("pr.type", pr_type)
        span.set_attribute("pr.number", prNumber)
        span.set_attribute("pr.files_count", len(files))

        try:
            state = await pr_validation_graph.ainvoke({
                "pr_type": pr_type,
                "pr_number": prNumber,
                "item_long_text": itemLongText or "",
                "files": files,
                "html_output": "",
                "is_valid": False,
                "error": None,
            })
        except ValueError as e:
            return JSONResponse({"error": str(e)}, status_code=400)

        span.set_attribute("pr.is_valid", state["is_valid"])

    return {
        "responseHtml": state["html_output"],
        "allValid": state["is_valid"],
        "prType": pr_type,
        "prNumber": prNumber,
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
