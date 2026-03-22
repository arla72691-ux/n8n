"""FastAPI PR Attachment Validation Agent."""
import os
from pathlib import Path
from typing import Annotated, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

load_dotenv()

import validators as val

app = FastAPI(title="PR Validation Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve the form HTML at root
FORM_HTML = Path(__file__).parent / "form.html"


@app.get("/", response_class=HTMLResponse)
async def root():
    return FORM_HTML.read_text()


@app.get("/health")
async def health():
    return {"status": "ok"}


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
            # Normalise common PDF/image types
            if upload.filename.lower().endswith(".pdf") and mime == "application/octet-stream":
                mime = "application/pdf"
            dt = doc_type or f"file{i}"
            files.append((dt, content, mime))

    pr_type = prType.strip()

    if "Non-APD" in pr_type:
        html, ok = await val.validate_non_apd(prNumber)
    elif "APD" in pr_type or "APD" in pr_type:
        html, ok = await val.validate_apd(prNumber, itemLongText or "", files)
    elif "FTP" in pr_type:
        html, ok = await val.validate_ftp(prNumber, files)
    elif "PAC" in pr_type:
        html, ok = await val.validate_pac(prNumber, files)
    elif "Service" in pr_type:
        html, ok = await val.validate_service(prNumber, files)
    else:
        return JSONResponse(
            {"error": f"Unknown PR type: {pr_type}"}, status_code=400
        )

    return {"responseHtml": html, "allValid": ok, "prType": pr_type, "prNumber": prNumber}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
