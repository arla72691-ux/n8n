"""Gemini 1.5 Pro multimodal client — uses REST API directly (no SDK)."""
import base64
import json
import os
import re
from typing import Optional

import requests
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.environ["GEMINI_API_KEY"]
_BASE = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"
_HEADERS = {"Content-Type": "application/json"}


def _call(parts: list) -> dict:
    """Call Gemini with a list of content parts and return parsed JSON response."""
    body = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1},
    }
    resp = requests.post(
        f"{_BASE}?key={_API_KEY}",
        headers=_HEADERS,
        json=body,
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["candidates"][0]["content"]["parts"][0]["text"]
    return _extract_json(text)


def _extract_json(text: str) -> dict:
    text = text.strip()
    m = re.search(r"```(?:json)?\s*([\s\S]+?)\s*```", text)
    if m:
        text = m.group(1)
    return json.loads(text)


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def _inline(mime: str, data: bytes) -> dict:
    return {"inline_data": {"mime_type": mime, "data": _b64(data)}}


def validate_drawing(
    drawing_number: str,
    revision: str,
    material: Optional[str],
    file_bytes: bytes,
    mime_type: str,
) -> dict:
    material_note = f"Expected material: {material}." if material else "No material specified."
    prompt = (
        f"You are a technical drawing validator. Examine this engineering drawing and "
        f"return ONLY a JSON object (no markdown fences) with these exact fields:\n"
        f"  drawingNumberVisible (bool)\n"
        f"  drawingNumberMatch (bool): does it match '{drawing_number}'?\n"
        f"  revisionVisible (bool)\n"
        f"  revisionMatch (bool): does it match '{revision}' "
        f"(treat 0, 00, 'NO REVISION', blank as equivalent)?\n"
        f"  approvalStampPresent (bool)\n"
        f"  isLegible (bool)\n"
        f"  materialMentioned (string or null): material spec visible on drawing\n"
        f"  issues (array of strings)\n"
        f"{material_note}"
    )
    return _call([{"text": prompt}, _inline(mime_type, file_bytes)])


def validate_costing_sheet(file_bytes: bytes, mime_type: str) -> dict:
    prompt = (
        "Examine this costing/quotation document and return ONLY a JSON object (no markdown) with:\n"
        "  hasCostBreakdown (bool)\n"
        "  hasValidityDate (bool)\n"
        "  validityDate (string or null)\n"
        "  isValidityExpired (bool): older than 6 months from 2026-03-22?\n"
        "  hasAuthorisingSignature (bool)\n"
        "  issues (array of strings)"
    )
    return _call([{"text": prompt}, _inline(mime_type, file_bytes)])


def validate_pac_certificate(file_bytes: bytes, mime_type: str) -> dict:
    prompt = (
        "Examine this PAC/accreditation certificate and return ONLY a JSON object (no markdown) with:\n"
        "  certNumber (string or null)\n"
        "  vendorName (string or null)\n"
        "  expiryDate (string or null)\n"
        "  isExpired (bool): expired as of 2026-03-22?\n"
        "  expiresWithin30Days (bool)\n"
        "  hasSignature (bool)\n"
        "  scopeDescription (string or null)\n"
        "  issues (array of strings)"
    )
    return _call([{"text": prompt}, _inline(mime_type, file_bytes)])


def validate_service_documents(
    scope_bytes: bytes, scope_mime: str,
    jsa_bytes: bytes, jsa_mime: str,
    skillset_bytes: bytes, skillset_mime: str,
) -> dict:
    prompt = (
        "Three documents: (0) Scope of Work, (1) Job Safety Analysis, (2) Technical Skill-Set. "
        "Return ONLY a JSON object (no markdown) with:\n"
        "  scopeOfWork: { hasDeliverables (bool), hasTimeline (bool), hasAcceptanceCriteria (bool), issues[] }\n"
        "  jsa: { identifiesHazards (bool), hasControlMeasures (bool), hasSafetySignature (bool), issues[] }\n"
        "  technicalSkillSet: { listsQualifications (bool), issues[] }\n"
        "  crossCheck: { consistent (bool), inconsistencies[] }"
    )
    return _call([
        {"text": prompt},
        _inline(scope_mime, scope_bytes),
        _inline(jsa_mime, jsa_bytes),
        _inline(skillset_mime, skillset_bytes),
    ])
