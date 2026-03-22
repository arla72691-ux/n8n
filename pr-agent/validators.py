"""PR type validators — each returns (responseHtml: str, allValid: bool)."""
import re
from typing import Optional
from fastapi import UploadFile
import requests

import gemini_client


def _gemini_available() -> bool:
    """Quick check if Gemini API is reachable."""
    try:
        import os
        key = os.environ.get("GEMINI_API_KEY", "")
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent?key={key}",
            json={"contents": [{"parts": [{"text": "ping"}]}]},
            headers={"Content-Type": "application/json"},
            timeout=10,
        )
        return r.status_code == 200
    except Exception:
        return False


_GEMINI_OK: bool | None = None  # cached after first check


def _check_gemini() -> bool:
    global _GEMINI_OK
    if _GEMINI_OK is None:
        _GEMINI_OK = _gemini_available()
    return _GEMINI_OK


# ── HTML helpers ─────────────────────────────────────────────────────────────

def _badge(status: str) -> str:
    cls = {"pass": "pass", "fail": "fail", "warn": "warn"}.get(status, "warn")
    labels = {"pass": "PASS", "fail": "FAIL", "warn": "WARNING"}
    return f'<span class="badge {cls}">{labels.get(status, status.upper())}</span>'


def _chk(ok: bool, text: str) -> str:
    cls = "ok" if ok else "fail"
    return f'<li class="{cls}">{text}</li>'


def _issues_html(issues: list) -> str:
    if not issues:
        return ""
    items = "".join(f"<li>{i}</li>" for i in issues)
    return f'<ul class="iss">{items}</ul>'


def _verdict(all_valid: bool, blockers: int = 0) -> str:
    if all_valid:
        return '<p class="verdict pass">✓ PR is ready to proceed</p>'
    return f'<p class="verdict fail">✗ PR has {blockers} blocker(s) — resolve before submission</p>'


# ── Non-APD ──────────────────────────────────────────────────────────────────

async def validate_non_apd(pr_number: str) -> tuple[str, bool]:
    html = (
        '<div class="vr-block">'
        '<h3>Supply PR (Non-APD)</h3>'
        f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
        '<p style="margin:10px 0">Non-APD Supply PRs do not require drawing attachments. '
        'Ensure the item description accurately matches the catalogue item and that '
        'unit of measure and quantity are correct.</p>'
        '<ul class="chk">'
        '<li class="ok">No drawing attachment required</li>'
        '<li class="ok">Proceed with standard procurement approval</li>'
        '</ul>'
        + _verdict(True) +
        '</div>'
    )
    return html, True


# ── APD ──────────────────────────────────────────────────────────────────────

def _parse_item_long_text(text: str) -> list[dict]:
    """Parse APD item long text lines into drawing objects."""
    def normalize_rev(r: str) -> str:
        r = r.strip().upper()
        return "0" if r in ("0", "00", "NO REVISION", "") else r

    drawings: dict[str, dict] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Handle leading prefix like "APD,"
        if "," in line:
            line = line.split(",", 1)[1]
        fields: dict[str, str] = {}
        for part in line.split(";"):
            if ":" in part:
                k, _, v = part.partition(":")
                fields[k.strip().upper()] = v.strip()

        dwg = fields.get("DRAWING NUMBER") or fields.get("DWG NO") or fields.get("DWG NUMBER")
        if not dwg:
            continue
        rev = normalize_rev(fields.get("REVISION", "0"))
        mat = fields.get("MATERIAL") or fields.get("MATERIAL SPECIFICATION")
        pos = fields.get("POSITION OR ITEM NUMBER") or fields.get("POSITION") or ""

        if dwg not in drawings:
            drawings[dwg] = {"drawingNumber": dwg, "revision": rev, "positions": [], "material": mat}
        drawings[dwg]["positions"].append(pos)
        if mat and not drawings[dwg]["material"]:
            drawings[dwg]["material"] = mat

    return list(drawings.values())


async def validate_apd(
    pr_number: str,
    item_long_text: str,
    files: list[tuple[str, bytes, str]],  # (docType, bytes, mimeType)
) -> tuple[str, bool]:
    drawings = _parse_item_long_text(item_long_text)

    if not drawings:
        html = (
            '<div class="vr-block">'
            '<h3>Supply PR (APD)</h3>'
            f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
            + _badge("fail") +
            '<p style="margin:8px 0">No drawing numbers could be parsed from the Item Long Text. '
            'Ensure the text contains <code>DRAWING NUMBER:xxxx</code> fields.</p>'
            + _verdict(False, 1) +
            '</div>'
        )
        return html, False

    # Build a lookup: drawing_number (lowercase) → uploaded file
    file_lookup: dict[str, tuple[bytes, str]] = {}
    for doc_type, fbytes, fmime in files:
        file_lookup[doc_type.lower()] = (fbytes, fmime)
        # Also index by filename-like key
        for dwg in drawings:
            if dwg["drawingNumber"].lower() in doc_type.lower():
                file_lookup[dwg["drawingNumber"].lower()] = (fbytes, fmime)

    blockers = 0
    drawing_blocks = []

    for dwg in drawings:
        dwg_num = dwg["drawingNumber"]
        rev = dwg["revision"]
        mat = dwg.get("material")
        positions = ", ".join(p for p in dwg["positions"] if p) or "—"

        # Try to find matching file
        uploaded = file_lookup.get(dwg_num.lower())
        if not uploaded and files:
            # Use the first uploaded file as fallback (single-drawing case)
            uploaded = (files[0][1], files[0][2])

        if not uploaded:
            # No file in Drive (not implemented yet) and no attachment
            blockers += 1
            drawing_blocks.append(
                f'<div class="dr fail">'
                f'<h4>Drawing {dwg_num} (Rev {rev})</h4>'
                f'<p class="st">Positions: {positions}</p>'
                + _badge("fail") +
                '<p>Drawing not found. Please attach the drawing file.</p>'
                + _issues_html(["Drawing file missing — cannot validate"])
                + '</div>'
            )
            continue

        # Validate with Gemini
        try:
            result = gemini_client.validate_drawing(dwg_num, rev, mat, uploaded[0], uploaded[1])
        except Exception as e:
            err_str = str(e)
            is_api_err = "403" in err_str or "401" in err_str or "permission" in err_str.lower()
            if is_api_err:
                drawing_blocks.append(
                    f'<div class="dr warn">'
                    f'<h4>Drawing {dwg_num} (Rev {rev})</h4>'
                    f'<p class="st">Positions: {positions}</p>'
                    + _badge("warn") +
                    '<p>⚠ Drawing file received. AI validation unavailable (Gemini API key needs '
                    '<a href="https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com" '
                    'target="_blank">Generative Language API enabled</a>). '
                    'Manual review required.</p>'
                    + _issues_html([f"File size: {len(uploaded[0])} bytes — verify drawing manually"])
                    + '</div>'
                )
            else:
                blockers += 1
                drawing_blocks.append(
                    f'<div class="dr fail">'
                    f'<h4>Drawing {dwg_num} (Rev {rev})</h4>'
                    + _badge("fail") +
                    f'<p>Validation error: {e}</p>'
                    '</div>'
                )
            continue

        is_pass = (
            result.get("drawingNumberMatch", False)
            and result.get("revisionMatch", False)
            and result.get("isLegible", True)
        )
        is_warn = not result.get("approvalStampPresent", True) and is_pass
        status = "pass" if is_pass and not is_warn else ("warn" if is_warn else "fail")
        if status == "fail":
            blockers += 1

        mat_html = ""
        if mat:
            mat_found = result.get("materialMentioned")
            mat_ok = bool(mat_found and mat.lower() in mat_found.lower())
            mat_html = f'<li class="{"ok" if mat_ok else "warn"}">Material: {mat_found or "not found"} (expected: {mat})</li>'

        drawing_blocks.append(
            f'<div class="dr {status}">'
            f'<h4>Drawing {dwg_num} (Rev {rev})</h4>'
            f'<p class="st">Positions: {positions}'
            + (f' | Material: {mat}' if mat else '') +
            '</p>'
            + _badge(status) +
            '<ul class="chk">'
            + _chk(result.get("drawingNumberVisible", False), "Drawing number visible")
            + _chk(result.get("drawingNumberMatch", False), f"Drawing number matches ({dwg_num})")
            + _chk(result.get("revisionVisible", False), "Revision block visible")
            + _chk(result.get("revisionMatch", False), f"Revision matches (Rev {rev})")
            + _chk(result.get("approvalStampPresent", False), "Approval stamp/signature present")
            + _chk(result.get("isLegible", True), "Drawing is legible")
            + mat_html +
            '</ul>'
            + _issues_html(result.get("issues", []))
            + '</div>'
        )

    all_valid = blockers == 0
    html = (
        '<div class="vr-block">'
        '<h3>Supply PR (APD)</h3>'
        f'<p class="meta">PR Number: <strong>{pr_number}</strong> | '
        f'{len(drawings)} drawing(s) checked</p>'
        + "".join(drawing_blocks)
        + _verdict(all_valid, blockers)
        + '</div>'
    )
    return html, all_valid


# ── FTP ──────────────────────────────────────────────────────────────────────

async def validate_ftp(
    pr_number: str,
    files: list[tuple[str, bytes, str]],
) -> tuple[str, bool]:
    if not files:
        html = (
            '<div class="vr-block">'
            '<h3>Supply PR (FTP — First Time Purchase)</h3>'
            f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
            + _badge("fail") +
            '<p style="margin:8px 0"><strong>Blocker:</strong> A Costing Sheet is required for FTP PRs.</p>'
            + _verdict(False, 1) +
            '</div>'
        )
        return html, False

    fbytes, fmime = files[0][1], files[0][2]
    try:
        result = gemini_client.validate_costing_sheet(fbytes, fmime)
    except Exception as e:
        err_str = str(e)
        if "403" in err_str or "401" in err_str or "permission" in err_str.lower():
            html = (
                '<div class="vr-block">'
                '<h3>Supply PR (FTP — First Time Purchase)</h3>'
                f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
                + _badge("warn") +
                '<p>⚠ Costing sheet received. AI validation unavailable (Gemini API key needs '
                '<a href="https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com" '
                'target="_blank">Generative Language API enabled</a>). '
                'Manual review required — ensure cost breakdown, validity date, and signature are present.</p>'
                + _verdict(True) +
                '</div>'
            )
            return html, True
        return (
            f'<div class="vr-block"><h3>FTP</h3>{_badge("fail")}'
            f'<p>Validation error: {e}</p>{_verdict(False, 1)}</div>',
            False,
        )

    blockers = 0
    issues_all: list[str] = result.get("issues", [])

    if not result.get("hasCostBreakdown", False):
        blockers += 1
        issues_all.append("Cost breakdown missing — required for FTP")
    if not result.get("hasAuthorisingSignature", False):
        blockers += 1
        issues_all.append("Authorising signature missing")
    if result.get("isValidityExpired", False):
        blockers += 1
        issues_all.append("Costing sheet validity has expired (older than 6 months)")

    warnings = []
    if not result.get("hasValidityDate", False):
        warnings.append("No validity date found — verify quotation is current")

    all_valid = blockers == 0
    status = "pass" if all_valid and not warnings else ("warn" if all_valid else "fail")

    html = (
        '<div class="vr-block">'
        '<h3>Supply PR (FTP — First Time Purchase)</h3>'
        f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
        + _badge(status) +
        '<ul class="chk">'
        + _chk(result.get("hasCostBreakdown", False), "Cost breakdown present")
        + _chk(result.get("hasValidityDate", False), "Validity date present")
        + _chk(not result.get("isValidityExpired", False),
               f"Validity date not expired (date: {result.get('validityDate') or 'unknown'})")
        + _chk(result.get("hasAuthorisingSignature", False), "Authorising signature present")
        + '</ul>'
        + _issues_html(issues_all)
        + ("".join(f'<p class="warn-txt">⚠ {w}</p>' for w in warnings))
        + _verdict(all_valid, blockers)
        + '</div>'
    )
    return html, all_valid


# ── PAC ──────────────────────────────────────────────────────────────────────

async def validate_pac(
    pr_number: str,
    files: list[tuple[str, bytes, str]],
) -> tuple[str, bool]:
    if not files:
        html = (
            '<div class="vr-block">'
            '<h3>PAC PR</h3>'
            f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
            + _badge("fail") +
            '<p style="margin:8px 0"><strong>Blocker:</strong> A PAC Certificate is required.</p>'
            + _verdict(False, 1) +
            '</div>'
        )
        return html, False

    fbytes, fmime = files[0][1], files[0][2]
    try:
        result = gemini_client.validate_pac_certificate(fbytes, fmime)
    except Exception as e:
        err_str = str(e)
        if "403" in err_str or "401" in err_str or "permission" in err_str.lower():
            html = (
                '<div class="vr-block">'
                '<h3>PAC PR</h3>'
                f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
                + _badge("warn") +
                '<p>⚠ PAC certificate received. AI validation unavailable (Gemini API key needs '
                '<a href="https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com" '
                'target="_blank">Generative Language API enabled</a>). '
                'Manual review required — ensure certificate is valid (not expired) and signed.</p>'
                + _verdict(True) +
                '</div>'
            )
            return html, True
        return (
            f'<div class="vr-block"><h3>PAC PR</h3>{_badge("fail")}'
            f'<p>Validation error: {e}</p>{_verdict(False, 1)}</div>',
            False,
        )

    blockers = 0
    issues_all: list[str] = result.get("issues", [])
    warnings: list[str] = []

    if result.get("isExpired", False):
        blockers += 1
        issues_all.append("Certificate is expired — cannot accept expired PAC certificate")
    if not result.get("hasSignature", False):
        blockers += 1
        issues_all.append("No authorising signature/stamp found")

    if result.get("expiresWithin30Days", False) and not result.get("isExpired", False):
        warnings.append(f"Certificate expires within 30 days ({result.get('expiryDate', 'unknown')})")

    all_valid = blockers == 0
    status = "pass" if all_valid and not warnings else ("warn" if all_valid else "fail")

    html = (
        '<div class="vr-block">'
        '<h3>PAC PR</h3>'
        f'<p class="meta">PR Number: <strong>{pr_number}</strong>'
        + (f' | Vendor: {result.get("vendorName", "unknown")}' if result.get("vendorName") else '')
        + '</p>'
        + _badge(status)
        + '<ul class="chk">'
        + _chk(bool(result.get("certNumber")), f"Certificate number: {result.get('certNumber') or 'not found'}")
        + _chk(not result.get("isExpired", True),
               f"Certificate valid (expiry: {result.get('expiryDate') or 'unknown'})")
        + _chk(result.get("hasSignature", False), "Authorising signature/stamp present")
        + '</ul>'
        + (f'<p class="notes">Scope: {result["scopeDescription"]}</p>' if result.get("scopeDescription") else '')
        + _issues_html(issues_all)
        + ("".join(f'<p class="warn-txt">⚠ {w}</p>' for w in warnings))
        + _verdict(all_valid, blockers)
        + '</div>'
    )
    return html, all_valid


# ── Service PR ────────────────────────────────────────────────────────────────

async def validate_service(
    pr_number: str,
    files: list[tuple[str, bytes, str]],  # (docType, bytes, mimeType)
) -> tuple[str, bool]:
    required_types = ["Scope of Work", "Job Safety Analysis (JSA)", "Technical Skill-Set"]
    file_map: dict[str, tuple[bytes, str]] = {}
    for doc_type, fbytes, fmime in files:
        # Match by docType key
        for rt in required_types:
            if rt.lower() in doc_type.lower() or doc_type.lower() in rt.lower():
                file_map[rt] = (fbytes, fmime)
                break

    missing = [rt for rt in required_types if rt not in file_map]
    if missing:
        missing_list = "".join(f"<li>{m}</li>" for m in missing)
        html = (
            '<div class="vr-block">'
            '<h3>Service PR</h3>'
            f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
            + _badge("fail") +
            f'<p><strong>Blocker:</strong> Missing required document(s):</p>'
            f'<ul class="iss">{missing_list}</ul>'
            + _verdict(False, len(missing))
            + '</div>'
        )
        return html, False

    scope = file_map["Scope of Work"]
    jsa = file_map["Job Safety Analysis (JSA)"]
    skillset = file_map["Technical Skill-Set"]

    try:
        result = gemini_client.validate_service_documents(
            scope[0], scope[1],
            jsa[0], jsa[1],
            skillset[0], skillset[1],
        )
    except Exception as e:
        err_str = str(e)
        if "403" in err_str or "401" in err_str or "permission" in err_str.lower():
            html = (
                '<div class="vr-block">'
                '<h3>Service PR</h3>'
                f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
                + _badge("warn") +
                '<p>⚠ All 3 documents received (Scope of Work, JSA, Technical Skill-Set). '
                'AI validation unavailable (Gemini API key needs '
                '<a href="https://console.cloud.google.com/apis/library/generativelanguage.googleapis.com" '
                'target="_blank">Generative Language API enabled</a>). '
                'Manual review required.</p>'
                + _verdict(True) +
                '</div>'
            )
            return html, True
        return (
            f'<div class="vr-block"><h3>Service PR</h3>{_badge("fail")}'
            f'<p>Validation error: {e}</p>{_verdict(False, 1)}</div>',
            False,
        )

    sow = result.get("scopeOfWork", {})
    jsa_r = result.get("jsa", {})
    ts = result.get("technicalSkillSet", {})
    cc = result.get("crossCheck", {})

    blockers = 0
    all_issues = []

    # Scope of Work
    sow_pass = sow.get("hasDeliverables", False) and sow.get("hasTimeline", False)
    if not sow_pass:
        blockers += 1

    # JSA
    jsa_pass = jsa_r.get("identifiesHazards", False) and jsa_r.get("hasControlMeasures", False)
    if not jsa_pass:
        blockers += 1

    # Skill-Set
    ts_pass = ts.get("listsQualifications", False)
    if not ts_pass:
        blockers += 1

    # Cross-check
    if not cc.get("consistent", True):
        blockers += 1
        all_issues.extend(cc.get("inconsistencies", []))

    all_valid = blockers == 0
    status = "pass" if all_valid else "fail"

    def doc_block(title: str, status_ok: bool, checks: list[tuple[bool, str]], issues: list) -> str:
        s = "pass" if status_ok else "fail"
        return (
            f'<div class="dr {s}">'
            f'<h4>{title}</h4>'
            + _badge(s)
            + '<ul class="chk">'
            + "".join(_chk(ok, label) for ok, label in checks)
            + '</ul>'
            + _issues_html(issues)
            + '</div>'
        )

    html = (
        '<div class="vr-block">'
        '<h3>Service PR</h3>'
        f'<p class="meta">PR Number: <strong>{pr_number}</strong></p>'
        + doc_block(
            "Scope of Work",
            sow_pass,
            [
                (sow.get("hasDeliverables", False), "Deliverables defined"),
                (sow.get("hasTimeline", False), "Timeline/duration specified"),
                (sow.get("hasAcceptanceCriteria", False), "Acceptance criteria stated"),
            ],
            sow.get("issues", []),
        )
        + doc_block(
            "Job Safety Analysis (JSA)",
            jsa_pass,
            [
                (jsa_r.get("identifiesHazards", False), "Hazards identified"),
                (jsa_r.get("hasControlMeasures", False), "Control measures for each hazard"),
                (jsa_r.get("hasSafetySignature", False), "Safety officer/supervisor signature"),
            ],
            jsa_r.get("issues", []),
        )
        + doc_block(
            "Technical Skill-Set",
            ts_pass,
            [
                (ts.get("listsQualifications", False), "Required qualifications/certifications listed"),
            ],
            ts.get("issues", []),
        )
        + (
            '<div class="dr ' + ("pass" if cc.get("consistent", True) else "fail") + '">'
            '<h4>Cross-Document Consistency</h4>'
            + _badge("pass" if cc.get("consistent", True) else "fail")
            + _issues_html(cc.get("inconsistencies", []))
            + '</div>'
        )
        + _issues_html(all_issues)
        + _verdict(all_valid, blockers)
        + '</div>'
    )
    return html, all_valid
