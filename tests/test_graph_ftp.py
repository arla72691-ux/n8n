"""
Integration tests for the FTP (First Time Purchase) graph flow.
Gemini service is mocked; Drive is not involved for FTP.
"""
from unittest.mock import patch, MagicMock

from app.graph import graph


def _ftp_state(files=None):
    return {
        "pr_type": "Supply PR (FTP \u2013 First Time Purchase)",
        "pr_number": "PR-FTP",
        "item_long_text": "",
        "pr_description": "",
        "uploaded_files": files or [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }


COSTING_FILE = {
    "name": "costing_sheet.pdf",
    "content": b"%PDF-1.4 mock costing sheet",
    "mime_type": "application/pdf",
    "doc_type": "costing_sheet",
}


def _mock_settings(gemini_enabled=True):
    s = MagicMock()
    s.gemini_enabled = gemini_enabled
    s.drive_enabled = False
    s.app_env = "test"
    return s


# ---------------------------------------------------------------------------
# FTP: valid costing sheet — all Gemini checks PASS → no blockers
# ---------------------------------------------------------------------------

def test_ftp_valid_costing_sheet_no_blockers():
    """Attachment present, Gemini returns all PASS and recent date → 0 blockers."""
    gemini_json = {
        "cost_breakdown_present": "PASS",
        "cost_breakdown_details": "material + processing + overhead + total",
        "validity_date_present": "PASS",
        "validity_date": "2026-02-01",  # within 6 months of March 2026
        "authorising_signature": "PASS",
        "notes": "",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.ftp_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_ftp_costing_sheet_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json):
        result = graph.invoke(_ftp_state([COSTING_FILE]))

    assert result["blocker_count"] == 0
    assert result["overall_status"] == "ready"
    icons = [m["icon"] for m in result["validation_messages"]]
    assert "✗" not in icons


# ---------------------------------------------------------------------------
# FTP: costing sheet date too old → blocker
# ---------------------------------------------------------------------------

def test_ftp_costing_sheet_too_old_is_blocker():
    """validity_date is > 6 months ago → date blocker."""
    gemini_json = {
        "cost_breakdown_present": "PASS",
        "cost_breakdown_details": "present",
        "validity_date_present": "PASS",
        "validity_date": "2023-01-01",  # ~38 months before March 2026
        "authorising_signature": "PASS",
        "notes": "",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.ftp_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_ftp_costing_sheet_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json):
        result = graph.invoke(_ftp_state([COSTING_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "blocker" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# FTP: cost breakdown missing → blocker
# ---------------------------------------------------------------------------

def test_ftp_missing_cost_breakdown_is_blocker():
    gemini_json = {
        "cost_breakdown_present": "FAIL",
        "cost_breakdown_details": "",
        "validity_date_present": "PASS",
        "validity_date": "2026-02-01",
        "authorising_signature": "PASS",
        "notes": "",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.ftp_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_ftp_costing_sheet_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json):
        result = graph.invoke(_ftp_state([COSTING_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "cost" in blocker_texts.lower() or "breakdown" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# FTP: authorising signature missing → blocker
# ---------------------------------------------------------------------------

def test_ftp_missing_signature_is_blocker():
    gemini_json = {
        "cost_breakdown_present": "PASS",
        "cost_breakdown_details": "present",
        "validity_date_present": "PASS",
        "validity_date": "2026-02-01",
        "authorising_signature": "FAIL",
        "notes": "",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.ftp_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_ftp_costing_sheet_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json):
        result = graph.invoke(_ftp_state([COSTING_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "signature" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# FTP: Gemini disabled, attachment present → found + skip warning, 0 blockers
# ---------------------------------------------------------------------------

def test_ftp_gemini_disabled_attachment_present():
    """With Gemini off, finding the attachment is enough — no blockers."""
    mock_s = _mock_settings(gemini_enabled=False)
    with patch("app.nodes.ftp_node.get_settings", return_value=mock_s):
        result = graph.invoke(_ftp_state([COSTING_FILE]))

    assert result["blocker_count"] == 0
    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "costing sheet" in texts.lower()
    assert "skipped" in texts.lower() or "manually verify" in texts.lower()
