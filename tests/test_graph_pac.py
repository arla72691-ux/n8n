"""
Integration tests for the PAC PR graph flow.
Gemini service is mocked.
"""
from unittest.mock import patch, MagicMock

from app.graph import graph


def _pac_state(files=None, pr_description=""):
    return {
        "pr_type": "PAC PR",
        "pr_number": "PR-PAC",
        "item_long_text": "",
        "pr_description": pr_description,
        "uploaded_files": files or [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }


PAC_FILE = {
    "name": "pac_certificate.pdf",
    "content": b"%PDF-1.4 mock pac cert",
    "mime_type": "application/pdf",
    "doc_type": "pac_cert",
}


def _mock_settings(gemini_enabled=True):
    s = MagicMock()
    s.gemini_enabled = gemini_enabled
    s.drive_enabled = False
    s.app_env = "test"
    return s


def _all_pass_gemini_json(expiry_date="2030-01-01"):
    return {
        "pac_cert_number": "PASS",
        "vendor_name": "PASS",
        "expiry_date_present": "PASS",
        "expiry_date": expiry_date,
        "authorising_signature": "PASS",
        "scope_match": "PASS",
        "notes": "",
    }


# ---------------------------------------------------------------------------
# PAC: valid certificate — all checks PASS → no blockers
# ---------------------------------------------------------------------------

def test_pac_valid_cert_no_blockers():
    """Certificate attached, Gemini returns all PASS, expiry far in future → 0 blockers."""
    mock_s = _mock_settings()
    with patch("app.nodes.pac_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_pac_cert_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=_all_pass_gemini_json("2030-01-01")):
        result = graph.invoke(_pac_state([PAC_FILE]))

    assert result["blocker_count"] == 0
    assert result["overall_status"] == "ready"
    icons = [m["icon"] for m in result["validation_messages"]]
    assert "✗" not in icons


# ---------------------------------------------------------------------------
# PAC: certificate expired → blocker
# ---------------------------------------------------------------------------

def test_pac_expired_cert_is_blocker():
    """Expiry date in the past → _check_expiry returns 'expired' → blocker."""
    mock_s = _mock_settings()
    with patch("app.nodes.pac_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_pac_cert_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response",
               return_value=_all_pass_gemini_json("2020-01-01")):  # clearly expired
        result = graph.invoke(_pac_state([PAC_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "expired" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# PAC: certificate expiring within 30 days → warning, NOT a blocker
# ---------------------------------------------------------------------------

def test_pac_cert_expiring_soon_is_warning_not_blocker():
    """_check_expiry mocked to return 'soon' → ⚠ warning added, blocker_count stays 0."""
    mock_s = _mock_settings()
    with patch("app.nodes.pac_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_pac_cert_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response",
               return_value=_all_pass_gemini_json("2026-04-01")), \
         patch("app.nodes.pac_node._check_expiry", return_value="soon"):
        result = graph.invoke(_pac_state([PAC_FILE]))

    icons = [m["icon"] for m in result["validation_messages"]]
    assert "✗" not in icons  # warning only, no blocker
    assert "⚠" in icons
    texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "⚠")
    assert "30 days" in texts or "renew" in texts.lower() or "expires" in texts.lower()


# ---------------------------------------------------------------------------
# PAC: expiry date missing → blocker
# ---------------------------------------------------------------------------

def test_pac_missing_expiry_date_is_blocker():
    """expiry_date_present=FAIL and no expiry_date → blocker."""
    gemini_json = {
        "pac_cert_number": "PASS",
        "vendor_name": "PASS",
        "expiry_date_present": "FAIL",
        "expiry_date": None,
        "authorising_signature": "PASS",
        "scope_match": "PASS",
        "notes": "",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.pac_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_pac_cert_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json):
        result = graph.invoke(_pac_state([PAC_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "expiry" in blocker_texts.lower() or "validity" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# PAC: Gemini disabled, cert present → found + skip warning, 0 blockers
# ---------------------------------------------------------------------------

def test_pac_gemini_disabled_cert_present():
    """With Gemini off, attaching the certificate is enough — no blockers."""
    mock_s = _mock_settings(gemini_enabled=False)
    with patch("app.nodes.pac_node.get_settings", return_value=mock_s):
        result = graph.invoke(_pac_state([PAC_FILE]))

    assert result["blocker_count"] == 0
    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "pac certificate" in texts.lower()
    assert "manually verify" in texts.lower() or "skipped" in texts.lower()
