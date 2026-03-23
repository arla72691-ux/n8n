"""
Integration tests for the APD graph flow.
Drive and Gemini services are mocked.
"""
import pytest
from unittest.mock import patch, MagicMock

from app.graph import graph
from app.state import PRValidationState


def _make_state(item_long_text: str, uploaded_files=None) -> PRValidationState:
    return {
        "pr_type": "Supply PR (APD)",
        "pr_number": "PR-001",
        "item_long_text": item_long_text,
        "pr_description": "",
        "uploaded_files": uploaded_files or [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }


APD_LINE = "APD,ITEM NAME:HOLDBACK;DRAWING NUMBER:4802-1594;POSITION OR ITEM NUMBER:P3;REVISION:0"


def _mock_settings(drive_enabled=True, gemini_enabled=True):
    s = MagicMock()
    s.drive_enabled = drive_enabled
    s.gemini_enabled = gemini_enabled
    s.app_env = "test"
    s.google_drive_drawings_folder_id = "folder123"
    return s


# ---------------------------------------------------------------------------
# Drawing found in Drive — Gemini validates and returns pass
# ---------------------------------------------------------------------------

def test_apd_drawing_found_in_drive():
    """When drawing is found in Drive, Gemini validates it and returns pass messages."""
    from app.services.drive_service import DriveFile

    mock_drive_file = DriveFile(id="file123", name="4802-1594-REV0.pdf", mime_type="application/pdf")
    mock_bytes = b"%PDF-1.4 mock drawing bytes"
    mock_gemini_json = {
        "drawing_number_match": "PASS",
        "revision_match": "PASS",
        "approval_stamp": "PASS",
        "legible": "PASS",
        "notes": "",
    }

    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.drive_service.search_drawing", return_value=mock_drive_file), \
         patch("app.services.drive_service.download_file_bytes", return_value=mock_bytes), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=mock_gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value="prompt"):

        result = graph.invoke(_make_state(APD_LINE))

    icons = [m["icon"] for m in result["validation_messages"]]
    assert result["blocker_count"] == 0
    assert result["overall_status"] == "ready"
    assert "✗" not in icons


# ---------------------------------------------------------------------------
# Drawing NOT in Drive, attachment provided — validate attachment
# ---------------------------------------------------------------------------

def test_apd_drawing_not_in_drive_attachment_provided():
    uploaded = [{
        "name": "4802-1594-drawing.pdf",
        "content": b"%PDF mock",
        "mime_type": "application/pdf",
        "doc_type": "drawing",
    }]
    mock_gemini_json = {
        "drawing_number_match": "PASS",
        "revision_match": "PASS",
        "approval_stamp": "PASS",
        "legible": "PASS",
        "notes": "",
    }

    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.drive_service.search_drawing", return_value=None), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=mock_gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value="prompt"):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=uploaded))

    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "not found in system" in texts.lower()
    assert result["blocker_count"] == 0


# ---------------------------------------------------------------------------
# Drawing NOT in Drive, NO attachment — blocker
# ---------------------------------------------------------------------------

def test_apd_drawing_not_in_drive_no_attachment():
    mock_s = _mock_settings(gemini_enabled=False)
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.drive_service.search_drawing", return_value=None):

        result = graph.invoke(_make_state(APD_LINE))

    assert result["blocker_count"] >= 1
    assert result["overall_status"] == "blockers"
    blocker_msgs = [m for m in result["validation_messages"] if m["icon"] == "✗"]
    assert len(blocker_msgs) >= 1
    assert "blocker" in blocker_msgs[0]["text"].lower()


# ---------------------------------------------------------------------------
# Non-APD: no attachment needed
# ---------------------------------------------------------------------------

def test_non_apd_no_blocker():
    state = {
        "pr_type": "Supply PR (Non-APD)",
        "pr_number": "PR-002",
        "item_long_text": "",
        "pr_description": "",
        "uploaded_files": [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }
    result = graph.invoke(state)
    assert result["blocker_count"] == 0
    assert result["overall_status"] == "ready"
    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "non-apd" in texts.lower()


# ---------------------------------------------------------------------------
# FTP: missing attachment → blocker
# ---------------------------------------------------------------------------

def test_ftp_missing_attachment():
    state = {
        "pr_type": "Supply PR (FTP – First Time Purchase)",
        "pr_number": "PR-003",
        "item_long_text": "",
        "pr_description": "",
        "uploaded_files": [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }
    result = graph.invoke(state)
    assert result["blocker_count"] >= 1
    assert result["overall_status"] == "blockers"


# ---------------------------------------------------------------------------
# PAC: missing certificate → blocker
# ---------------------------------------------------------------------------

def test_pac_missing_certificate():
    state = {
        "pr_type": "PAC PR",
        "pr_number": "PR-004",
        "item_long_text": "",
        "pr_description": "",
        "uploaded_files": [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }
    result = graph.invoke(state)
    assert result["blocker_count"] >= 1
    assert result["overall_status"] == "blockers"


# ---------------------------------------------------------------------------
# Service PR: missing documents → blockers
# ---------------------------------------------------------------------------

def test_service_missing_all_docs():
    state = {
        "pr_type": "Service PR",
        "pr_number": "PR-005",
        "item_long_text": "",
        "pr_description": "",
        "uploaded_files": [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }
    result = graph.invoke(state)
    assert result["blocker_count"] == 3  # one per missing doc
    assert result["overall_status"] == "blockers"


def test_service_partial_docs():
    """Only scope_of_work provided; jsa and technical_skill_set missing → 2 blockers."""
    uploaded = [{
        "name": "scope.pdf",
        "content": b"%PDF",
        "mime_type": "application/pdf",
        "doc_type": "scope_of_work",
    }]
    state = {
        "pr_type": "Service PR",
        "pr_number": "PR-006",
        "item_long_text": "",
        "pr_description": "",
        "uploaded_files": uploaded,
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }
    result = graph.invoke(state)
    assert result["blocker_count"] == 2
