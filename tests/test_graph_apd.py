"""
Integration tests for the APD graph flow.

New validation logic (no Drive):
  - Requires an uploaded drawing attachment.
  - Gemini reads the drawing's title block and schedule/BOM table.
  - Checks: has_drawings, drawing_number_match, revision_match, position_number_match.
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
        "trace_id": "",
    }


APD_LINE = "APD,ITEM NAME:HOLDBACK;DRAWING NUMBER:4802-1594;POSITION OR ITEM NUMBER:P3;REVISION:0"

_DRAWING_FILE = {
    "name": "4802-1594-drawing.pdf",
    "content": b"%PDF mock drawing",
    "mime_type": "application/pdf",
    "doc_type": "drawing",
}

_ALL_PASS_GEMINI = {
    "has_drawings": "PASS",
    "drawing_number_match": "PASS",
    "revision_match": "PASS",
    "position_number_match": "PASS",
    "found_drawing_number": "4802-1594",
    "found_revision": "0",
    "found_position_numbers": "P3",
    "notes": "",
}


def _mock_settings(gemini_enabled=True):
    s = MagicMock()
    s.drive_enabled = False
    s.gemini_enabled = gemini_enabled
    s.app_env = "test"
    return s


# ---------------------------------------------------------------------------
# Happy path: attachment provided, all schedule-table checks pass
# ---------------------------------------------------------------------------

def test_apd_attachment_all_checks_pass():
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=_ALL_PASS_GEMINI), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] == 0
    assert result["overall_status"] == "ready"
    icons = [m["icon"] for m in result["validation_messages"]]
    assert "✗" not in icons


# ---------------------------------------------------------------------------
# No drawing attached → blocker
# ---------------------------------------------------------------------------

def test_apd_no_attachment_is_blocker():
    mock_s = _mock_settings(gemini_enabled=False)
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s):
        result = graph.invoke(_make_state(APD_LINE))

    assert result["blocker_count"] >= 1
    assert result["overall_status"] == "blockers"
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "no drawing attachment" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# has_drawings=FAIL → blank/empty file → blocker, other checks skipped
# ---------------------------------------------------------------------------

def test_apd_empty_drawing_is_blocker():
    gemini_json = {**_ALL_PASS_GEMINI, "has_drawings": "FAIL"}
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "blank" in blocker_texts.lower() or "no drawing content" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# drawing_number_match=FAIL → blocker
# ---------------------------------------------------------------------------

def test_apd_drawing_number_mismatch_is_blocker():
    gemini_json = {
        **_ALL_PASS_GEMINI,
        "drawing_number_match": "FAIL",
        "found_drawing_number": "9999-0000",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "drawing number" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# revision_match=FAIL → blocker
# ---------------------------------------------------------------------------

def test_apd_revision_mismatch_is_blocker():
    gemini_json = {
        **_ALL_PASS_GEMINI,
        "revision_match": "FAIL",
        "found_revision": "B",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "revision" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# position_number_match=FAIL → blocker
# ---------------------------------------------------------------------------

def test_apd_position_number_mismatch_is_blocker():
    gemini_json = {
        **_ALL_PASS_GEMINI,
        "position_number_match": "FAIL",
        "found_position_numbers": "P1, P2",
    }
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "position" in blocker_texts.lower() or "item" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# position_number_match=NOT_CHECKED → warning (no BOM table), not a blocker
# ---------------------------------------------------------------------------

def test_apd_position_number_not_checked_gives_warning():
    """No schedule table on drawing → ⚠ warning to manually verify, no blocker."""
    gemini_json = {**_ALL_PASS_GEMINI, "position_number_match": "NOT_CHECKED"}
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] == 0
    warn_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "⚠")
    assert "manually confirm" in warn_texts.lower() or "no parts list" in warn_texts.lower()


# ---------------------------------------------------------------------------
# Gemini returns unparseable JSON → warning, not crash
# ---------------------------------------------------------------------------

def test_apd_gemini_malformed_json_gives_warning():
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="not-json"), \
         patch("app.services.gemini_service.parse_json_response", side_effect=ValueError("bad json")), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] == 0
    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "manual review" in texts.lower()


# ---------------------------------------------------------------------------
# Gemini disabled → warning message, no blocker
# ---------------------------------------------------------------------------

def test_apd_gemini_disabled_gives_warning_not_blocker():
    mock_s = _mock_settings(gemini_enabled=False)
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s):
        result = graph.invoke(_make_state(APD_LINE, uploaded_files=[_DRAWING_FILE]))

    assert result["blocker_count"] == 0
    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "gemini validation skipped" in texts.lower() or "manually verify" in texts.lower()


# ---------------------------------------------------------------------------
# Multi-position drawing: same drawing covers two items → ⚠ advisory
# ---------------------------------------------------------------------------

def test_apd_multi_position_drawing_advisory():
    two_items = (
        "APD,ITEM NAME:HOLDBACK;DRAWING NUMBER:4802-1594;POSITION OR ITEM NUMBER:P3;REVISION:0\n"
        "APD,ITEM NAME:BOLT;DRAWING NUMBER:4802-1594;POSITION OR ITEM NUMBER:P4;REVISION:0"
    )
    gemini_json = {**_ALL_PASS_GEMINI, "found_part_numbers": "P3, P4"}
    mock_s = _mock_settings()
    with patch("app.nodes.apd_nodes.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("prompt", None)):

        result = graph.invoke(_make_state(two_items, uploaded_files=[_DRAWING_FILE]))

    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "covers" in texts.lower() or "positions" in texts.lower()
    assert result["blocker_count"] == 0


# ---------------------------------------------------------------------------
# Empty item_long_text → parse failure blocker
# ---------------------------------------------------------------------------

def test_apd_empty_item_long_text_is_blocker():
    result = graph.invoke(_make_state(""))
    assert result["blocker_count"] >= 1
    assert result["overall_status"] == "blockers"
    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "no valid apd items" in texts.lower()


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
        "trace_id": "",
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
        "trace_id": "",
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
        "trace_id": "",
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
        "trace_id": "",
    }
    result = graph.invoke(state)
    assert result["blocker_count"] == 3
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
        "trace_id": "",
    }
    result = graph.invoke(state)
    assert result["blocker_count"] == 2
