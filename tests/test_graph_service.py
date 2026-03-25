"""
Integration tests for the Service PR graph flow.
Gemini service is mocked.
"""
from unittest.mock import patch, MagicMock

from app.graph import graph


def _service_state(files=None):
    return {
        "pr_type": "Service PR",
        "pr_number": "PR-SVC",
        "item_long_text": "",
        "pr_description": "Annual maintenance of pumping station",
        "uploaded_files": files or [],
        "parsed_items": [],
        "drive_results": {},
        "validation_messages": [],
        "blocker_count": 0,
        "overall_status": "ready",
    }


def _make_all_docs():
    return [
        {"name": "scope.pdf",  "content": b"%PDF", "mime_type": "application/pdf", "doc_type": "scope_of_work"},
        {"name": "jsa.pdf",    "content": b"%PDF", "mime_type": "application/pdf", "doc_type": "jsa"},
        {"name": "tech.pdf",   "content": b"%PDF", "mime_type": "application/pdf", "doc_type": "technical_skill_set"},
    ]


def _mock_settings(gemini_enabled=True):
    s = MagicMock()
    s.gemini_enabled = gemini_enabled
    s.drive_enabled = False
    s.app_env = "test"
    return s


# ---------------------------------------------------------------------------
# Service: all 3 docs present, all Gemini checks PASS → no blockers
# ---------------------------------------------------------------------------

def test_service_all_docs_all_pass():
    """All three documents provided; Gemini returns PASS for all fields → 0 blockers."""
    scope_result = {
        "deliverables_present": "PASS",
        "timeline_present": "PASS",
        "acceptance_criteria_present": "PASS",
        "vendor_name": "ACME Corp",
        "work_description_summary": "Pump maintenance",
        "notes": "",
    }
    jsa_result = {
        "hazards_identified": "PASS",
        "control_measures_present": "PASS",
        "safety_officer_signature": "PASS",
        "vendor_name": "ACME Corp",
        "work_description_summary": "Pump maintenance",
        "notes": "",
    }
    tech_result = {
        "qualifications_listed": "PASS",
        "certifications_listed": "PASS",
        "vendor_name": "ACME Corp",
        "notes": "",
    }

    mock_s = _mock_settings()
    with patch("app.nodes.service_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_scope_of_work_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.build_jsa_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.build_technical_skillset_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response",
               side_effect=[scope_result, jsa_result, tech_result]):
        result = graph.invoke(_service_state(_make_all_docs()))

    assert result["blocker_count"] == 0
    assert result["overall_status"] == "ready"
    icons = [m["icon"] for m in result["validation_messages"]]
    assert "✗" not in icons


# ---------------------------------------------------------------------------
# Service: scope deliverables missing → blocker
# ---------------------------------------------------------------------------

def test_service_scope_missing_deliverables_is_blocker():
    """Scope of Work lacks deliverables → 1 blocker."""
    scope_result = {
        "deliverables_present": "FAIL",
        "timeline_present": "PASS",
        "acceptance_criteria_present": "PASS",
        "vendor_name": "ACME Corp",
        "notes": "",
    }
    jsa_result = {
        "hazards_identified": "PASS",
        "control_measures_present": "PASS",
        "safety_officer_signature": "PASS",
        "vendor_name": "ACME Corp",
        "notes": "",
    }
    tech_result = {
        "qualifications_listed": "PASS",
        "certifications_listed": "PASS",
        "vendor_name": "ACME Corp",
        "notes": "",
    }

    mock_s = _mock_settings()
    with patch("app.nodes.service_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_scope_of_work_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.build_jsa_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.build_technical_skillset_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response",
               side_effect=[scope_result, jsa_result, tech_result]):
        result = graph.invoke(_service_state(_make_all_docs()))

    assert result["blocker_count"] >= 1
    blocker_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "✗")
    assert "deliverable" in blocker_texts.lower()


# ---------------------------------------------------------------------------
# Service: vendor name mismatch across docs → ⚠ warning, NOT a blocker
# ---------------------------------------------------------------------------

def test_service_vendor_name_mismatch_is_warning_not_blocker():
    """Scope and JSA have different vendor names → cross-check ⚠, blocker_count stays 0."""
    scope_result = {
        "deliverables_present": "PASS",
        "timeline_present": "PASS",
        "acceptance_criteria_present": "PASS",
        "vendor_name": "ACME Corp",
        "notes": "",
    }
    jsa_result = {
        "hazards_identified": "PASS",
        "control_measures_present": "PASS",
        "safety_officer_signature": "PASS",
        "vendor_name": "XYZ Services Ltd",  # different vendor!
        "notes": "",
    }
    tech_result = {
        "qualifications_listed": "PASS",
        "certifications_listed": "PASS",
        "vendor_name": "ACME Corp",
        "notes": "",
    }

    mock_s = _mock_settings()
    with patch("app.nodes.service_node.get_settings", return_value=mock_s), \
         patch("app.services.gemini_service.build_scope_of_work_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.build_jsa_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.build_technical_skillset_prompt", return_value=("p", None)), \
         patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response",
               side_effect=[scope_result, jsa_result, tech_result]):
        result = graph.invoke(_service_state(_make_all_docs()))

    assert result["blocker_count"] == 0  # cross-check is a warning, not a blocker
    warning_texts = " ".join(m["text"] for m in result["validation_messages"] if m["icon"] == "⚠")
    assert "vendor" in warning_texts.lower() or "inconsistent" in warning_texts.lower()


# ---------------------------------------------------------------------------
# Service: Gemini disabled, all docs present → found + skip warning, 0 blockers
# ---------------------------------------------------------------------------

def test_service_gemini_disabled_all_docs_present():
    """With Gemini off, providing all 3 documents is enough — no blockers."""
    mock_s = _mock_settings(gemini_enabled=False)
    with patch("app.nodes.service_node.get_settings", return_value=mock_s):
        result = graph.invoke(_service_state(_make_all_docs()))

    assert result["blocker_count"] == 0
    texts = " ".join(m["text"] for m in result["validation_messages"])
    assert "all three" in texts.lower() or "three required" in texts.lower()
    assert "manually verify" in texts.lower() or "skipped" in texts.lower()
