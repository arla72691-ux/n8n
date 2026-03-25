"""
API integration tests using FastAPI TestClient.
"""
import pytest
from unittest.mock import patch
from httpx import AsyncClient, ASGITransport

from app.main import app


@pytest.mark.asyncio
async def test_health():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert "status" in data
    assert data["status"] == "ok"


@pytest.mark.asyncio
async def test_validate_non_apd():
    """Non-APD PR should return ready with no blockers."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/validate",
            data={"pr_type": "Supply PR (Non-APD)", "pr_number": "PR-TEST-001"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["overall_status"] == "ready"
    assert data["blocker_count"] == 0
    assert len(data["messages"]) > 0


@pytest.mark.asyncio
async def test_validate_ftp_no_attachment():
    """FTP without attachment should return blockers."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/validate",
            data={"pr_type": "Supply PR (FTP – First Time Purchase)", "pr_number": "PR-TEST-002"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["overall_status"] == "blockers"
    assert data["blocker_count"] >= 1


@pytest.mark.asyncio
async def test_validate_pac_no_attachment():
    """PAC PR without certificate should return blockers."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/validate",
            data={"pr_type": "PAC PR", "pr_number": "PR-TEST-003"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["overall_status"] == "blockers"


@pytest.mark.asyncio
async def test_validate_service_no_docs():
    """Service PR with no documents should return 3 blockers."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/validate",
            data={"pr_type": "Service PR", "pr_number": "PR-TEST-004"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["blocker_count"] == 3


@pytest.mark.asyncio
async def test_index_page():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/")
    assert r.status_code == 200
    assert "PR Attachment Validation" in r.text


@pytest.mark.asyncio
async def test_validate_apd_empty_text_gives_blocker():
    """APD PR with no item_long_text → parse failure → blocker in response."""
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post(
            "/validate",
            data={"pr_type": "Supply PR (APD)", "pr_number": "PR-APD-EMPTY"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["blocker_count"] >= 1
    assert data["overall_status"] == "blockers"


@pytest.mark.asyncio
async def test_validate_apd_with_file_upload():
    """APD PR with drawing file uploaded — request accepted, structured response returned."""
    pdf_bytes = b"%PDF-1.4 mock drawing"
    item_text = "APD,ITEM NAME:BRACKET;DRAWING NUMBER:TEST-001;REVISION:0;POSITION OR ITEM NUMBER:P1"
    gemini_json = {
        "has_drawings": "PASS",
        "drawing_number_match": "PASS",
        "revision_match": "PASS",
        "part_number_match": "PASS",
        "found_drawing_number": "TEST-001",
        "found_revision": "0",
        "found_part_numbers": "P1",
        "notes": "",
    }
    with patch("app.services.gemini_service.validate_document", return_value="{}"), \
         patch("app.services.gemini_service.parse_json_response", return_value=gemini_json), \
         patch("app.services.gemini_service.build_apd_drawing_prompt", return_value=("p", None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            r = await client.post(
                "/validate",
                data={
                    "pr_type": "Supply PR (APD)",
                    "pr_number": "PR-APD-UPLOAD",
                    "item_long_text": item_text,
                },
                files={"files": ("TEST-001-drawing.pdf", pdf_bytes, "application/pdf")},
            )
    assert r.status_code == 200
    data = r.json()
    assert "messages" in data
    assert "overall_status" in data
    assert "blocker_count" in data
