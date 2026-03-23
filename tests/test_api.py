"""
API integration tests using FastAPI TestClient.
"""
import pytest
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
