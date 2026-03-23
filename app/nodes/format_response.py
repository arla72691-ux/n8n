"""
Final LangGraph node — sets overall_status based on accumulated blocker_count.
"""
from app.state import PRValidationState


def format_response(state: PRValidationState) -> dict:
    blocker_count = state.get("blocker_count", 0)

    if blocker_count > 0:
        overall_status = "blockers"
    else:
        overall_status = "ready"

    return {
        "overall_status": overall_status,
        "blocker_count": blocker_count,
    }
