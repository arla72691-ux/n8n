"""
LangGraph router node — determines which branch to take based on PR type.
"""
from app.state import PRValidationState


_PR_TYPE_MAP = {
    "Supply PR (APD)": "apd",
    "Supply PR (Non-APD)": "non_apd",
    "Supply PR (FTP – First Time Purchase)": "ftp",
    "PAC PR": "pac",
    "Service PR": "service",
}


def route_pr_type(state: PRValidationState) -> PRValidationState:
    """
    Initialise default state fields and set the routing key.
    This node doesn't modify routing itself — that's done via pick_branch.
    """
    return {
        **state,
        "parsed_items": state.get("parsed_items", []),
        "drive_results": state.get("drive_results", {}),
        "validation_messages": state.get("validation_messages", []),
        "blocker_count": state.get("blocker_count", 0),
        "overall_status": state.get("overall_status", "ready"),
    }


def pick_branch(state: PRValidationState) -> str:
    """Conditional edge function: returns the branch key for the current PR type."""
    pr_type = state.get("pr_type", "")
    return _PR_TYPE_MAP.get(pr_type, "non_apd")
