"""
LangGraph node for Supply PR (Non-APD) validation.
No attachment is required; agent returns standard advisory message.
"""
from app.state import PRValidationState, ValidationMessage


def handle(state: PRValidationState) -> dict:
    messages = list(state.get("validation_messages", []))
    messages.append({
        "icon": "✓",
        "text": (
            "This is a Non-APD Supply PR. No drawing attachment is required. "
            "Ensure the item description includes make, model, part number, or OEM reference "
            "for procurement clarity."
        ),
    })
    return {"validation_messages": messages}
