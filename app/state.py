from typing import TypedDict, List, Any, Optional


class UploadedFile(TypedDict):
    name: str
    content: bytes
    mime_type: str
    doc_type: str  # "drawing", "costing_sheet", "pac_cert", "scope_of_work", "jsa", "technical_skill_set"


class ValidationMessage(TypedDict):
    icon: str   # "✓", "⚠", or "✗"
    text: str


class PRValidationState(TypedDict):
    pr_type: str
    pr_number: str
    item_long_text: str
    pr_description: str
    uploaded_files: List[UploadedFile]
    # Populated during validation
    parsed_items: List[Any]        # List[APDItem] — set by apd_parse node
    drive_results: dict            # drawing_number → {"file": DriveFile|None, "bytes": bytes|None}
    validation_messages: List[ValidationMessage]
    blocker_count: int
    overall_status: str            # "ready" or "blockers"
    trace_id: str                  # Langfuse trace ID, threaded through for generation linking
