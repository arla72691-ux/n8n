"""
Gemini service for multimodal document validation.

Uses the google-genai SDK (google.genai) with Gemini 2.5 Pro.
Files are passed as inline base64 data so no server-side file upload is needed.
"""
import json
import logging
import re
import time
from functools import lru_cache

from google import genai
from google.genai import types
from google.api_core.exceptions import ResourceExhausted, ServiceUnavailable, InternalServerError

from app.config import get_settings

logger = logging.getLogger(__name__)

_RETRYABLE = (ResourceExhausted, ServiceUnavailable, InternalServerError)
_RETRY_DELAYS = (2, 4, 8)  # seconds between attempts 1→2, 2→3, 3→4


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key)


def validate_document(
    file_bytes: bytes,
    mime_type: str,
    prompt: str,
    langfuse_prompt=None,
    trace_id: str = "",
) -> str:
    """
    Send a file (as inline base64) and a text prompt to Gemini.
    Returns the raw text response.

    Retries up to 3 times with exponential backoff on transient 503/429/500 errors.

    If `langfuse_prompt` (a PromptClient) and `trace_id` are supplied, a
    Langfuse generation observation is recorded and linked to the prompt
    template so that Prompt Management shows observation counts.
    """
    client = get_gemini_client()

    last_exc = None
    for attempt, delay in enumerate((*_RETRY_DELAYS, None), start=1):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[
                    types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                ),
            )
            response_text = response.text
            break
        except _RETRYABLE as exc:
            last_exc = exc
            if delay is None:
                raise
            logger.warning(
                "Gemini transient error (attempt %d/4): %s — retrying in %ds",
                attempt, exc, delay,
            )
            time.sleep(delay)
    else:
        raise last_exc  # should never reach here but satisfies linters

    if langfuse_prompt and trace_id:
        try:
            from app.services.langfuse_service import get_langfuse_client
            lf = get_langfuse_client()
            if lf:
                lf.generation(
                    trace_id=trace_id,
                    name="gemini-validation",
                    model="gemini-2.5-flash",
                    input=prompt,
                    output=response_text,
                    prompt=langfuse_prompt,
                )
        except Exception:
            pass  # never block validation for observability failures

    return response_text


def parse_json_response(text: str) -> dict:
    """
    Extract a JSON object from Gemini's response text.
    Gemini sometimes wraps JSON in markdown code fences.
    """
    # Try to find JSON block inside ```json ... ``` or ``` ... ```
    match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try to parse the whole string as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Last resort: find the first { ... } block
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        raise ValueError(f"Could not parse JSON from Gemini response: {text[:200]}")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def build_apd_drawing_prompt(drawing_number: str, revision: str, part_numbers: list = None):
    part_numbers_str = ", ".join(part_numbers) if part_numbers else "not specified"
    fallback = f"""You are validating an engineering drawing attached to an APD Purchase Requisition.

Expected values from the Purchase Requisition item long text:
- Drawing number: {drawing_number}
- Revision: {revision} (treat 0, 00, and "NO REVISION" as equivalent base revisions)
- Position/item number(s) to verify: {part_numbers_str}

─── ENGINEERING DRAWING LAYOUT — WHERE TO LOOK ────────────────────────────────

This drawing likely contains a PARTS LIST / SCHEDULE / BOM TABLE. Look for it in:

  1. RIGHT SIDE OF SHEET — a table in the right portion, either:
     • A full-height vertical strip running along the right edge (IEC/ISO/DIN — very
       common in Latin America, Europe), OR
     • A mid-sheet table block on the right half of the drawing

  2. LOWER RIGHT CORNER — a horizontal block directly above the title block
     (ASME/ANSI/BS — common in North America)

  3. UPPER RIGHT CORNER — stacked horizontal block

The table reads BOTTOM TO TOP: item 1 is at the BOTTOM, higher numbers are above it.

HOW TO IDENTIFY THE TABLE — look for ANY bordered grid with:
  • A column of sequential numbers: 1, 2, 3 … (or 01, 02 …) on the left
  • Columns for quantity and/or description to the right of the number column
  • A header row labelled (English OR Spanish):
    "PARTS LIST" / "LISTA DE MATERIALES" / "LISTA DE PARTES" / "BOM" / "SCHEDULE" /
    "BILL OF MATERIALS" / "MATERIAL LIST" / "DESPIECE" / "NOMENCLATURA"
  • Column headers: "ITEM"/"POS."/"POSICION"/"Nº"/"No." | "QTY"/"CANT." | "DESCRIPTION"/"DESCRIPCION"
  • The first/leftmost column may be UNLABELLED and just contain the sequential numbers

BALLOON CALLOUTS (secondary): circled or enclosed numbers scattered on the drawing
  view pointing to components — these match the item numbers in the parts list.

TITLE BLOCK: lower right — drawing number, revision, date.
REVISION BLOCK: upper right — revision history (separate from parts list).

─── NORMALISATION ──────────────────────────────────────────────────────────────
• "P3" in the PR == "3" in the table. "01" == "1". Strip leading zeros and "P" prefix.

─── IMPORTANT: WHEN TO USE EACH STATUS ────────────────────────────────────────
PASS      — You can clearly read the table and ALL required position numbers are present.
FAIL      — You can read the table and one or more required numbers are MISSING.
UNCLEAR   — You can SEE what looks like a table, numbered column, or circled numbers
            BUT the text/numbers are too small, blurry, or compressed to read reliably.
            USE THIS if the image quality prevents confident reading.
NOT_CHECKED — Use ONLY when the drawing is a single-component detail drawing with
            genuinely NO table, NO numbered grid, and NO circled balloon numbers
            ANYWHERE on the sheet. This is rare. If you see ANY grid or ANY circled
            numbers, use UNCLEAR instead of NOT_CHECKED.

────────────────────────────────────────────────────────────────────────────────

Scan the ENTIRE sheet. Then respond ONLY with a JSON object in exactly this format:
{{
  "has_drawings": "PASS" | "FAIL",
  "drawing_number_match": "PASS" | "FAIL" | "UNCLEAR",
  "revision_match": "PASS" | "FAIL" | "UNCLEAR",
  "position_number_match": "PASS" | "FAIL" | "UNCLEAR" | "NOT_CHECKED",
  "found_drawing_number": "<drawing number found on document, or null>",
  "found_revision": "<revision found on document, or null>",
  "found_position_numbers": "<ALL item/position numbers you can read, or null>",
  "notes": "<brief explanation>"
}}

Criteria:
- has_drawings: Real engineering geometry present? FAIL if completely blank.
- drawing_number_match: Title block drawing number == {drawing_number}?
- revision_match: Revision in title block or revision table == {revision}?
- position_number_match: Using the rules above, do ALL of [{part_numbers_str}] appear?"""
    from app.services.langfuse_service import get_prompt_and_client, PROMPT_APD_DRAWING
    return get_prompt_and_client(PROMPT_APD_DRAWING, fallback=fallback,
                                 drawing_number=drawing_number, revision=revision,
                                 part_numbers=part_numbers_str)


def build_ftp_costing_sheet_prompt():
    fallback = """You are validating a First Time Purchase (FTP) costing sheet attached to a Purchase Requisition.

Examine the document carefully and respond ONLY with a JSON object in exactly this format:
{
  "cost_breakdown_present": "PASS" | "FAIL",
  "cost_breakdown_details": "<what cost components were found or what is missing>",
  "validity_date_present": "PASS" | "FAIL",
  "validity_date": "<the date found, or null if not present>",
  "authorising_signature": "PASS" | "FAIL" | "UNCLEAR",
  "notes": "<any additional observations>"
}

Criteria:
- cost_breakdown_present: Does the document contain a cost breakdown with material, processing, overhead, and total unit cost?
- validity_date_present: Is there a validity or effective date visible?
- authorising_signature: Is there an authorising signature or approver name visible?"""
    from app.services.langfuse_service import get_prompt_and_client, PROMPT_FTP_COSTING
    return get_prompt_and_client(PROMPT_FTP_COSTING, fallback=fallback)


def build_pac_cert_prompt(pr_description: str = ""):
    scope_check = (
        f"\n- scope_match: Does the scope/description on the certificate broadly match the PR description: \"{pr_description}\"?"
        if pr_description
        else ""
    )
    scope_field = (
        '"scope_match": "PASS" | "FAIL" | "UNCLEAR" | "NOT_CHECKED",'
        if pr_description
        else '"scope_match": "NOT_CHECKED",'
    )
    fallback = f"""You are validating a PAC (Pre-Approved Certificate / Preferred Approved Contractor) certificate attached to a Purchase Requisition.

Examine the document carefully and respond ONLY with a JSON object in exactly this format:
{{
  "pac_cert_number": "PASS" | "FAIL",
  "vendor_name": "PASS" | "FAIL",
  "expiry_date_present": "PASS" | "FAIL",
  "expiry_date": "<the expiry/validity date found, or null>",
  "authorising_signature": "PASS" | "FAIL" | "UNCLEAR",
  {scope_field}
  "notes": "<any additional observations>"
}}

Criteria:
- pac_cert_number: Is a PAC certificate number visible on the document?
- vendor_name: Is a vendor or company name visible?
- expiry_date_present: Is a validity or expiry date present on the document?
- authorising_signature: Is there an authorising signature or approver name?{scope_check}"""
    from app.services.langfuse_service import get_prompt_and_client, PROMPT_PAC_CERT
    return get_prompt_and_client(PROMPT_PAC_CERT, fallback=fallback, pr_description=pr_description)


def build_scope_of_work_prompt():
    fallback = """You are validating a Scope of Work document attached to a Service Purchase Requisition.

Examine the document carefully and respond ONLY with a JSON object in exactly this format:
{
  "deliverables_present": "PASS" | "FAIL",
  "timeline_present": "PASS" | "FAIL",
  "acceptance_criteria_present": "PASS" | "FAIL",
  "vendor_name": "<vendor or company name found, or null>",
  "work_description_summary": "<brief 1-2 sentence summary of the work described>",
  "notes": "<any additional observations>"
}

Criteria:
- deliverables_present: Does the document contain identifiable deliverables or work outputs?
- timeline_present: Is there a timeline, duration, or schedule mentioned?
- acceptance_criteria_present: Are there acceptance criteria or sign-off conditions mentioned?"""
    from app.services.langfuse_service import get_prompt_and_client, PROMPT_SCOPE_OF_WORK
    return get_prompt_and_client(PROMPT_SCOPE_OF_WORK, fallback=fallback)


def build_jsa_prompt():
    fallback = """You are validating a Job Safety Analysis (JSA) document attached to a Service Purchase Requisition.

Examine the document carefully and respond ONLY with a JSON object in exactly this format:
{
  "hazards_identified": "PASS" | "FAIL",
  "control_measures_present": "PASS" | "FAIL",
  "safety_officer_signature": "PASS" | "FAIL" | "UNCLEAR",
  "vendor_name": "<vendor or company name found, or null>",
  "work_description_summary": "<brief 1-2 sentence summary of the work covered>",
  "notes": "<any additional observations>"
}

Criteria:
- hazards_identified: Does the JSA identify specific hazards associated with the work?
- control_measures_present: Are control measures or mitigations listed for the identified hazards?
- safety_officer_signature: Is there a safety officer, HSE authority, or supervisor signature visible?"""
    from app.services.langfuse_service import get_prompt_and_client, PROMPT_JSA
    return get_prompt_and_client(PROMPT_JSA, fallback=fallback)


def build_technical_skillset_prompt():
    fallback = """You are validating a Technical Skill-Set document attached to a Service Purchase Requisition.

Examine the document carefully and respond ONLY with a JSON object in exactly this format:
{
  "qualifications_listed": "PASS" | "FAIL",
  "certifications_listed": "PASS" | "FAIL",
  "vendor_name": "<vendor or company name found, or null>",
  "work_description_summary": "<brief 1-2 sentence summary of the skills/roles described>",
  "notes": "<any additional observations>"
}

Criteria:
- qualifications_listed: Does the document list required qualifications for the roles involved?
- certifications_listed: Does the document specify required certifications or licences for the roles?"""
    from app.services.langfuse_service import get_prompt_and_client, PROMPT_TECHNICAL_SKILLSET
    return get_prompt_and_client(PROMPT_TECHNICAL_SKILLSET, fallback=fallback)
