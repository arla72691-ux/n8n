"""
Gemini service for multimodal document validation.

Uses the google-genai SDK (google.genai) with Gemini 2.5 Pro.
Files are passed as inline base64 data so no server-side file upload is needed.
"""
import base64
import json
import re
from functools import lru_cache

from google import genai
from google.genai import types

from app.config import get_settings


@lru_cache(maxsize=1)
def get_gemini_client() -> genai.Client:
    settings = get_settings()
    return genai.Client(api_key=settings.google_api_key)


def validate_document(
    file_bytes: bytes,
    mime_type: str,
    prompt: str,
) -> str:
    """
    Send a file (as inline base64) and a text prompt to Gemini 1.5 Pro.
    Returns the raw text response.
    """
    client = get_gemini_client()
    b64 = base64.b64encode(file_bytes).decode("utf-8")

    response = client.models.generate_content(
        model="gemini-2.5-pro-preview-03-25",
        contents=[
            types.Part.from_bytes(data=file_bytes, mime_type=mime_type),
            prompt,
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
        ),
    )
    return response.text


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

def build_apd_drawing_prompt(drawing_number: str, revision: str) -> str:
    return f"""You are validating an engineering drawing attached to a Purchase Requisition.
Drawing number expected: {drawing_number}
Revision expected: {revision} (treat 0, 00, and "NO REVISION" as equivalent base revisions)

Examine the drawing carefully and respond ONLY with a JSON object in exactly this format:
{{
  "drawing_number_match": "PASS" | "FAIL" | "UNCLEAR",
  "revision_match": "PASS" | "FAIL" | "UNCLEAR",
  "approval_stamp": "PASS" | "FAIL" | "UNCLEAR",
  "legible": "PASS" | "FAIL",
  "notes": "<brief explanation of any issues, or empty string>"
}}

Criteria:
- drawing_number_match: Is the drawing number {drawing_number} clearly visible on the drawing?
- revision_match: Is the revision {revision} (or equivalent) visible on the title block?
- approval_stamp: Is there an approval stamp, signature, or authorising engineer name on the title block?
- legible: Is the drawing legible — not heavily blurred, cropped, or illegible?"""


def build_ftp_costing_sheet_prompt() -> str:
    return """You are validating a First Time Purchase (FTP) costing sheet attached to a Purchase Requisition.

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


def build_pac_cert_prompt(pr_description: str = "") -> str:
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
    return f"""You are validating a PAC (Pre-Approved Certificate / Preferred Approved Contractor) certificate attached to a Purchase Requisition.

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


def build_scope_of_work_prompt() -> str:
    return """You are validating a Scope of Work document attached to a Service Purchase Requisition.

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


def build_jsa_prompt() -> str:
    return """You are validating a Job Safety Analysis (JSA) document attached to a Service Purchase Requisition.

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


def build_technical_skillset_prompt() -> str:
    return """You are validating a Technical Skill-Set document attached to a Service Purchase Requisition.

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
