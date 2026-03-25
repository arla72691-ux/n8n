"""
Langfuse direct-SDK helpers.

This module provides:
- get_langfuse_client(): cached Langfuse SDK client (separate from the LangChain CallbackHandler)
- get_prompt(): fetch + compile a prompt from Langfuse, falling back to a hardcoded string
- seed_prompts(): push all hardcoded prompts into Langfuse (run once to populate the UI)
"""
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

# Langfuse prompt names — must match what's stored in Langfuse Prompt Management
PROMPT_APD_DRAWING = "pr-validation-apd-drawing"
PROMPT_FTP_COSTING = "pr-validation-ftp-costing-sheet"
PROMPT_PAC_CERT = "pr-validation-pac-cert"
PROMPT_SCOPE_OF_WORK = "pr-validation-scope-of-work"
PROMPT_JSA = "pr-validation-jsa"
PROMPT_TECHNICAL_SKILLSET = "pr-validation-technical-skillset"


@lru_cache(maxsize=1)
def get_langfuse_client():
    """Return a cached Langfuse SDK client, or None if Langfuse is not configured."""
    from app.config import get_settings
    settings = get_settings()
    if not settings.langfuse_enabled:
        return None
    try:
        from langfuse import Langfuse
        client = Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
        return client
    except Exception as exc:
        logger.warning(f"Failed to initialise Langfuse client: {exc}")
        return None


def get_prompt(name: str, fallback: str, **variables) -> str:
    """
    Fetch a prompt from Langfuse Prompt Management and compile it with the given variables.
    Falls back to `fallback` if Langfuse is unavailable or the prompt doesn't exist.

    Langfuse prompts use {{variable}} syntax; compile() substitutes them.
    """
    client = get_langfuse_client()
    if client is None:
        return fallback
    try:
        prompt_obj = client.get_prompt(name)
        return prompt_obj.compile(**variables)
    except Exception:
        return fallback


def get_prompt_and_client(name: str, fallback: str, **variables):
    """
    Like get_prompt(), but also returns the raw PromptClient object so callers
    can pass it to a Langfuse generation for observation tracking.

    Returns: (compiled_str, PromptClient | None)
    """
    client = get_langfuse_client()
    if client is None:
        return fallback, None
    try:
        prompt_obj = client.get_prompt(name)
        return prompt_obj.compile(**variables), prompt_obj
    except Exception:
        return fallback, None


def seed_prompts():
    """
    Push all hardcoded prompts into Langfuse Prompt Management (idempotent — creates if absent).
    Call this once to populate the Langfuse UI so prompts can be edited there.

    Controlled by the SEED_PROMPTS environment variable:
        SEED_PROMPTS=true uvicorn app.main:app ...
    """
    client = get_langfuse_client()
    if client is None:
        logger.warning("seed_prompts: Langfuse not configured, skipping.")
        return

    # Import hardcoded templates (converted to {{var}} syntax for Langfuse)
    prompts = _get_prompt_templates()
    for name, template in prompts.items():
        try:
            # get_prompt will raise NotFoundError if absent; create it then
            client.get_prompt(name)
            logger.info(f"seed_prompts: prompt '{name}' already exists, skipping.")
        except Exception:
            try:
                client.create_prompt(name=name, prompt=template, labels=["production"], type="text")
                logger.info(f"seed_prompts: created prompt '{name}'.")
            except Exception as exc:
                logger.warning(f"seed_prompts: failed to create '{name}': {exc}")


def _get_prompt_templates() -> dict:
    """
    Return all prompt templates in Langfuse {{variable}} format.
    These are the source-of-truth templates; edit in Langfuse UI after seeding.
    """
    return {
        PROMPT_APD_DRAWING: (
            "You are validating an engineering drawing attached to an APD Purchase Requisition.\n\n"
            "Expected values from the Purchase Requisition item long text:\n"
            "- Drawing number: {{drawing_number}}\n"
            "- Revision: {{revision}} (treat 0, 00, and \"NO REVISION\" as equivalent base revisions)\n"
            "- Position/item number(s) to verify: {{part_numbers}}\n\n"
            "─── ENGINEERING DRAWING LAYOUT — WHERE TO LOOK ────────────────────────────────\n\n"
            "PARTS LIST / SCHEDULE TABLE — it can appear in ANY of these locations:\n\n"
            "  1. FULL-HEIGHT VERTICAL STRIP on the RIGHT SIDE of the sheet (IEC/ISO/DIN/European\n"
            "     standard, very common in Latin America and Europe). The table runs as a column\n"
            "     along the right edge, from near the top of the sheet down to just above the title\n"
            "     block. This is a TALL NARROW TABLE on the right margin.\n\n"
            "  2. HORIZONTAL BLOCK in the LOWER RIGHT corner, directly above the title block\n"
            "     (ASME/ANSI/BS standard, common in North America and UK).\n\n"
            "  3. UPPER RIGHT corner as a stacked horizontal block.\n\n"
            "  In ALL cases the table typically reads BOTTOM TO TOP: item 1 is the BOTTOM row,\n"
            "  higher item numbers are in the rows above it.\n\n"
            "TABLE IDENTIFICATION — it is a bordered grid and its header row may be labelled:\n"
            "  English:  \"PARTS LIST\", \"BILL OF MATERIALS\", \"BOM\", \"SCHEDULE\", \"MATERIAL LIST\",\n"
            "            \"COMPONENT LIST\", \"ASSEMBLY LIST\", \"ITEM LIST\"\n"
            "  Spanish:  \"LISTA DE MATERIALES\", \"LISTA DE PARTES\", \"LISTA DE COMPONENTES\",\n"
            "            \"MATERIALES\", \"DESPIECE\", \"NOMENCLATURA\", \"DESCRIPCION\"\n\n"
            "The item-number column header may read:\n"
            "  English:  \"ITEM\", \"ITEM NO.\", \"NO.\", \"POS.\", \"FIND NO.\"\n"
            "  Spanish:  \"ITEM\", \"POSICION\", \"POS.\", \"Nº\", \"No.\"\n"
            "  Or it may be an unlabelled leftmost column of sequential numbers.\n\n"
            "BALLOON CALLOUTS (secondary source):\n"
            "  Circled, hexagonal, or flag-shaped numbers on the drawing view itself that point\n"
            "  to individual components — these match the item number column in the parts list.\n\n"
            "TITLE BLOCK: lower right corner — drawing number, revision, date, approvals.\n"
            "REVISION TABLE: upper right corner — revision history.\n\n"
            "─── NORMALISATION RULES ────────────────────────────────────────────────────────\n"
            "• \"P3\" in the PR == \"3\" in the table; \"01\" == \"1\".\n\n"
            "────────────────────────────────────────────────────────────────────────────────\n\n"
            "Scan the ENTIRE sheet — pay special attention to the RIGHT EDGE (full-height strip)\n"
            "and the LOWER RIGHT CORNER — then respond ONLY with a JSON object in exactly this format:\n"
            "{\n"
            "  \"has_drawings\": \"PASS\" | \"FAIL\",\n"
            "  \"drawing_number_match\": \"PASS\" | \"FAIL\" | \"UNCLEAR\",\n"
            "  \"revision_match\": \"PASS\" | \"FAIL\" | \"UNCLEAR\",\n"
            "  \"position_number_match\": \"PASS\" | \"FAIL\" | \"UNCLEAR\" | \"NOT_CHECKED\",\n"
            "  \"found_drawing_number\": \"<drawing number found on document, or null>\",\n"
            "  \"found_revision\": \"<revision found on document, or null>\",\n"
            "  \"found_position_numbers\": \"<ALL item/position numbers you can read from the parts list or balloon callouts, or null>\",\n"
            "  \"notes\": \"<brief explanation of any issues, or empty string>\"\n"
            "}\n\n"
            "Criteria:\n"
            "- has_drawings: Are actual engineering drawings present? FAIL if pages are blank or contain no geometry.\n"
            "- drawing_number_match: Does the drawing number in the title block match {{drawing_number}}?\n"
            "- revision_match: Does the revision in the title block or revision table match {{revision}} (treat 0, 00, NO REVISION as equivalent)?\n"
            "- position_number_match:\n"
            "    Check the right-side vertical strip AND the lower-right block AND any balloon callouts.\n"
            "    Does the item-number column contain ALL of: {{part_numbers}}?\n"
            "    PASS   — every required position number is found.\n"
            "    FAIL   — one or more are missing (report what you did find in found_position_numbers).\n"
            "    UNCLEAR — a table or balloon exists but is too blurry/small to read individual entries.\n"
            "    NOT_CHECKED — ONLY when there is genuinely no bordered table and no balloon callouts\n"
            "    anywhere on the entire sheet. DO NOT use if any numbered table or circled numbers exist."
        ),
        PROMPT_FTP_COSTING: (
            "You are validating a First Time Purchase (FTP) costing sheet attached to a Purchase Requisition.\n\n"
            "Examine the document carefully and respond ONLY with a JSON object in exactly this format:\n"
            "{\n"
            "  \"cost_breakdown_present\": \"PASS\" | \"FAIL\",\n"
            "  \"cost_breakdown_details\": \"<what cost components were found or what is missing>\",\n"
            "  \"validity_date_present\": \"PASS\" | \"FAIL\",\n"
            "  \"validity_date\": \"<the date found, or null if not present>\",\n"
            "  \"authorising_signature\": \"PASS\" | \"FAIL\" | \"UNCLEAR\",\n"
            "  \"notes\": \"<any additional observations>\"\n"
            "}\n\n"
            "Criteria:\n"
            "- cost_breakdown_present: Does the document contain a cost breakdown with material, processing, overhead, and total unit cost?\n"
            "- validity_date_present: Is there a validity or effective date visible?\n"
            "- authorising_signature: Is there an authorising signature or approver name visible?"
        ),
        PROMPT_PAC_CERT: (
            "You are validating a PAC (Pre-Approved Certificate / Preferred Approved Contractor) certificate attached to a Purchase Requisition.\n\n"
            "Examine the document carefully and respond ONLY with a JSON object in exactly this format:\n"
            "{\n"
            "  \"pac_cert_number\": \"PASS\" | \"FAIL\",\n"
            "  \"vendor_name\": \"PASS\" | \"FAIL\",\n"
            "  \"expiry_date_present\": \"PASS\" | \"FAIL\",\n"
            "  \"expiry_date\": \"<the expiry/validity date found, or null>\",\n"
            "  \"authorising_signature\": \"PASS\" | \"FAIL\" | \"UNCLEAR\",\n"
            "  \"scope_match\": \"PASS\" | \"FAIL\" | \"UNCLEAR\" | \"NOT_CHECKED\",\n"
            "  \"notes\": \"<any additional observations>\"\n"
            "}\n\n"
            "Criteria:\n"
            "- pac_cert_number: Is a PAC certificate number visible on the document?\n"
            "- vendor_name: Is a vendor or company name visible?\n"
            "- expiry_date_present: Is a validity or expiry date present on the document?\n"
            "- authorising_signature: Is there an authorising signature or approver name?\n"
            "- scope_match: Does the scope/description on the certificate broadly match the PR description: \"{{pr_description}}\"?"
        ),
        PROMPT_SCOPE_OF_WORK: (
            "You are validating a Scope of Work document attached to a Service Purchase Requisition.\n\n"
            "Examine the document carefully and respond ONLY with a JSON object in exactly this format:\n"
            "{\n"
            "  \"deliverables_present\": \"PASS\" | \"FAIL\",\n"
            "  \"timeline_present\": \"PASS\" | \"FAIL\",\n"
            "  \"acceptance_criteria_present\": \"PASS\" | \"FAIL\",\n"
            "  \"vendor_name\": \"<vendor or company name found, or null>\",\n"
            "  \"work_description_summary\": \"<brief 1-2 sentence summary of the work described>\",\n"
            "  \"notes\": \"<any additional observations>\"\n"
            "}\n\n"
            "Criteria:\n"
            "- deliverables_present: Does the document contain identifiable deliverables or work outputs?\n"
            "- timeline_present: Is there a timeline, duration, or schedule mentioned?\n"
            "- acceptance_criteria_present: Are there acceptance criteria or sign-off conditions mentioned?"
        ),
        PROMPT_JSA: (
            "You are validating a Job Safety Analysis (JSA) document attached to a Service Purchase Requisition.\n\n"
            "Examine the document carefully and respond ONLY with a JSON object in exactly this format:\n"
            "{\n"
            "  \"hazards_identified\": \"PASS\" | \"FAIL\",\n"
            "  \"control_measures_present\": \"PASS\" | \"FAIL\",\n"
            "  \"safety_officer_signature\": \"PASS\" | \"FAIL\" | \"UNCLEAR\",\n"
            "  \"vendor_name\": \"<vendor or company name found, or null>\",\n"
            "  \"work_description_summary\": \"<brief 1-2 sentence summary of the work covered>\",\n"
            "  \"notes\": \"<any additional observations>\"\n"
            "}\n\n"
            "Criteria:\n"
            "- hazards_identified: Does the JSA identify specific hazards associated with the work?\n"
            "- control_measures_present: Are control measures or mitigations listed for the identified hazards?\n"
            "- safety_officer_signature: Is there a safety officer, HSE authority, or supervisor signature visible?"
        ),
        PROMPT_TECHNICAL_SKILLSET: (
            "You are validating a Technical Skill-Set document attached to a Service Purchase Requisition.\n\n"
            "Examine the document carefully and respond ONLY with a JSON object in exactly this format:\n"
            "{\n"
            "  \"qualifications_listed\": \"PASS\" | \"FAIL\",\n"
            "  \"certifications_listed\": \"PASS\" | \"FAIL\",\n"
            "  \"vendor_name\": \"<vendor or company name found, or null>\",\n"
            "  \"work_description_summary\": \"<brief 1-2 sentence summary of the skills/roles described>\",\n"
            "  \"notes\": \"<any additional observations>\"\n"
            "}\n\n"
            "Criteria:\n"
            "- qualifications_listed: Does the document list required qualifications for the roles involved?\n"
            "- certifications_listed: Does the document specify required certifications or licences for the roles?"
        ),
    }
