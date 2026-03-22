"""LangGraph StateGraph for PR Attachment Validation.

Replaces the if/elif routing in server.py with an explicit DAG:

  START → route_pr_type() → [non_apd | apd | ftp | pac | service] → END

Each node calls the corresponding validator from validators.py and
writes the result back into PRState.
"""
from typing import TypedDict
from opentelemetry import trace

from langgraph.graph import StateGraph, START, END
import validators as val

tracer = trace.get_tracer(__name__)


# ── State ────────────────────────────────────────────────────────────────────

class PRState(TypedDict):
    pr_type: str
    pr_number: str
    item_long_text: str
    files: list           # list of (docType: str, bytes, mimeType: str)
    html_output: str
    is_valid: bool
    error: str | None


# ── Routing ───────────────────────────────────────────────────────────────────

def route_pr_type(state: PRState) -> str:
    pr = state["pr_type"]
    if "Non-APD" in pr:
        return "non_apd"
    if "APD" in pr:
        return "apd"
    if "FTP" in pr:
        return "ftp"
    if "PAC" in pr:
        return "pac"
    if "Service" in pr:
        return "service"
    raise ValueError(f"Unknown PR type: {pr!r}")


# ── Nodes ─────────────────────────────────────────────────────────────────────

async def non_apd_node(state: PRState) -> dict:
    with tracer.start_as_current_span("validator.non_apd") as span:
        span.set_attribute("pr.number", state["pr_number"])
        html, ok = await val.validate_non_apd(state["pr_number"])
        span.set_attribute("validator.is_valid", ok)
        return {"html_output": html, "is_valid": ok, "error": None}


async def apd_node(state: PRState) -> dict:
    with tracer.start_as_current_span("validator.apd") as span:
        span.set_attribute("pr.number", state["pr_number"])
        span.set_attribute("pr.item_long_text_len", len(state.get("item_long_text") or ""))
        span.set_attribute("pr.files_count", len(state.get("files") or []))
        html, ok = await val.validate_apd(
            state["pr_number"],
            state.get("item_long_text") or "",
            state.get("files") or [],
        )
        span.set_attribute("validator.is_valid", ok)
        return {"html_output": html, "is_valid": ok, "error": None}


async def ftp_node(state: PRState) -> dict:
    with tracer.start_as_current_span("validator.ftp") as span:
        span.set_attribute("pr.number", state["pr_number"])
        span.set_attribute("pr.files_count", len(state.get("files") or []))
        html, ok = await val.validate_ftp(
            state["pr_number"],
            state.get("files") or [],
        )
        span.set_attribute("validator.is_valid", ok)
        return {"html_output": html, "is_valid": ok, "error": None}


async def pac_node(state: PRState) -> dict:
    with tracer.start_as_current_span("validator.pac") as span:
        span.set_attribute("pr.number", state["pr_number"])
        span.set_attribute("pr.files_count", len(state.get("files") or []))
        html, ok = await val.validate_pac(
            state["pr_number"],
            state.get("files") or [],
        )
        span.set_attribute("validator.is_valid", ok)
        return {"html_output": html, "is_valid": ok, "error": None}


async def service_node(state: PRState) -> dict:
    with tracer.start_as_current_span("validator.service") as span:
        span.set_attribute("pr.number", state["pr_number"])
        span.set_attribute("pr.files_count", len(state.get("files") or []))
        html, ok = await val.validate_service(
            state["pr_number"],
            state.get("files") or [],
        )
        span.set_attribute("validator.is_valid", ok)
        return {"html_output": html, "is_valid": ok, "error": None}


# ── Graph compilation ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(PRState)

    builder.add_node("non_apd", non_apd_node)
    builder.add_node("apd", apd_node)
    builder.add_node("ftp", ftp_node)
    builder.add_node("pac", pac_node)
    builder.add_node("service", service_node)

    builder.add_conditional_edges(
        START,
        route_pr_type,
        {
            "non_apd": "non_apd",
            "apd": "apd",
            "ftp": "ftp",
            "pac": "pac",
            "service": "service",
        },
    )

    for node_name in ("non_apd", "apd", "ftp", "pac", "service"):
        builder.add_edge(node_name, END)

    return builder.compile()


pr_validation_graph = build_graph()
