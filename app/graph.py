"""
LangGraph StateGraph definition for the PR Attachment Validation Agent.

Graph structure:
  START → route_pr_type → [conditional] →
    apd:      apd_parse → apd_drive_search → apd_validate_docs → apd_secondary → format_response
    non_apd:  non_apd → format_response
    ftp:      ftp_validate → format_response
    pac:      pac_validate → format_response
    service:  service_validate → format_response
  → END
"""
from langgraph.graph import StateGraph, END

from app.state import PRValidationState
from app.nodes import router as router_module
from app.nodes import apd_nodes, non_apd_node, ftp_node, pac_node, service_node
from app.nodes.format_response import format_response


def build_graph() -> StateGraph:
    builder = StateGraph(PRValidationState)

    # ── Nodes ──────────────────────────────────────────────────────────────
    builder.add_node("route_pr_type",     router_module.route_pr_type)
    builder.add_node("apd_parse",         apd_nodes.parse_items)
    builder.add_node("apd_drive_search",  apd_nodes.search_drive)
    builder.add_node("apd_validate_docs", apd_nodes.validate_documents)
    builder.add_node("apd_secondary",     apd_nodes.secondary_validations)
    builder.add_node("non_apd",           non_apd_node.handle)
    builder.add_node("ftp_validate",      ftp_node.handle)
    builder.add_node("pac_validate",      pac_node.handle)
    builder.add_node("service_validate",  service_node.handle)
    builder.add_node("format_response",   format_response)

    # ── Entry point ─────────────────────────────────────────────────────────
    builder.set_entry_point("route_pr_type")

    # ── Conditional routing by PR type ──────────────────────────────────────
    builder.add_conditional_edges(
        "route_pr_type",
        router_module.pick_branch,
        {
            "apd":     "apd_parse",
            "non_apd": "non_apd",
            "ftp":     "ftp_validate",
            "pac":     "pac_validate",
            "service": "service_validate",
        },
    )

    # ── APD subflow edges ────────────────────────────────────────────────────
    builder.add_edge("apd_parse",         "apd_drive_search")
    builder.add_edge("apd_drive_search",  "apd_validate_docs")
    builder.add_edge("apd_validate_docs", "apd_secondary")
    builder.add_edge("apd_secondary",     "format_response")

    # ── Other PR type edges ──────────────────────────────────────────────────
    for node_name in ("non_apd", "ftp_validate", "pac_validate", "service_validate"):
        builder.add_edge(node_name, "format_response")

    # ── Terminal edge ────────────────────────────────────────────────────────
    builder.add_edge("format_response", END)

    return builder.compile()


# Compiled graph singleton — imported by main.py
graph = build_graph()
