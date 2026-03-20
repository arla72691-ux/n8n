from pydantic import BaseModel, field_validator
from typing import Any


class N8nNode(BaseModel):
    id: str
    name: str
    type: str
    typeVersion: int = 1
    position: list[float]
    parameters: dict[str, Any] = {}
    credentials: dict[str, Any] = {}

    @field_validator("position")
    @classmethod
    def position_must_have_two_elements(cls, v: list) -> list:
        if len(v) != 2:
            raise ValueError("position must be [x, y]")
        return v

    @field_validator("id")
    @classmethod
    def id_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("node id must not be empty")
        return v


class N8nWorkflow(BaseModel):
    name: str
    nodes: list[N8nNode]
    connections: dict[str, Any] = {}
    settings: dict[str, Any] = {}

    @field_validator("nodes")
    @classmethod
    def nodes_must_not_be_empty(cls, v: list) -> list:
        if not v:
            raise ValueError("workflow must have at least one node")
        return v

    @field_validator("nodes")
    @classmethod
    def node_ids_must_be_unique(cls, v: list[N8nNode]) -> list[N8nNode]:
        ids = [node.id for node in v]
        if len(ids) != len(set(ids)):
            raise ValueError("all node ids must be unique")
        return v

    def validate_connections(self) -> list[str]:
        """Returns list of validation errors for connections."""
        errors = []
        node_names = {node.name for node in self.nodes}
        for source_name, outputs in self.connections.items():
            if source_name not in node_names:
                errors.append(f"Connection source '{source_name}' not found in nodes")
            if not isinstance(outputs, dict) or "main" not in outputs:
                errors.append(f"Connection for '{source_name}' must have 'main' key")
                continue
            for output_group in outputs["main"]:
                for conn in output_group:
                    target = conn.get("node")
                    if target and target not in node_names:
                        errors.append(f"Connection target '{target}' not found in nodes")
        return errors

    def to_api_dict(self) -> dict:
        return self.model_dump(exclude_none=True)
