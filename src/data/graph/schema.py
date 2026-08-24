from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class GraphNode:
    """A node in the FinGraph temporal knowledge graph."""

    node_id: str
    node_type: str
    properties: dict[str, Any] = field(default_factory=dict)
    event_time: str | None = None
    available_time: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the node into a JSON-compatible dictionary."""

        return {
            "node_id": self.node_id,
            "node_type": self.node_type,
            "properties": self.properties,
            "event_time": self.event_time,
            "available_time": self.available_time,
        }


@dataclass
class GraphEdge:
    """A temporal relationship between two graph nodes."""

    edge_id: str
    source: str
    target: str
    relationship: str
    event_time: str | None = None
    available_time: str | None = None
    properties: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the edge into a JSON-compatible dictionary."""

        return {
            "edge_id": self.edge_id,
            "source": self.source,
            "target": self.target,
            "relationship": self.relationship,
            "event_time": self.event_time,
            "available_time": self.available_time,
            "properties": self.properties,
        }


@dataclass
class TemporalGraph:
    """Container for the FinGraph temporal knowledge graph."""

    nodes: list[GraphNode] = field(default_factory=list)
    edges: list[GraphEdge] = field(default_factory=list)
    as_of: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize the complete graph."""

        return {
            "nodes": [node.to_dict() for node in self.nodes],
            "edges": [edge.to_dict() for edge in self.edges],
            "as_of": self.as_of,
        }

    @property
    def node_count(self) -> int:
        """Return the number of graph nodes."""

        return len(self.nodes)

    @property
    def edge_count(self) -> int:
        """Return the number of graph edges."""

        return len(self.edges)