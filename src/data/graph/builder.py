from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from src.data.entities.resolution import (
    build_entity_indexes,
    load_company_universe,
    normalize_text,
    resolve_company,
    resolve_event_company,
)
from src.data.extraction.entities import build_entity_id
from src.data.extraction.relations import validate_temporal_metadata
from src.data.graph.relationships import (
    build_relationship,
    build_semantic_relationship,
    is_valid_node_type,
    is_valid_relationship,
    normalize_entity_type,
)
from src.data.graph.schema import GraphEdge, GraphNode, TemporalGraph


def _stable_hash(value: Any) -> str:
    """Create a deterministic short hash."""

    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )

    return hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Semantic KG identifiers
# ---------------------------------------------------------------------------


def build_semantic_node_id(
    node_type: str,
    canonical_name: str,
    indexes: dict[str, Any] | None = None,
) -> str:
    """
    Build a canonical deterministic identifier for a semantic KG node.

    If node_type is 'Company' and resolves to a universe company, returns
    f'company:{ticker}' (e.g. company:NVDA, company:MSFT).
    Otherwise, returns the canonical extraction entity ID f'{normalized_type}:{digest}'.
    """

    canonical_type = normalize_entity_type(node_type)
    name = str(canonical_name).strip()

    if not name:
        raise ValueError(
            "Semantic node requires a non-empty canonical name."
        )

    if canonical_type == "Company":
        if indexes is None:
            try:
                indexes = build_entity_indexes(load_company_universe())
            except Exception:
                indexes = None

        if indexes:
            resolved = resolve_company(
                company=name,
                indexes=indexes,
            )
            if resolved:
                return resolved["entity_id"]

    return build_entity_id(canonical_type, name)


def build_semantic_node(
    node_type: str,
    canonical_name: str,
    *,
    entity_id: str | None = None,
    properties: dict[str, Any] | None = None,
    event_time: str | None = None,
    available_time: str | None = None,
    indexes: dict[str, Any] | None = None,
) -> GraphNode:
    """Build a semantic KG node with temporal metadata."""

    canonical_type = normalize_entity_type(node_type)
    name = str(canonical_name).strip()

    if not name:
        raise ValueError(
            "Semantic node requires a non-empty canonical name."
        )

    if entity_id:
        if canonical_type == "Company":
            node_id = build_semantic_node_id(
                canonical_type,
                name,
                indexes=indexes,
            )
            if not node_id.startswith("company:") or len(node_id) > 13:
                node_id = entity_id
        else:
            node_id = entity_id
    else:
        node_id = build_semantic_node_id(
            canonical_type,
            name,
            indexes=indexes,
        )

    node_properties = dict(properties or {})
    node_properties.setdefault(
        "canonical_name",
        name,
    )
    node_properties.setdefault(
        "name",
        name,
    )

    return GraphNode(
        node_id=node_id,
        node_type=canonical_type,
        properties=node_properties,
        event_time=event_time,
        available_time=available_time,
    )


def build_semantic_edge(
    source_id: str,
    target_id: str,
    relationship: str,
    *,
    event_time: str | None = None,
    available_time: str | None = None,
    properties: dict[str, Any] | None = None,
) -> GraphEdge:
    """Build a deterministic semantic KG edge."""

    source = str(source_id).strip()
    target = str(target_id).strip()
    rel = str(relationship).strip().upper()

    if not source:
        raise ValueError("Source entity ID cannot be empty.")

    if not target:
        raise ValueError("Target entity ID cannot be empty.")

    if source == target:
        raise ValueError("Self-referential relations are not allowed.")

    if not is_valid_relationship(rel):
        raise ValueError(
            f"Unsupported FinGraph relationship: {relationship}"
        )

    validate_temporal_metadata(event_time, available_time)

    edge_properties = dict(properties or {})

    edge_identity = {
        "source": source,
        "target": target,
        "relationship": rel,
        "event_time": event_time,
        "available_time": available_time,
        "properties": edge_properties,
    }

    edge_id = (
        f"{source}"
        f"->{rel}"
        f"->{target}"
        f":{_stable_hash(edge_identity)}"
    )

    relationship_data = build_semantic_relationship(
        source_id=source,
        target_id=target,
        relationship=rel,
        event_time=event_time,
        available_time=available_time,
        properties=edge_properties,
    )

    return GraphEdge(
        edge_id=edge_id,
        source=relationship_data["source"],
        target=relationship_data["target"],
        relationship=relationship_data["relationship"],
        event_time=relationship_data["event_time"],
        available_time=relationship_data["available_time"],
        properties=relationship_data["properties"],
    )


# ---------------------------------------------------------------------------
# Semantic graph builder
# ---------------------------------------------------------------------------


def build_semantic_graph(
    entities: Iterable[dict[str, Any]],
    relations: Iterable[dict[str, Any]],
    indexes: dict[str, Any] | None = None,
) -> TemporalGraph:
    """
    Build a temporal semantic knowledge graph from extracted entities
    and relations.

    Ensures 100% ID compatibility with extraction outputs, handles
    both extraction schema (entity_id, source_entity_id, target_entity_id)
    and graph schema (node_id, source, target), and rejects dangling,
    self-referential, and invalid relations.
    """

    if indexes is None:
        try:
            indexes = build_entity_indexes(load_company_universe())
        except Exception:
            indexes = None

    graph = TemporalGraph()
    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}
    alias_map: dict[str, str] = {}

    for entity in entities:
        node_type = entity.get("node_type") or entity.get("entity_type")
        if not node_type or not is_valid_node_type(node_type):
            continue

        canonical_type = normalize_entity_type(node_type)
        canonical_name = entity.get("canonical_name") or entity.get("name")
        if not canonical_name:
            continue

        raw_id = entity.get("entity_id") or entity.get("node_id")

        node = build_semantic_node(
            canonical_type,
            canonical_name,
            entity_id=raw_id,
            properties=entity.get("properties"),
            event_time=entity.get("event_time"),
            available_time=entity.get("available_time"),
            indexes=indexes,
        )

        node_id = node.node_id

        # Register mapping aliases
        if raw_id:
            alias_map[raw_id] = node_id
        alias_map[node_id] = node_id
        alias_map[canonical_name] = node_id
        alias_map[canonical_name.casefold()] = node_id
        name = entity.get("name")
        if name:
            alias_map[name] = node_id
            alias_map[name.casefold()] = node_id
        alias_map[f"{canonical_type.lower()}:{normalize_text(canonical_name)}"] = node_id

        if node_id in nodes_by_id:
            existing = nodes_by_id[node_id]
            merged_props = {**existing.properties, **node.properties}
            existing.properties = merged_props
            if node.available_time and (
                not existing.available_time
                or node.available_time < existing.available_time
            ):
                existing.available_time = node.available_time
        else:
            nodes_by_id[node_id] = node

    for relation in relations:
        source_ref = (
            relation.get("source_entity_id")
            or relation.get("source")
            or relation.get("source_entity")
        )
        target_ref = (
            relation.get("target_entity_id")
            or relation.get("target")
            or relation.get("target_entity")
        )

        if not source_ref or not target_ref:
            continue

        source_id = alias_map.get(source_ref) or alias_map.get(str(source_ref).casefold())
        target_id = alias_map.get(target_ref) or alias_map.get(str(target_ref).casefold())

        # Never create dangling edges.
        if not source_id or source_id not in nodes_by_id:
            continue
        if not target_id or target_id not in nodes_by_id:
            continue

        # Disallow self-referential relations
        if source_id == target_id:
            continue

        rel_name = str(relation.get("relationship", "")).strip().upper()
        if not is_valid_relationship(rel_name):
            continue

        event_time = relation.get("event_time")
        available_time = relation.get("available_time")

        try:
            validate_temporal_metadata(event_time, available_time)
        except ValueError:
            continue

        edge = build_semantic_edge(
            source_id=source_id,
            target_id=target_id,
            relationship=rel_name,
            event_time=event_time,
            available_time=available_time,
            properties=relation.get("properties"),
        )

        if edge.edge_id in edges_by_id:
            continue

        edges_by_id[edge.edge_id] = edge

    graph.nodes = list(nodes_by_id.values())
    graph.edges = list(edges_by_id.values())

    return graph


def build_unified_semantic_graph(
    extractions_dir: str | Path = "data/processed/extractions",
    universe_path: str | Path | None = None,
) -> TemporalGraph:
    """
    Build a unified multi-company Temporal Knowledge Graph from all
    SEC extraction JSON files.

    Preserves document ID, chunk provenance, event_time, available_time,
    publication_date, and fiscal_period.
    """

    ext_dir = Path(extractions_dir)
    if not ext_dir.exists():
        raise FileNotFoundError(
            f"Extractions directory not found: {ext_dir}"
        )

    files = sorted(ext_dir.glob("*.json"))
    if not files:
        raise RuntimeError(
            f"No extraction files found in {ext_dir}"
        )

    companies = (
        load_company_universe(universe_path)
        if universe_path
        else load_company_universe()
    )
    indexes = build_entity_indexes(companies)

    graph = TemporalGraph()
    nodes_by_id: dict[str, GraphNode] = {}
    edges_by_id: dict[str, GraphEdge] = {}
    alias_map: dict[str, str] = {}

    for file_path in files:
        with file_path.open("r", encoding="utf-8") as handle:
            chunks = json.load(handle)

        if not isinstance(chunks, list):
            continue

        for chunk in chunks:
            chunk_id = chunk.get("chunk_id")
            doc_id = chunk.get("document_id")
            provenance = chunk.get("provenance", {})
            pub_date = provenance.get("publication_date")
            avail_time = provenance.get("available_time")
            fiscal_period = chunk.get("metadata", {}).get("fiscal_period")

            chunk_alias_map: dict[str, str] = {}

            for entity in chunk.get("entities", []):
                node_type = entity.get("entity_type") or entity.get("node_type")
                if not node_type or not is_valid_node_type(node_type):
                    continue

                canonical_type = normalize_entity_type(node_type)
                canonical_name = entity.get("canonical_name") or entity.get("name")
                if not canonical_name:
                    continue

                raw_id = entity.get("entity_id") or entity.get("node_id")

                node = build_semantic_node(
                    canonical_type,
                    canonical_name,
                    entity_id=raw_id,
                    properties=entity.get("properties"),
                    event_time=entity.get("event_time"),
                    available_time=entity.get("available_time") or avail_time,
                    indexes=indexes,
                )

                node_id = node.node_id

                # Attach document and chunk provenance
                node_props = node.properties
                doc_ids = node_props.setdefault("document_ids", [])
                if doc_id and doc_id not in doc_ids:
                    doc_ids.append(doc_id)
                chunk_ids = node_props.setdefault("chunk_ids", [])
                if chunk_id and chunk_id not in chunk_ids:
                    chunk_ids.append(chunk_id)

                # Map aliases locally and globally
                if raw_id:
                    chunk_alias_map[raw_id] = node_id
                    alias_map[raw_id] = node_id
                chunk_alias_map[node_id] = node_id
                alias_map[node_id] = node_id
                chunk_alias_map[canonical_name] = node_id
                chunk_alias_map[canonical_name.casefold()] = node_id
                alias_map[canonical_name] = node_id
                alias_map[canonical_name.casefold()] = node_id

                name = entity.get("name")
                if name:
                    chunk_alias_map[name] = node_id
                    chunk_alias_map[name.casefold()] = node_id
                    alias_map[name] = node_id
                    alias_map[name.casefold()] = node_id

                if node_id in nodes_by_id:
                    existing = nodes_by_id[node_id]
                    existing_docs = existing.properties.setdefault("document_ids", [])
                    if doc_id and doc_id not in existing_docs:
                        existing_docs.append(doc_id)
                    existing_chunks = existing.properties.setdefault("chunk_ids", [])
                    if chunk_id and chunk_id not in existing_chunks:
                        existing_chunks.append(chunk_id)

                    if node.available_time and (
                        not existing.available_time
                        or node.available_time < existing.available_time
                    ):
                        existing.available_time = node.available_time
                else:
                    nodes_by_id[node_id] = node

            for relation in chunk.get("relations", []):
                source_ref = (
                    relation.get("source_entity_id")
                    or relation.get("source")
                    or relation.get("source_entity")
                )
                target_ref = (
                    relation.get("target_entity_id")
                    or relation.get("target")
                    or relation.get("target_entity")
                )

                if not source_ref or not target_ref:
                    continue

                source_id = (
                    chunk_alias_map.get(source_ref)
                    or chunk_alias_map.get(str(source_ref).casefold())
                    or alias_map.get(source_ref)
                    or alias_map.get(str(source_ref).casefold())
                )
                target_id = (
                    chunk_alias_map.get(target_ref)
                    or chunk_alias_map.get(str(target_ref).casefold())
                    or alias_map.get(target_ref)
                    or alias_map.get(str(target_ref).casefold())
                )

                if not source_id or source_id not in nodes_by_id:
                    continue
                if not target_id or target_id not in nodes_by_id:
                    continue
                if source_id == target_id:
                    continue

                rel_name = str(relation.get("relationship", "")).strip().upper()
                if not is_valid_relationship(rel_name):
                    continue

                rel_available = relation.get("available_time") or avail_time
                event_time = relation.get("event_time")

                try:
                    validate_temporal_metadata(event_time, rel_available)
                except ValueError:
                    continue

                edge_props = dict(relation.get("properties") or {})
                edge_props["document_id"] = doc_id
                edge_props["chunk_id"] = chunk_id
                edge_props["source"] = provenance.get("source", "sec")
                edge_props["source_url"] = provenance.get("source_url")
                edge_props["source_reference"] = provenance.get("source_reference")
                edge_props["publication_date"] = pub_date
                edge_props["fiscal_period"] = fiscal_period
                edge_props["confidence"] = relation.get("confidence", 1.0)

                edge = build_semantic_edge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=rel_name,
                    event_time=event_time,
                    available_time=rel_available,
                    properties=edge_props,
                )

                if edge.edge_id not in edges_by_id:
                    edges_by_id[edge.edge_id] = edge

    graph.nodes = list(nodes_by_id.values())
    graph.edges = list(edges_by_id.values())

    return graph


# ---------------------------------------------------------------------------
# Existing temporal event graph
#
# Kept intact for compatibility while the semantic KG pipeline is introduced.
# ---------------------------------------------------------------------------


def build_event_id(event: dict[str, Any]) -> str:
    """Build a deterministic identifier for a unified event."""

    event_type = event.get("event_type", "unknown")
    ticker = event.get("ticker") or event.get("data", {}).get("ticker", "")

    data = event.get("data", {})

    if event_type == "sec_filing":
        accession = data.get("accession_number")

        if accession:
            return f"sec:{ticker}:{accession}"

    if event_type == "market":
        date = event.get("event_time") or data.get("date")

        if date:
            return f"market:{ticker}:{date}"

    if event_type == "macro":
        series_id = data.get("series_id", "unknown")
        date = event.get("event_time") or data.get("date")

        if date:
            return f"macro:{series_id}:{date}"

    if event_type == "news":
        identity = {
            "ticker": ticker,
            "title": data.get("title"),
            "url": data.get("url"),
            "published_at": data.get("published_at"),
        }

        return f"news:{ticker}:{_stable_hash(identity)}"

    return f"{event_type}:{ticker}:{_stable_hash(event)}"


def build_event_node(event: dict[str, Any]) -> GraphNode:
    """Convert a unified event into a graph node."""

    event_id = build_event_id(event)
    event_type = event["event_type"]
    data = event.get("data", {})

    return GraphNode(
        node_id=event_id,
        node_type=event_type,
        properties=data,
        event_time=event.get("event_time"),
        available_time=event.get("available_time"),
    )


def build_company_node(
    company: dict[str, str],
) -> GraphNode:
    """Convert a company universe record into a graph node."""

    return GraphNode(
        node_id=company["entity_id"],
        node_type="company",
        properties={
            "ticker": company["ticker"],
            "company": company["company"],
            "sector": company["sector"],
            "cik": company["cik"],
        },
    )


def build_graph(
    events: list[dict[str, Any]],
    companies: list[dict[str, str]] | None = None,
) -> TemporalGraph:
    """
    Build the existing temporal event graph.

    This compatibility path remains available while FinGraph AI migrates
    to the semantic KG builder above.
    """

    if companies is None:
        companies = load_company_universe()

    indexes = build_entity_indexes(companies)

    graph = TemporalGraph()

    company_nodes: dict[str, GraphNode] = {}

    for company in companies:
        node = build_company_node(company)

        company_nodes[node.node_id] = node
        graph.nodes.append(node)

    seen_event_ids: set[str] = set()
    seen_edge_ids: set[str] = set()

    for event in events:
        resolved_event = resolve_event_company(
            event,
            indexes,
        )

        event_id = build_event_id(resolved_event)

        if event_id not in seen_event_ids:
            graph.nodes.append(
                build_event_node(resolved_event)
            )
            seen_event_ids.add(event_id)

        if not resolved_event.get("entity_resolved"):
            continue

        company_id = resolved_event.get("entity_id")

        if company_id not in company_nodes:
            continue

        relationship = build_relationship(
            company_id,
            {
                **resolved_event,
                "event_id": event_id,
            },
        )

        if relationship is None:
            continue

        edge_id = (
            f"{relationship['source']}"
            f"->{relationship['relationship']}"
            f"->{relationship['target']}"
        )

        if edge_id in seen_edge_ids:
            continue

        graph.edges.append(
            GraphEdge(
                edge_id=edge_id,
                source=relationship["source"],
                target=relationship["target"],
                relationship=relationship["relationship"],
                event_time=relationship["event_time"],
                available_time=relationship["available_time"],
                properties=relationship.get("properties", {}),
            )
        )

        seen_edge_ids.add(edge_id)

    return graph
