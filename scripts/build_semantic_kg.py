from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Ensure project root is on sys.path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.graph.builder import build_unified_semantic_graph
from src.data.graph.storage import save_graph


DEFAULT_EXTRACTION_DIR = ROOT / "data" / "processed" / "extractions"
DEFAULT_OUTPUT_PATH = (
    ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build unified semantic Temporal Knowledge Graph from SEC extractions."
    )
    parser.add_argument(
        "--extractions",
        type=Path,
        default=DEFAULT_EXTRACTION_DIR,
        help="Path to extracted chunk JSON files directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Path to save the compiled Temporal Knowledge Graph JSON.",
    )
    args = parser.parse_args()

    print("=" * 75)
    print("FinGraph AI — Unified Temporal Knowledge Graph Builder")
    print("=" * 75)
    print(f"Extraction Directory: {args.extractions}")
    print(f"Target Output Path:   {args.output}")

    files = sorted(args.extractions.glob("*.json"))
    print(f"Extraction Files:     {len(files)}")

    start_time = time.perf_counter()
    graph = build_unified_semantic_graph(extractions_dir=args.extractions)
    elapsed = time.perf_counter() - start_time

    # Save compiled KG
    saved_path = save_graph(graph, args.output)

    # Compute graph statistics
    node_types: dict[str, int] = {}
    nodes_with_avail: int = 0
    for node in graph.nodes:
        node_types[node.node_type] = node_types.get(node.node_type, 0) + 1
        if node.available_time:
            nodes_with_avail += 1

    rel_types: dict[str, int] = {}
    edges_with_avail: int = 0
    edges_with_event: int = 0
    edges_with_doc: int = 0
    edges_with_chunk: int = 0

    for edge in graph.edges:
        rel_types[edge.relationship] = rel_types.get(edge.relationship, 0) + 1
        if edge.available_time:
            edges_with_avail += 1
        if edge.event_time:
            edges_with_event += 1
        if edge.properties.get("document_id"):
            edges_with_doc += 1
        if edge.properties.get("chunk_id"):
            edges_with_chunk += 1

    print("\n" + "=" * 75)
    print("BUILD SUMMARY & INTEGRITY REPORT")
    print("=" * 75)
    print(f"Elapsed Time:               {elapsed:.2f} s")
    print(f"Total Graph Nodes:          {graph.node_count}")
    print(f"Total Graph Edges:          {graph.edge_count}")
    print(f"Nodes with Available Time:  {nodes_with_avail} ({nodes_with_avail/max(1, graph.node_count)*100:.1f}%)")
    print(f"Edges with Available Time:  {edges_with_avail} ({edges_with_avail/max(1, graph.edge_count)*100:.1f}%)")
    print(f"Edges with Event Time:      {edges_with_event} ({edges_with_event/max(1, graph.edge_count)*100:.1f}%)")
    print(f"Edges with Doc Provenance:  {edges_with_doc} ({edges_with_doc/max(1, graph.edge_count)*100:.1f}%)")
    print(f"Edges with Chunk Provenance:{edges_with_chunk} ({edges_with_chunk/max(1, graph.edge_count)*100:.1f}%)")

    print("\nNode Types Distribution:")
    for ntype, count in sorted(node_types.items(), key=lambda x: -x[1]):
        print(f"  {ntype:20}: {count}")

    print("\nRelationship Types Distribution:")
    for rtype, count in sorted(rel_types.items(), key=lambda x: -x[1]):
        print(f"  {rtype:20}: {count}")

    print(f"\nSaved Unified KG to: {saved_path}")
    print("=" * 75)


if __name__ == "__main__":
    main()
