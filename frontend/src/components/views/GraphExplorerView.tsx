"use client";

import React, { memo, useCallback, useEffect, useMemo, useRef, useState } from "react";
import ReactFlow, {
  Background,
  BackgroundVariant,
  BaseEdge,
  Edge,
  EdgeProps,
  Handle,
  MarkerType,
  Node,
  NodeProps,
  Position,
  ReactFlowInstance,
} from "reactflow";
import {
  BookOpen,
  Crosshair,
  ExternalLink,
  Focus,
  Maximize2,
  Minus,
  PanelRightClose,
  PanelRightOpen,
  Plus,
  RotateCcw,
  Search,
  Share2,
  X,
} from "lucide-react";
import { api, ApiError, isAbortError } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { useTokens } from "@/lib/useTokens";
import { NavTab } from "@/lib/navigation";
import { useResearchContext, ResearchEntity } from "@/lib/research-context";
import {
  CORPUS_ISSUERS,
  REL_FAMILY_LABEL,
  RelFamily,
  fmtDate,
  fmtDateTime,
  fmtInt,
  fmtNum,
  humanize,
  nodeTicker,
  relColor,
  relFamily,
  relLabel,
} from "@/lib/format";
import { GraphEdgeItem, GraphNodeItem, SubgraphResponse } from "@/lib/types";
import {
  Badge,
  Button,
  Drawer,
  EmptyState,
  ErrorState,
  IconButton,
  Input,
  KeyValue,
  ProgressBar,
  Segmented,
  Spinner,
  cx,
} from "@/components/ui";

/* ------------------------------------------------------------------ */
/* Constants                                                           */
/* ------------------------------------------------------------------ */

const PAGE_SIZE = 100; // edges per API page
const NEIGHBOUR_STEP = 24; // neighbours revealed per "show more"
const EXPANSION_LIMIT = 40; // edges fetched when expanding a neighbour
const EXPANSION_SHOWN = 8; // neighbours shown per expansion
const FAMILIES: RelFamily[] = ["acquired", "competes", "supplies", "partners", "invested", "risk", "other"];

type Scope = "companies" | "all";

/* ------------------------------------------------------------------ */
/* Aggregation                                                         */
/* ------------------------------------------------------------------ */

interface Neighbour {
  id: string;
  node: GraphNodeItem | undefined;
  edges: GraphEdgeItem[];
  outgoing: number; // focal → neighbour
  incoming: number; // neighbour → focal
  family: RelFamily; // dominant family
}

function aggregate(center: string, edges: GraphEdgeItem[], nodes: Map<string, GraphNodeItem>): Neighbour[] {
  const by = new Map<string, Neighbour>();
  for (const e of edges) {
    const other = e.source === center ? e.target : e.source;
    if (other === center) continue;
    let n = by.get(other);
    if (!n) {
      n = { id: other, node: nodes.get(other), edges: [], outgoing: 0, incoming: 0, family: "other" };
      by.set(other, n);
    }
    n.edges.push(e);
    if (e.source === center) n.outgoing++;
    else n.incoming++;
  }
  by.forEach((n) => {
    const counts = new Map<RelFamily, number>();
    n.edges.forEach((e) => counts.set(relFamily(e.relationship), (counts.get(relFamily(e.relationship)) ?? 0) + 1));
    n.family = Array.from(counts.entries()).sort((a, b) => b[1] - a[1])[0][0];
  });
  // Strongest first: more supporting edges, then higher graph degree.
  return Array.from(by.values()).sort(
    (a, b) => b.edges.length - a.edges.length || (b.node?.degree ?? 0) - (a.node?.degree ?? 0) || a.id.localeCompare(b.id)
  );
}

/* ------------------------------------------------------------------ */
/* Nodes                                                               */
/* ------------------------------------------------------------------ */

interface EntityNodeData {
  label: string;
  sub?: string | null;
  family: RelFamily;
  role: "focal" | "neighbour" | "expansion";
  selected: boolean;
  dimmed: boolean;
  expanded?: boolean;
}

const CENTER_HANDLE: React.CSSProperties = { left: "50%", top: "50%", right: "auto", bottom: "auto", transform: "translate(-50%,-50%)" };
const DOT_HANDLE: React.CSSProperties = { left: 7, top: "50%", right: "auto", transform: "translate(-50%,-50%)" };
const DOT_HANDLE_SM: React.CSSProperties = { left: 5, top: "50%", right: "auto", transform: "translate(-50%,-50%)" };

const EntityNode = memo(function EntityNode({ data }: NodeProps<EntityNodeData>) {
  if (data.role === "focal") {
    return (
      <div
        className={cx(
          "rounded-md border-2 border-accent bg-surface px-3.5 py-2 text-center min-w-[150px] max-w-[220px] shadow-overlay",
          data.selected && "ring-4 ring-accent/20"
        )}
      >
        <Handle type="target" position={Position.Top} style={CENTER_HANDLE} />
        <Handle type="source" position={Position.Bottom} style={CENTER_HANDLE} />
        <div className="text-sm font-semibold text-fg leading-tight truncate">{data.label}</div>
        {data.sub && <div className="font-mono-num text-2xs text-fg-3 mt-0.5 truncate">{data.sub}</div>}
      </div>
    );
  }
  const small = data.role === "expansion";
  return (
    <div
      className={cx("flex items-center gap-1.5 transition-opacity", data.dimmed ? "opacity-25" : "opacity-100")}
      style={{ maxWidth: small ? 150 : 180 }}
    >
      <Handle type="target" position={Position.Left} style={small ? DOT_HANDLE_SM : DOT_HANDLE} />
      <Handle type="source" position={Position.Right} style={small ? DOT_HANDLE_SM : DOT_HANDLE} />
      <span
        className={cx("rounded-full shrink-0 border-2", small ? "w-2.5 h-2.5" : "w-3.5 h-3.5", data.selected && "ring-4 ring-accent/30")}
        style={{
          background: relColor(data.family, data.expanded ? 1 : 0.22),
          borderColor: relColor(data.family),
        }}
      />
      <span
        className={cx(
          "truncate rounded-sm px-1 leading-tight",
          small ? "text-2xs text-fg-2" : "text-xs text-fg",
          data.selected ? "bg-accent/15 text-fg font-medium" : "bg-canvas/70"
        )}
      >
        {data.label}
      </span>
    </div>
  );
});

const NODE_TYPES = { entity: EntityNode };

/** Half-extent of the shape an edge must stop at (focal = box, others = dot). */
interface Extent {
  hw: number;
  hh: number;
}
const FOCAL_EXTENT: Extent = { hw: 84, hh: 26 };
const DOT_EXTENT: Extent = { hw: 10, hh: 10 };
const DOT_EXTENT_SM: Extent = { hw: 8, hh: 8 };

function trim(dx: number, dy: number, e: Extent) {
  const ax = Math.abs(dx) || 1e-6;
  const ay = Math.abs(dy) || 1e-6;
  return Math.min(e.hw / ax, e.hh / ay); // fraction of the vector inside the shape
}

/** Straight edge shortened to both node boundaries so direction arrows stay visible. */
function TrimmedEdge({ id, sourceX, sourceY, targetX, targetY, data, style, markerEnd, markerStart, label, labelStyle, labelBgPadding, labelBgBorderRadius, labelBgStyle, interactionWidth }: EdgeProps<{ sourceExtent: Extent; targetExtent: Extent }>) {
  const dx = targetX - sourceX;
  const dy = targetY - sourceY;
  const ts = trim(dx, dy, data?.sourceExtent ?? DOT_EXTENT);
  const tt = trim(dx, dy, data?.targetExtent ?? DOT_EXTENT);
  const sx = sourceX + dx * ts;
  const sy = sourceY + dy * ts;
  const tx = targetX - dx * tt;
  const ty = targetY - dy * tt;
  return (
    <BaseEdge
      id={id}
      path={`M ${sx},${sy} L ${tx},${ty}`}
      style={style}
      markerEnd={markerEnd}
      markerStart={markerStart}
      label={label}
      labelX={(sx + tx) / 2}
      labelY={(sy + ty) / 2}
      labelStyle={labelStyle}
      labelBgPadding={labelBgPadding}
      labelBgBorderRadius={labelBgBorderRadius}
      labelBgStyle={labelBgStyle}
      interactionWidth={interactionWidth}
    />
  );
}

const EDGE_TYPES = { trimmed: TrimmedEdge };

/* ------------------------------------------------------------------ */
/* Layout                                                              */
/* ------------------------------------------------------------------ */

function ringLayout(count: number): { x: number; y: number; angle: number }[] {
  // Rings sized so neighbours keep ≥ ~110px arc spacing.
  const out: { x: number; y: number; angle: number }[] = [];
  let placed = 0;
  let ring = 0;
  while (placed < count) {
    const radius = 230 + ring * 150;
    const capacity = Math.max(8, Math.floor((2 * Math.PI * radius) / 120));
    const n = Math.min(capacity, count - placed);
    const offset = ring % 2 === 0 ? -Math.PI / 2 : -Math.PI / 2 + Math.PI / n;
    for (let i = 0; i < n; i++) {
      const angle = offset + (2 * Math.PI * i) / n;
      out.push({ x: Math.cos(angle) * radius, y: Math.sin(angle) * radius, angle });
    }
    placed += n;
    ring++;
  }
  return out;
}

/* ------------------------------------------------------------------ */
/* Inspector                                                           */
/* ------------------------------------------------------------------ */

type Selection = { kind: "node"; id: string } | { kind: "edge"; neighbourId: string; parent: string } | null;

function EdgeRecord({ e, labels, onOpenFiling }: { e: GraphEdgeItem; labels: (id: string) => string; onOpenFiling: (doc: string, chunk?: string | null) => void }) {
  const props = e.properties || {};
  const verified = props.provenance_verified as boolean | undefined;
  const attribution = props.attribution_status as string | undefined;
  const sourceCompany = props.source_company as string | undefined;
  const fam = relFamily(e.relationship);
  return (
    <li className="py-3 border-b border-line-soft last:border-0">
      <div className="flex items-center gap-2 text-sm">
        <span className="w-2 h-2 rounded-full" style={{ background: relColor(fam) }} />
        <span className="font-medium">{relLabel(e.relationship)}</span>
        <span className="text-fg-3 text-xs truncate">
          {labels(e.source)} → {labels(e.target)}
        </span>
      </div>
      <KeyValue
        dense
        className="mt-1.5"
        items={[
          {
            label: "Confidence",
            value: e.confidence == null ? <span className="text-fg-3">Not recorded</span> : <span className="num">{fmtNum(e.confidence, 2)}</span>,
            hint: "Extraction confidence",
          },
          {
            label: "Provenance",
            value:
              verified == null ? (
                <span className="text-fg-3">Not recorded</span>
              ) : verified ? (
                <Badge tone="pos">Verified issuer</Badge>
              ) : (
                <span className="inline-flex flex-wrap items-center gap-1.5">
                  <Badge tone="warn">{attribution ? humanize(attribution) : "Unverified"}</Badge>
                </span>
              ),
          },
          ...(sourceCompany ? [{ label: "Stated in filing of", value: sourceCompany, mono: true }] : []),
          { label: "Filed", value: fmtDate(e.publication_date), mono: true },
          { label: "Available from", value: fmtDateTime(e.available_time), mono: true },
          ...(e.event_time ? [{ label: "Event time", value: fmtDateTime(e.event_time), mono: true }] : []),
          ...(props.fiscal_period ? [{ label: "Fiscal period", value: fmtDate(String(props.fiscal_period)), mono: true }] : []),
          ...(e.source_reference ? [{ label: "Accession no.", value: e.source_reference, mono: true }] : []),
          ...(e.document_id ? [{ label: "Document", value: e.document_id, mono: true }] : []),
          ...(e.chunk_id ? [{ label: "Chunk", value: e.chunk_id, mono: true }] : []),
          { label: "Edge id", value: e.id, mono: true },
        ]}
      />
      {verified === false && attribution === "unsupported_cross_company" && (
        <p className="text-xs text-warn mt-1.5">
          This relationship was extracted from another company’s filing{sourceCompany ? ` (${sourceCompany})` : ""}, so it is
          not attributed to the issuer itself.
        </p>
      )}
      <div className="flex flex-wrap gap-1 mt-2">
        {e.document_id && (
          <Button size="sm" variant="ghost" icon={<BookOpen className="w-3.5 h-3.5" />} onClick={() => onOpenFiling(e.document_id!, e.chunk_id)}>
            Read source chunk
          </Button>
        )}
        {e.source_url && (
          <a
            href={e.source_url}
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center gap-1.5 h-7 px-2.5 text-xs rounded text-accent hover:bg-accent/10"
          >
            SEC filing <ExternalLink className="w-3.5 h-3.5" />
          </a>
        )}
      </div>
    </li>
  );
}

function Inspector({
  selection,
  focalId,
  nodeIndex,
  neighboursOf,
  expanded,
  expanding,
  onExpand,
  onCollapse,
  onPivot,
  onSetContext,
  contextId,
  onOpenFiling,
  onClose,
}: {
  selection: Selection;
  focalId: string;
  nodeIndex: Map<string, GraphNodeItem>;
  neighboursOf: (parent: string) => Neighbour[];
  expanded: Set<string>;
  expanding: string | null;
  onExpand: (id: string) => void;
  onCollapse: (id: string) => void;
  onPivot: (id: string) => void;
  onSetContext: (n: GraphNodeItem) => void;
  contextId: string | null;
  onOpenFiling: (doc: string, chunk?: string | null) => void;
  onClose?: () => void;
}) {
  const label = (id: string) => nodeIndex.get(id)?.label ?? id;
  const sel = selection ?? { kind: "node" as const, id: focalId };

  if (sel.kind === "edge") {
    const n = neighboursOf(sel.parent).find((x) => x.id === sel.neighbourId);
    if (!n) return <EmptyState title="Relationship no longer loaded" />;
    const sorted = [...n.edges].sort((a, b) => (b.available_time ?? "").localeCompare(a.available_time ?? ""));
    return (
      <div className="p-4">
        <div className="text-xs text-fg-3">Relationship</div>
        <h3 className="text-md font-semibold mt-0.5 leading-snug">
          {label(sel.parent)} <span className="text-fg-3 font-normal">and</span> {label(n.id)}
        </h3>
        <p className="text-xs text-fg-2 mt-1">
          {n.edges.length} extracted edge{n.edges.length > 1 ? "s" : ""} between these entities in the loaded data
          {n.outgoing > 0 && n.incoming > 0
            ? `, in both directions (${n.outgoing} outgoing, ${n.incoming} incoming).`
            : n.outgoing > 0
            ? `, directed from ${label(sel.parent)}.`
            : `, directed towards ${label(sel.parent)}.`}
        </p>
        <ul className="mt-2">
          {sorted.map((e) => (
            <EdgeRecord key={e.id} e={e} labels={label} onOpenFiling={onOpenFiling} />
          ))}
        </ul>
      </div>
    );
  }

  const node = nodeIndex.get(sel.id);
  if (!node) return <EmptyState title="Entity not loaded" />;
  const props = node.properties || {};
  const isFocal = node.id === focalId;
  const aliases = (props.aliases as string[] | undefined) ?? [];
  const docIds = (props.document_ids as string[] | undefined) ?? [];
  const chunkIds = (props.chunk_ids as string[] | undefined) ?? [];
  const otherProps = Object.entries(props).filter(
    ([k]) => !["canonical_name", "name", "aliases", "document_ids", "chunk_ids"].includes(k)
  );
  const loadedNeighbours = isFocal ? neighboursOf(focalId) : neighboursOf(node.id);
  const relCounts = new Map<string, number>();
  (isFocal ? loadedNeighbours.flatMap((n) => n.edges) : neighboursOf(focalId).find((n) => n.id === node.id)?.edges ?? []).forEach((e) =>
    relCounts.set(e.relationship, (relCounts.get(e.relationship) ?? 0) + 1)
  );
  const ticker = nodeTicker(node.id);
  const isContext = contextId === node.id;

  return (
    <div className="p-4">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="text-xs text-fg-3">{isFocal ? "Focal entity" : "Entity"}</div>
          <h3 className="text-lg font-semibold leading-snug mt-0.5 break-words">{node.label}</h3>
        </div>
        {onClose && (
          <IconButton label="Close inspector" size="sm" onClick={onClose}>
            <X className="w-4 h-4" />
          </IconButton>
        )}
      </div>
      <div className="flex flex-wrap gap-1.5 mt-3">
        {!isFocal && (
          <Button size="sm" variant="primary" icon={<Focus className="w-3.5 h-3.5" />} onClick={() => onPivot(node.id)}>
            Make focal
          </Button>
        )}
        {!isFocal &&
          (expanded.has(node.id) ? (
            <Button size="sm" onClick={() => onCollapse(node.id)}>
              Collapse
            </Button>
          ) : (
            <Button size="sm" loading={expanding === node.id} icon={<Share2 className="w-3.5 h-3.5" />} onClick={() => onExpand(node.id)}>
              Expand neighbours
            </Button>
          ))}
        <Button
          size="sm"
          variant={isContext ? "secondary" : "ghost"}
          icon={<Crosshair className={cx("w-3.5 h-3.5", isContext && "text-accent")} />}
          onClick={() => onSetContext(node)}
          disabled={isContext}
        >
          {isContext ? "Research context" : "Set as context"}
        </Button>
      </div>
      <KeyValue
        className="mt-4"
        dense
        items={[
          { label: "Id", value: node.id, mono: true },
          { label: "Type", value: node.type },
          ...(ticker ? [{ label: "Ticker", value: ticker, mono: true }] : []),
          ...(props.name && props.name !== node.label ? [{ label: "Name", value: String(props.name) }] : []),
          { label: "Aliases", value: aliases.length ? aliases.join(", ") : <span className="text-fg-3">None recorded</span> },
          { label: "Degree", value: <span className="num">{fmtInt(node.degree)}</span>, hint: "Edges in the whole graph" },
          {
            label: "Mentioned in",
            value: (
              <span className="num">
                {docIds.length} document{docIds.length === 1 ? "" : "s"}, {chunkIds.length} chunk{chunkIds.length === 1 ? "" : "s"}
              </span>
            ),
          },
          ...otherProps.slice(0, 8).map(([k, v]) => ({
            label: humanize(k),
            value: typeof v === "object" ? JSON.stringify(v) : String(v ?? "—"),
            mono: typeof v !== "string" || /time|date|_id$/.test(k),
          })),
        ]}
      />
      {props.available_time == null && (
        <p className="text-2xs text-fg-3 mt-1.5">The node record carries no temporal fields; see edges for filing dates.</p>
      )}
      {relCounts.size > 0 && (
        <div className="mt-5">
          <div className="text-xs font-medium text-fg-2 mb-1.5">
            {isFocal ? "Loaded relationships by type" : `Relationships with the focal entity`}
          </div>
          <ul className="space-y-1">
            {Array.from(relCounts.entries())
              .sort((a, b) => b[1] - a[1])
              .map(([rel, c]) => (
                <li key={rel} className="flex items-center gap-2 text-sm">
                  <span className="w-2 h-2 rounded-full" style={{ background: relColor(relFamily(rel)) }} />
                  <span className="flex-1">{relLabel(rel)}</span>
                  <span className="num text-fg-3 text-xs">{c}</span>
                </li>
              ))}
          </ul>
        </div>
      )}
      {docIds.length > 0 && (
        <div className="mt-5">
          <div className="text-xs font-medium text-fg-2 mb-1.5">Source documents</div>
          <ul className="space-y-0.5">
            {docIds.slice(0, 6).map((d) => (
              <li key={d}>
                <button className="font-mono-num text-xs text-accent hover:underline" onClick={() => onOpenFiling(d)}>
                  {d}
                </button>
              </li>
            ))}
            {docIds.length > 6 && <li className="text-xs text-fg-3">and {docIds.length - 6} more</li>}
          </ul>
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Search                                                              */
/* ------------------------------------------------------------------ */

function EntitySearch({ onPick }: { onPick: (n: GraphNodeItem) => void }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<GraphNodeItem[] | null>(null);
  const [error, setError] = useState<ApiError | null>(null);
  const [loading, setLoading] = useState(false);
  const [open, setOpen] = useState(false);
  const [hi, setHi] = useState(0);
  const boxRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const term = q.trim();
    if (term.length < 2) {
      setResults(null);
      setError(null);
      return;
    }
    const ctrl = new AbortController();
    const t = window.setTimeout(async () => {
      setLoading(true);
      try {
        const r = await api.searchNodes(term, 12, ctrl.signal);
        setResults(r);
        setError(null);
        setHi(0);
      } catch (e) {
        if (!isAbortError(e)) setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
      } finally {
        if (!ctrl.signal.aborted) setLoading(false);
      }
    }, 220);
    return () => {
      ctrl.abort();
      window.clearTimeout(t);
    };
  }, [q]);

  useEffect(() => {
    const onDown = (e: MouseEvent) => {
      if (boxRef.current && !boxRef.current.contains(e.target as globalThis.Node)) setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    return () => document.removeEventListener("mousedown", onDown);
  }, []);

  const pick = (n: GraphNodeItem) => {
    onPick(n);
    setOpen(false);
    setQ("");
  };

  return (
    <div ref={boxRef} className="relative w-full sm:w-72">
      <Input
        inputSize="sm"
        role="combobox"
        aria-expanded={open && !!results}
        aria-controls="entity-results"
        aria-label="Search entities"
        placeholder="Search entities, e.g. Apple"
        leading={loading ? <Spinner className="w-3.5 h-3.5" /> : <Search className="w-3.5 h-3.5" />}
        value={q}
        onChange={(e) => {
          setQ(e.target.value);
          setOpen(true);
        }}
        onFocus={() => setOpen(true)}
        onKeyDown={(e) => {
          if (!results?.length) return;
          if (e.key === "ArrowDown") {
            e.preventDefault();
            setHi((h) => Math.min(h + 1, results.length - 1));
          } else if (e.key === "ArrowUp") {
            e.preventDefault();
            setHi((h) => Math.max(h - 1, 0));
          } else if (e.key === "Enter") {
            e.preventDefault();
            pick(results[hi]);
          } else if (e.key === "Escape") setOpen(false);
        }}
      />
      {open && q.trim().length >= 2 && (results || error) && (
        <div
          id="entity-results"
          role="listbox"
          className="absolute z-30 top-full mt-1 left-0 right-0 bg-surface border border-line rounded-md shadow-overlay max-h-80 overflow-y-auto scrollbar-thin animate-rise-in"
        >
          {error ? (
            <div className="p-3 text-sm text-neg">{error.message}</div>
          ) : results && results.length === 0 ? (
            <div className="p-3 text-sm text-fg-2">
              No entity matches “{q.trim()}”. Try a company name or ticker.
            </div>
          ) : (
            results?.map((n, i) => (
              <button
                key={n.id}
                role="option"
                aria-selected={i === hi}
                onMouseEnter={() => setHi(i)}
                onClick={() => pick(n)}
                className={cx("w-full text-left px-3 py-2 flex items-center gap-3", i === hi && "bg-surface-2")}
              >
                <span className="min-w-0 flex-1">
                  <span className="block text-sm text-fg truncate">{n.label}</span>
                  <span className="block font-mono-num text-2xs text-fg-3 truncate">{n.id}</span>
                </span>
                <span className="text-2xs text-fg-3 text-right shrink-0">
                  {n.type}
                  <span className="block num">{fmtInt(n.degree)} edges</span>
                </span>
              </button>
            ))
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* View                                                                */
/* ------------------------------------------------------------------ */

interface FocalData {
  center: string;
  pages: SubgraphResponse[];
}

export function GraphExplorerView({ active, onNavigate }: { active: boolean; onNavigate: (t: NavTab) => void }) {
  const { entity, setEntity } = useResearchContext();
  const health = useApi((s) => api.getHealth(s), []);
  const tokens = useTokens();
  const followContext = useRef(true);
  const [focalId, setFocalId] = useState<string>(entity?.nodeId ?? "company:NVDA");
  const [scope, setScope] = useState<Scope>("companies");
  const [focal, setFocal] = useState<FocalData | null>(null);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<ApiError | null>(null);
  const [shown, setShown] = useState(NEIGHBOUR_STEP);
  const [hiddenFamilies, setHiddenFamilies] = useState<Set<RelFamily>>(new Set());
  const [expansions, setExpansions] = useState<Map<string, SubgraphResponse>>(new Map());
  const [expanding, setExpanding] = useState<string | null>(null);
  const [selection, setSelection] = useState<Selection>(null);
  const [inspectorOpen, setInspectorOpen] = useState(true);
  const [mobileInspector, setMobileInspector] = useState(false);
  const [reloadNonce, setReloadNonce] = useState(0);
  const [legendOpen, setLegendOpen] = useState(true);
  useEffect(() => {
    if (window.innerWidth < 1024) setLegendOpen(false);
  }, []);
  const rf = useRef<ReactFlowInstance | null>(null);
  const ctrlRef = useRef<AbortController | null>(null);

  // Follow the research context when it changes elsewhere.
  useEffect(() => {
    if (!followContext.current) {
      followContext.current = true; // context was set from inside the graph: keep the current view
      return;
    }
    if (entity?.nodeId) setFocalId(entity.nodeId);
  }, [entity?.nodeId]);

  // Load first page for focal + scope.
  useEffect(() => {
    ctrlRef.current?.abort();
    const ctrl = new AbortController();
    ctrlRef.current = ctrl;
    setLoading(true);
    setError(null);
    setSelection(null);
    setExpansions(new Map());
    setShown(NEIGHBOUR_STEP);
    setHiddenFamilies(new Set());
    api
      .getSubgraph(focalId, { limit: PAGE_SIZE, offset: 0, filterType: scope, signal: ctrl.signal })
      .then((d) => setFocal({ center: d.center_node, pages: [d] }))
      .catch((e) => {
        if (isAbortError(e)) return;
        setFocal(null);
        setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
      })
      .finally(() => !ctrl.signal.aborted && setLoading(false));
    return () => ctrl.abort();
  }, [focalId, scope, reloadNonce]);

  const first = focal?.pages[0] ?? null;
  const center = focal?.center ?? focalId;
  const allFocalEdges = useMemo(() => focal?.pages.flatMap((p) => p.edges) ?? [], [focal]);
  const last = focal?.pages[focal.pages.length - 1] ?? null;
  const hasMore = !!last?.has_more;

  const nodeIndex = useMemo(() => {
    const m = new Map<string, GraphNodeItem>();
    focal?.pages.forEach((p) => p.nodes.forEach((n) => m.set(n.id, n)));
    expansions.forEach((p) => p.nodes.forEach((n) => !m.has(n.id) && m.set(n.id, n)));
    return m;
  }, [focal, expansions]);

  const allNeighbours = useMemo(() => aggregate(center, allFocalEdges, nodeIndex), [center, allFocalEdges, nodeIndex]);

  const familyCounts = useMemo(() => {
    const c = new Map<RelFamily, number>();
    allFocalEdges.forEach((e) => c.set(relFamily(e.relationship), (c.get(relFamily(e.relationship)) ?? 0) + 1));
    return c;
  }, [allFocalEdges]);

  // Neighbour filtered by relationship family: keep if any of its edges is in a visible family.
  const filteredNeighbours = useMemo(
    () =>
      hiddenFamilies.size === 0
        ? allNeighbours
        : allNeighbours.filter((n) => n.edges.some((e) => !hiddenFamilies.has(relFamily(e.relationship)))),
    [allNeighbours, hiddenFamilies]
  );
  const visibleNeighbours = filteredNeighbours.slice(0, shown);
  const visibleEdgeCount = visibleNeighbours.reduce(
    (s, n) => s + n.edges.filter((e) => !hiddenFamilies.has(relFamily(e.relationship))).length,
    0
  );

  const neighboursOf = useCallback(
    (parent: string): Neighbour[] => {
      if (parent === center) return allNeighbours;
      const ex = expansions.get(parent);
      if (!ex) return [];
      return aggregate(parent, ex.edges, nodeIndex);
    },
    [center, allNeighbours, expansions, nodeIndex]
  );

  /* ---------- graph elements ---------- */

  const { nodes, edges } = useMemo(() => {
    const fnodes: Node<EntityNodeData>[] = [];
    const fedges: Edge[] = [];
    if (!first) return { nodes: fnodes, edges: fedges };
    const centerNode = nodeIndex.get(center);
    const selNode = selection?.kind === "node" ? selection.id : null;
    const selEdge = selection?.kind === "edge" ? `${selection.parent}::${selection.neighbourId}` : null;

    fnodes.push({
      id: center,
      type: "entity",
      position: { x: -90, y: -22 },
      data: {
        label: centerNode?.label ?? center,
        sub: nodeTicker(center) ?? centerNode?.type ?? null,
        family: "other",
        role: "focal",
        selected: selNode === center || selection == null,
        dimmed: false,
      },
    });

    const placed = new Map<string, { x: number; y: number; angle: number }>();
    const pos = ringLayout(visibleNeighbours.length);
    visibleNeighbours.forEach((n, i) => {
      const p = pos[i];
      placed.set(n.id, p);
      const label = n.node?.label ?? n.id;
      fnodes.push({
        id: n.id,
        type: "entity",
        position: { x: p.x - 8, y: p.y - 8 },
        data: {
          label,
          family: n.family,
          role: "neighbour",
          selected: selNode === n.id,
          dimmed: false,
          expanded: expansions.has(n.id),
        },
      });
    });

    const edgeFor = (parent: string, n: Neighbour, small = false) => {
      const visibleEdges = n.edges.filter((e) => !hiddenFamilies.has(relFamily(e.relationship)));
      if (visibleEdges.length === 0) return;
      const id = `${parent}::${n.id}`;
      const color = tokens[`rel-${n.family}` as const] ?? relColor(n.family);
      const parentExtent = parent === center ? FOCAL_EXTENT : DOT_EXTENT;
      const childExtent = small ? DOT_EXTENT_SM : DOT_EXTENT;
      const selected = selEdge === id;
      const out = n.outgoing > 0;
      const inc = n.incoming > 0;
      fedges.push({
        id,
        source: out || !inc ? parent : n.id,
        target: out || !inc ? n.id : parent,
        type: "trimmed",
        data: {
          parent,
          neighbourId: n.id,
          sourceExtent: out || !inc ? parentExtent : childExtent,
          targetExtent: out || !inc ? childExtent : parentExtent,
        },
        labelStyle: { fill: tokens["fg-2"], fontSize: 10 },
        labelBgStyle: { fill: tokens.surface },
        label: selected ? `${visibleEdges.length} × ${relLabel(visibleEdges[0].relationship)}${visibleEdges.length > 1 ? "…" : ""}` : undefined,
        labelBgPadding: [4, 2],
        labelBgBorderRadius: 3,
        style: {
          stroke: color,
          strokeOpacity: selected ? 1 : small ? 0.45 : 0.6,
          strokeWidth: (selected ? 1 : 0) + Math.min(1 + Math.log2(visibleEdges.length), small ? 2.5 : 3.5),
        },
        markerEnd: { type: MarkerType.ArrowClosed, color, width: 14, height: 14 },
        markerStart: out && inc ? { type: MarkerType.ArrowClosed, color, width: 14, height: 14 } : undefined,
        interactionWidth: 14,
      });
    };

    visibleNeighbours.forEach((n) => edgeFor(center, n));

    // Expansions: secondary neighbours fanned outward from the expanded node.
    expansions.forEach((sub, parentId) => {
      const anchor = placed.get(parentId);
      if (!anchor) return;
      const kids = aggregate(parentId, sub.edges, nodeIndex)
        .filter((k) => k.id !== center && !placed.has(k.id))
        .slice(0, EXPANSION_SHOWN);
      const spread = Math.PI / 2.2;
      kids.forEach((k, i) => {
        const a = anchor.angle + (kids.length === 1 ? 0 : -spread / 2 + (spread * i) / (kids.length - 1));
        const r = 150 + (i % 2) * 40;
        const x = anchor.x + Math.cos(a) * r;
        const y = anchor.y + Math.sin(a) * r;
        if (!fnodes.some((fn) => fn.id === k.id)) {
          fnodes.push({
            id: k.id,
            type: "entity",
            position: { x: x - 6, y: y - 6 },
            data: { label: k.node?.label ?? k.id, family: k.family, role: "expansion", selected: selNode === k.id, dimmed: false },
          });
          placed.set(k.id, { x, y, angle: a });
        }
        edgeFor(parentId, k, true);
      });
    });

    // Dim everything not adjacent to the current selection.
    if (selNode && selNode !== center) {
      const adj = new Set([selNode, center]);
      fedges.forEach((e) => {
        if (e.source === selNode || e.target === selNode) {
          adj.add(e.source);
          adj.add(e.target);
        }
      });
      fnodes.forEach((n) => (n.data = { ...n.data, dimmed: !adj.has(n.id) }));
      fedges.forEach((e) => {
        if (!(e.source === selNode || e.target === selNode)) e.style = { ...e.style, strokeOpacity: 0.12 };
      });
    }
    return { nodes: fnodes, edges: fedges };
  }, [first, center, nodeIndex, visibleNeighbours, expansions, hiddenFamilies, selection, tokens]);

  /* ---------- viewport ---------- */

  const fit = useCallback(() => rf.current?.fitView({ padding: 0.18, duration: 300 }), []);
  // Re-fit once nodes have rendered and whenever the canvas size or content changes.
  const nodeCount = nodes.length;
  useEffect(() => {
    if (!active || loading || nodeCount === 0) return;
    const t1 = window.setTimeout(fit, 80);
    const t2 = window.setTimeout(fit, 320);
    return () => {
      window.clearTimeout(t1);
      window.clearTimeout(t2);
    };
  }, [active, loading, nodeCount, inspectorOpen, fit]);

  /* ---------- actions ---------- */

  const loadMore = async () => {
    if (!focal || !hasMore || loadingMore) return;
    setLoadingMore(true);
    try {
      const offset = focal.pages.reduce((s, p) => s + p.edges.length, 0);
      const d = await api.getSubgraph(focal.center, { limit: PAGE_SIZE, offset, filterType: scope });
      setFocal((f) => (f && f.center === d.center_node ? { ...f, pages: [...f.pages, d] } : f));
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
    } finally {
      setLoadingMore(false);
    }
  };

  const expand = async (id: string) => {
    if (expansions.has(id) || expanding) return;
    setExpanding(id);
    try {
      const d = await api.getSubgraph(id, { limit: EXPANSION_LIMIT, offset: 0, filterType: scope });
      setExpansions((m) => new Map(m).set(id, d));
    } catch (e) {
      setError(e instanceof ApiError ? e : new ApiError(String(e), 0));
    } finally {
      setExpanding(null);
    }
  };

  const collapse = (id: string) =>
    setExpansions((m) => {
      const n = new Map(m);
      n.delete(id);
      return n;
    });

  const pivot = (id: string) => {
    setFocalId(id);
    setMobileInspector(false);
  };

  const toContext = (n: GraphNodeItem): ResearchEntity => ({
    nodeId: n.id,
    name: n.label,
    ticker: CORPUS_ISSUERS.some((c) => c.ticker === nodeTicker(n.id)) ? nodeTicker(n.id) : null,
    type: n.type,
  });

  const openFiling = (documentId: string, chunkId?: string | null) => {
    try {
      sessionStorage.setItem("fingraph-open-filing", JSON.stringify({ documentId, chunkId: chunkId ?? null }));
    } catch {
      /* ignore */
    }
    window.dispatchEvent(new CustomEvent("fingraph:open-filing", { detail: { documentId, chunkId: chunkId ?? null } }));
    onNavigate("evidence");
  };

  const toggleFamily = (f: RelFamily) =>
    setHiddenFamilies((s) => {
      const n = new Set(s);
      if (n.has(f)) n.delete(f);
      else n.add(f);
      return n;
    });

  const onNodeClick = (_: React.MouseEvent, n: Node) => {
    setSelection({ kind: "node", id: n.id });
    setMobileInspector(true);
  };
  const onEdgeClick = (_: React.MouseEvent, e: Edge) => {
    const d = e.data as { parent: string; neighbourId: string };
    setSelection({ kind: "edge", parent: d.parent, neighbourId: d.neighbourId });
    setMobileInspector(true);
  };

  /* ---------- render ---------- */

  const kgNodes = health.data?.knowledge_graph.details?.node_count as number | undefined;
  const kgEdges = health.data?.knowledge_graph.details?.edge_count as number | undefined;
  const loadedEdges = allFocalEdges.length;
  const matching = first?.total_matching ?? null;

  const inspector = (
    <Inspector
      selection={selection}
      focalId={center}
      nodeIndex={nodeIndex}
      neighboursOf={neighboursOf}
      expanded={new Set(expansions.keys())}
      expanding={expanding}
      onExpand={expand}
      onCollapse={collapse}
      onPivot={pivot}
      onSetContext={(n) => {
        followContext.current = false;
        setEntity(toContext(n));
      }}
      contextId={entity?.nodeId ?? null}
      onOpenFiling={openFiling}
    />
  );

  return (
    <div className="h-full flex flex-col">
      {/* Toolbar */}
      <div className="shrink-0 px-4 sm:px-5 pt-3 pb-2.5 border-b border-line-soft space-y-2.5">
        <div className="flex flex-wrap items-center gap-2">
          <EntitySearch
            onPick={(n) => {
              setFocalId(n.id);
            }}
          />
          <div className="flex flex-wrap gap-1" role="group" aria-label="Corpus issuers">
            {CORPUS_ISSUERS.map((c) => {
              const on = center === `company:${c.ticker}`;
              return (
                <button
                  key={c.ticker}
                  onClick={() => setFocalId(`company:${c.ticker}`)}
                  aria-pressed={on}
                  title={c.name}
                  className={cx(
                    "h-7 px-2 rounded font-mono-num text-xs border transition-colors",
                    on ? "border-accent/50 bg-accent/10 text-fg" : "border-transparent text-fg-2 hover:text-fg hover:bg-surface-2"
                  )}
                >
                  {c.ticker}
                </button>
              );
            })}
          </div>
          <div className="ml-auto flex items-center gap-2">
            <Segmented
              label="Relationship scope"
              value={scope}
              onChange={setScope}
              options={[
                { value: "companies", label: "Companies" },
                { value: "all", label: "All entities" },
              ]}
            />
            <IconButton
              label={inspectorOpen ? "Hide inspector" : "Show inspector"}
              className="hidden lg:inline-flex"
              onClick={() => setInspectorOpen((o) => !o)}
            >
              {inspectorOpen ? <PanelRightClose className="w-4 h-4" /> : <PanelRightOpen className="w-4 h-4" />}
            </IconButton>
          </div>
        </div>

        {/* Count ledger */}
        <div className="flex flex-wrap items-center gap-x-5 gap-y-1 text-xs text-fg-3" aria-live="polite">
          <span title="Whole knowledge graph">
            Graph <span className="num text-fg-2">{fmtInt(kgNodes)}</span> nodes, <span className="num text-fg-2">{fmtInt(kgEdges)}</span> edges
          </span>
          {first && (
            <>
              <span title="Edges touching the focal entity, any relationship type">
                Direct edges <span className="num text-fg-2">{fmtInt(first.total_direct_edges)}</span>
              </span>
              {scope === "companies" && (
                <span title="Distinct company, supplier, competitor or regulator nodes connected directly">
                  Connected companies <span className="num text-fg-2">{fmtInt(first.total_connected_companies)}</span>
                </span>
              )}
              <span title={scope === "companies" ? "Edges to corporate entities" : "All direct edges"}>
                In scope <span className="num text-fg-2">{fmtInt(matching)}</span>
              </span>
              <span>
                Loaded <span className="num text-fg-2">{fmtInt(loadedEdges)}</span> edges to{" "}
                <span className="num text-fg-2">{fmtInt(allNeighbours.length)}</span> entities
              </span>
              <span className="text-fg-2">
                Showing <span className="num">{visibleNeighbours.length}</span> of <span className="num">{filteredNeighbours.length}</span>{" "}
                entities, <span className="num">{fmtInt(visibleEdgeCount)}</span> edges
                {hiddenFamilies.size > 0 && " after filters"}
              </span>
            </>
          )}
        </div>
      </div>

      <ProgressBar active={loading || loadingMore || !!expanding} />

      <div className="flex-1 min-h-0 flex">
        {/* Canvas */}
        <div className="flex-1 min-w-0 relative bg-canvas">
          {error && !first ? (
            <div className="absolute inset-0 flex items-center justify-center p-6">
              {error.isNotFound ? (
                <EmptyState
                  title={`“${focalId}” is not in the knowledge graph`}
                  action={
                    <Button size="sm" onClick={() => setFocalId("company:NVDA")}>
                      Open NVIDIA instead
                    </Button>
                  }
                >
                  Search for the entity by name, or pick one of the corpus issuers above.
                </EmptyState>
              ) : (
                <ErrorState
                  className="max-w-md"
                  title="Could not load the graph"
                  error={error}
                  action={
                    <Button size="sm" onClick={() => setReloadNonce((n) => n + 1)}>
                      Retry
                    </Button>
                  }
                />
              )}
            </div>
          ) : !first && loading ? (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-fg-3 gap-2">
              <Spinner /> Loading neighbourhood…
            </div>
          ) : first && allNeighbours.length === 0 ? (
            <div className="absolute inset-0 flex items-center justify-center">
              <EmptyState
                title="No direct relationships in this scope"
                action={
                  scope === "companies" ? (
                    <Button size="sm" onClick={() => setScope("all")}>
                      Show all entity types
                    </Button>
                  ) : undefined
                }
              >
                {nodeIndex.get(center)?.label ?? center} has {fmtInt(first.total_direct_edges)} direct edges in total.
              </EmptyState>
            </div>
          ) : (
            <div className="absolute inset-0">
            <ReactFlow
              nodes={nodes}
              edges={edges}
              nodeTypes={NODE_TYPES}
              edgeTypes={EDGE_TYPES}
              onInit={(i) => {
                rf.current = i;
                window.setTimeout(fit, 120);
              }}
              onNodeClick={onNodeClick}
              onEdgeClick={onEdgeClick}
              onPaneClick={() => setSelection(null)}
              nodesDraggable={false}
              nodesConnectable={false}
              elementsSelectable
              minZoom={0.2}
              maxZoom={2.5}
              proOptions={{ hideAttribution: true }}
              className={cx("transition-opacity duration-300", loading ? "opacity-40" : "opacity-100")}
            >
              <Background variant={BackgroundVariant.Dots} gap={22} size={1.2} color={tokens["graph-dot"]} />
            </ReactFlow>
            </div>
          )}

          {/* Canvas controls */}
          {first && (
            <div className="absolute left-3 bottom-3 flex flex-col bg-surface border border-line rounded-md shadow-overlay overflow-hidden">
              <IconButton label="Zoom in" size="sm" className="rounded-none" onClick={() => rf.current?.zoomIn({ duration: 150 })}>
                <Plus className="w-3.5 h-3.5" />
              </IconButton>
              <IconButton label="Zoom out" size="sm" className="rounded-none" onClick={() => rf.current?.zoomOut({ duration: 150 })}>
                <Minus className="w-3.5 h-3.5" />
              </IconButton>
              <IconButton label="Fit to view" size="sm" className="rounded-none" onClick={fit}>
                <Maximize2 className="w-3.5 h-3.5" />
              </IconButton>
              <IconButton
                label="Reset view"
                size="sm"
                className="rounded-none"
                onClick={() => {
                  setSelection(null);
                  setExpansions(new Map());
                  setHiddenFamilies(new Set());
                  setShown(NEIGHBOUR_STEP);
                  window.setTimeout(fit, 30);
                }}
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </IconButton>
            </div>
          )}

          {/* Legend / filters + progressive reveal */}
          {first && allNeighbours.length > 0 && !legendOpen && (
            <button
              onClick={() => setLegendOpen(true)}
              className="absolute right-3 top-3 h-8 px-3 rounded-md border border-line bg-surface/95 backdrop-blur-sm shadow-overlay text-xs text-fg-2 hover:text-fg inline-flex items-center gap-2"
              aria-expanded={false}
            >
              <span className="flex -space-x-0.5">
                {FAMILIES.filter((f) => familyCounts.has(f))
                  .slice(0, 4)
                  .map((f) => (
                    <span key={f} className="w-2 h-2 rounded-full ring-1 ring-surface" style={{ background: relColor(f) }} />
                  ))}
              </span>
              Filters and more
              {hiddenFamilies.size > 0 && <Badge tone="accent">{hiddenFamilies.size} hidden</Badge>}
            </button>
          )}
          {first && allNeighbours.length > 0 && legendOpen && (
            <div className="absolute right-3 top-3 w-[220px] max-w-[calc(100%-1.5rem)] bg-surface/95 backdrop-blur-sm border border-line rounded-lg shadow-overlay p-3 animate-fade-in">
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-xs font-medium text-fg-2">Relationship types</span>
                <IconButton label="Hide panel" size="sm" className="-mr-1.5 -mt-1" onClick={() => setLegendOpen(false)}>
                  <X className="w-3.5 h-3.5" />
                </IconButton>
              </div>
              <ul className="space-y-0.5">
                {FAMILIES.filter((f) => familyCounts.has(f)).map((f) => {
                  const off = hiddenFamilies.has(f);
                  return (
                    <li key={f}>
                      <button
                        onClick={() => toggleFamily(f)}
                        aria-pressed={!off}
                        className={cx("w-full flex items-center gap-2 text-xs py-1 px-1 rounded hover:bg-surface-2", off && "opacity-45")}
                      >
                        <span
                          className="w-2.5 h-2.5 rounded-full border-2"
                          style={{ borderColor: relColor(f), background: off ? "transparent" : relColor(f, 0.5) }}
                        />
                        <span className="flex-1 text-left text-fg">{REL_FAMILY_LABEL[f]}</span>
                        <span className="num text-fg-3">{familyCounts.get(f)}</span>
                      </button>
                    </li>
                  );
                })}
              </ul>
              <div className="border-t border-line-soft mt-2 pt-2 space-y-1.5">
                {filteredNeighbours.length > visibleNeighbours.length && (
                  <Button size="sm" className="w-full" onClick={() => setShown((s) => s + NEIGHBOUR_STEP)}>
                    Show {Math.min(NEIGHBOUR_STEP, filteredNeighbours.length - visibleNeighbours.length)} more entities
                  </Button>
                )}
                {visibleNeighbours.length > NEIGHBOUR_STEP && (
                  <Button size="sm" variant="ghost" className="w-full" onClick={() => setShown(NEIGHBOUR_STEP)}>
                    Show strongest {NEIGHBOUR_STEP} only
                  </Button>
                )}
                {hasMore && (
                  <Button size="sm" variant="ghost" className="w-full" loading={loadingMore} onClick={loadMore}>
                    Load next {Math.min(PAGE_SIZE, (matching ?? 0) - loadedEdges)} edges
                  </Button>
                )}
                {!hasMore && <p className="text-2xs text-fg-3">All {fmtInt(matching)} in-scope edges are loaded.</p>}
              </div>
              <p className="text-2xs text-fg-3 mt-2 leading-snug">
                One line per entity pair; thickness grows with the number of extracted edges. Click a line for each edge’s source.
              </p>
            </div>
          )}
          {error && first && (
            <div className="absolute left-14 bottom-3 right-3 max-w-md">
              <ErrorState compact title="Request failed" error={error} />
            </div>
          )}
        </div>

        {/* Inspector (desktop) */}
        {inspectorOpen && first && (
          <aside className="hidden lg:block w-[340px] shrink-0 border-l border-line-soft bg-surface overflow-y-auto scrollbar-thin animate-fade-in">
            {inspector}
          </aside>
        )}
      </div>

      {/* Inspector (tablet / mobile) */}
      <div className="lg:hidden">
        <Drawer
          open={mobileInspector && !!first}
          onClose={() => setMobileInspector(false)}
          title={selection?.kind === "edge" ? "Relationship" : "Entity"}
        >
          {inspector}
        </Drawer>
      </div>
    </div>
  );
}
