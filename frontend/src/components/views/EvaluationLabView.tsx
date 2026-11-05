"use client";

import React, { useMemo, useState } from "react";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  LabelList,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { api } from "@/lib/api";
import { useApi } from "@/lib/useApi";
import { useTokens } from "@/lib/useTokens";
import { fmtDate, fmtNum, fmtPct, humanize } from "@/lib/format";
import { ConfigurationDetail, ParametricLeakageResponse, PerQuestionItem } from "@/lib/types";
import {
  Badge,
  ChartFrame,
  EmptyState,
  ErrorState,
  Note,
  Segmented,
  Select,
  Skeleton,
  cx,
} from "@/components/ui";

const BASELINE_ID = "cfg1_frozen_baseline";
const COMBINED_ID = "cfg5_combined";

const SHORT_NAMES: Record<string, string> = {
  cfg1_frozen_baseline: "Frozen baseline",
  cfg2_balanced_rrf_only: "Balanced RRF",
  cfg3_adaptive_hop_only: "Adaptive hops",
  cfg4_regulatory_linking_only: "Regulatory linking",
  cfg5_combined: "Combined (Phase 7)",
};
const shortName = (c: ConfigurationDetail) => SHORT_NAMES[c.config_id] ?? c.name;

interface MetricDef {
  key: keyof ConfigurationDetail;
  label: string;
  group: "Decision" | "Answer quality" | "Retrieval" | "Reasoning" | "Temporal" | "Latency";
  format: (v: number, c: ConfigurationDetail, n: { total: number; ans: number; unans: number }) => string;
  better: "higher" | "lower";
  definition: string;
}

const METRICS: MetricDef[] = [
  {
    key: "decision_accuracy",
    label: "Decision accuracy",
    group: "Decision",
    format: (v, _c, n) => `${fmtPct(v, 2)} (${Math.round(v * n.total)}/${n.total})`,
    better: "higher",
    definition: "Share of all questions where the answer-or-refuse decision was correct. Not a measure of factual correctness.",
  },
  {
    key: "correct_refusals",
    label: "Correct refusals",
    group: "Decision",
    format: (v, _c, n) => `${v} / ${n.unans}`,
    better: "higher",
    definition: "Unanswerable questions that were refused.",
  },
  {
    key: "false_refusals",
    label: "False refusals",
    group: "Decision",
    format: (v, _c, n) => `${v} / ${n.ans}`,
    better: "lower",
    definition: "Answerable questions that were refused.",
  },
  {
    key: "refusal_precision",
    label: "Refusal precision",
    group: "Decision",
    format: (v) => fmtPct(v, 1),
    better: "higher",
    definition: "Of all refusals, the share that were for genuinely unanswerable questions.",
  },
  {
    key: "hallucinations_total",
    label: "Hallucinations",
    group: "Decision",
    format: (v, _c, n) => `${v} / ${n.unans}`,
    better: "lower",
    definition: "Unanswerable questions that received an answer instead of a refusal.",
  },
  {
    key: "answerable_token_f1",
    label: "Answerable token F1",
    group: "Answer quality",
    format: (v) => fmtNum(v, 4),
    better: "higher",
    definition: "Mean token overlap (F1) between generated and reference answers, over the 50 answerable questions.",
  },
  {
    key: "answerable_exact_match",
    label: "Exact match",
    group: "Answer quality",
    format: (v) => fmtPct(v, 0),
    better: "higher",
    definition: "Answerable questions whose answer equals the reference after lower-casing and removing punctuation and articles.",
  },
  {
    key: "mean_hit_at_1",
    label: "Hit@1",
    group: "Retrieval",
    format: (v) => fmtPct(v, 0),
    better: "higher",
    definition: "A relevant (gold) chunk was ranked first. Chunk-level; unanswerable questions score 0.",
  },
  {
    key: "mean_hit_at_5",
    label: "Hit@5",
    group: "Retrieval",
    format: (v) => fmtPct(v, 0),
    better: "higher",
    definition: "A relevant chunk appears in the top five retrieved chunks.",
  },
  {
    key: "mean_hit_at_10",
    label: "Hit@10",
    group: "Retrieval",
    format: (v) => fmtPct(v, 0),
    better: "higher",
    definition: "A relevant chunk appears in the top ten retrieved chunks.",
  },
  {
    key: "mean_mrr",
    label: "MRR",
    group: "Retrieval",
    format: (v) => fmtNum(v, 4),
    better: "higher",
    definition: "Mean of 1 / rank of the first relevant chunk (0 if none retrieved).",
  },
  {
    key: "cross_doc_coverage",
    label: "Cross-document coverage",
    group: "Retrieval",
    format: (v) => fmtPct(v, 1),
    better: "higher",
    definition: "For answerable questions needing more than one document, the share of those documents that were retrieved.",
  },
  {
    key: "multi_hop_path_accuracy",
    label: "Multi-hop path accuracy",
    group: "Reasoning",
    format: (v) => fmtPct(v, 1),
    better: "higher",
    definition: "Answerable questions with an expected graph path where a retrieved path matched its relation types, order and direction.",
  },
  {
    key: "feir_context",
    label: "FEIR (context)",
    group: "Temporal",
    format: (v) => fmtNum(v, 3),
    better: "lower",
    definition: "Future evidence infiltration rate: context items available after the reference time, divided by all context items. Undefined for questions without a reference time.",
  },
  {
    key: "feir_cited",
    label: "FEIR (cited)",
    group: "Temporal",
    format: (v) => fmtNum(v, 3),
    better: "lower",
    definition: "The same rate, over cited evidence only.",
  },
  {
    key: "mean_latency_seconds",
    label: "Mean latency",
    group: "Latency",
    format: (v) => `${fmtNum(v, 1)} s`,
    better: "lower",
    definition: "Mean end-to-end time per question on the benchmark machine.",
  },
  {
    key: "median_latency_seconds",
    label: "Median latency",
    group: "Latency",
    format: (v) => `${fmtNum(v, 1)} s`,
    better: "lower",
    definition: "Median end-to-end time per question.",
  },
];

function Delta({ v, base, better, pct }: { v: number; base: number; better: "higher" | "lower"; pct?: boolean }) {
  const d = v - base;
  if (Math.abs(d) < 1e-9) return <span className="text-fg-3">0</span>;
  const good = better === "higher" ? d > 0 : d < 0;
  const text = pct ? `${d > 0 ? "+" : ""}${(d * 100).toFixed(1)} pp` : `${d > 0 ? "+" : ""}${Math.abs(d) < 1 ? d.toFixed(3) : d.toFixed(1)}`;
  return <span className={good ? "text-pos" : "text-neg"}>{text}</span>;
}

const PCT_KEYS = new Set<keyof ConfigurationDetail>([
  "decision_accuracy",
  "refusal_precision",
  "answerable_exact_match",
  "mean_hit_at_1",
  "mean_hit_at_5",
  "mean_hit_at_10",
  "cross_doc_coverage",
  "multi_hop_path_accuracy",
]);

export function EvaluationLabView({ active = true }: { active?: boolean }) {
  const summary = useApi((s) => api.getBenchmarkSummary(s), []);
  const perQ = useApi((s) => api.getPerQuestion(s), []);
  const plr = useApi((s) => api.getParametricLeakage(s), []);
  const t = useTokens();
  const [selected, setSelected] = useState(COMBINED_ID);
  const [category, setCategory] = useState("ALL");
  const [outcome, setOutcome] = useState<"all" | "wrong" | "changed">("all");

  const configs = useMemo(() => {
    const c = summary.data?.configurations;
    return c ? Object.values(c) : [];
  }, [summary.data]);
  const base = configs.find((c) => c.config_id === BASELINE_ID);
  const sel = configs.find((c) => c.config_id === selected) ?? configs[configs.length - 1];
  const n = {
    total: summary.data?.total_questions ?? 0,
    ans: summary.data?.answerable_questions ?? 0,
    unans: summary.data?.unanswerable_questions ?? 0,
  };

  const decisionData = configs.map((c) => ({
    name: shortName(c),
    id: c.config_id,
    accuracy: +(c.decision_accuracy * 100).toFixed(2),
    falseRefusals: c.false_refusals,
    f1: +c.answerable_token_f1.toFixed(4),
  }));

  const categoryData = useMemo(() => {
    const qs = perQ.data?.questions ?? [];
    const by = new Map<string, { total: number; base: number; comb: number }>();
    qs.forEach((q) => {
      const r = by.get(q.category) ?? { total: 0, base: 0, comb: 0 };
      r.total++;
      if (q.baseline_decision_correct) r.base++;
      if (q.combined_decision_correct) r.comb++;
      by.set(q.category, r);
    });
    return Array.from(by.entries()).map(([cat, r]) => ({
      name: humanize(cat),
      n: r.total,
      baseline: +((r.base / r.total) * 100).toFixed(1),
      combined: +((r.comb / r.total) * 100).toFixed(1),
      baseLabel: `${r.base}/${r.total}`,
      combLabel: `${r.comb}/${r.total}`,
    }));
  }, [perQ.data]);

  const questions = useMemo(() => {
    const qs = perQ.data?.questions ?? [];
    return qs.filter(
      (q) =>
        (category === "ALL" || q.category === category) &&
        (outcome === "all" ||
          (outcome === "wrong" && !q.combined_decision_correct) ||
          (outcome === "changed" && q.baseline_decision !== q.combined_decision))
    );
  }, [perQ.data, category, outcome]);

  const tooltipStyle = {
    background: t.surface,
    border: `1px solid ${t.line}`,
    borderRadius: 6,
    fontSize: 12,
    color: t.fg,
  };

  if (summary.error) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-10">
        <ErrorState title="Benchmark summary could not be loaded" error={summary.error} />
        <p className="text-sm text-fg-3 mt-3">
          The lab reads <code className="font-mono-num">results/phase7/phase7_summary.json</code> through the API. No values are shown
          without it.
        </p>
      </div>
    );
  }

  if (!summary.data || !sel) {
    return (
      <div className="max-w-[1180px] mx-auto px-8 py-8 space-y-4">
        <Skeleton className="h-8 w-1/2" />
        <Skeleton className="h-64 w-full" />
      </div>
    );
  }

  if (summary.data.error) {
    return (
      <div className="max-w-3xl mx-auto px-6 py-10">
        <EmptyState title="No Phase 7 summary on disk">{summary.data.error}</EmptyState>
      </div>
    );
  }

  return (
    <div className="max-w-[1180px] mx-auto px-4 sm:px-8 py-8 space-y-10">
      {/* Frame */}
      <section>
        <div className="flex flex-wrap items-center gap-2 mb-2">
          <Badge tone="neutral">Historical</Badge>
          <Badge mono>Phase 7</Badge>
          {summary.data.timestamp && <Badge mono>Run {fmtDate(summary.data.timestamp)}</Badge>}
        </div>
        <h2 className="text-[2rem] font-medium tracking-[-0.035em] leading-[1.1] max-w-[40ch]">
          A {n.total}-question SEC benchmark, {n.ans} answerable and {n.unans} designed to be refused.
        </h2>
        <p className="text-sm text-fg-2 mt-2 max-w-[78ch]">
          Five pipeline configurations were run on the same questions. Results are read from{" "}
          <code className="font-mono-num text-xs">results/phase7/phase7_summary.json</code>
          {perQ.data && (
            <>
              {" "}and <code className="font-mono-num text-xs">per_question_results.csv</code>
            </>
          )}{" "}
          and are not recomputed or altered here. New experiments should be reported separately from these.
        </p>
      </section>

      {/* At a glance — five distinct kinds of measurement */}
      <AtAGlance combined={configs.find((c) => c.config_id === COMBINED_ID) ?? null} n={n} plr={plr.data} />

      {/* Selected configuration */}
      <section aria-labelledby="cfg">
        <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
          <div>
            <h2 id="cfg" className="text-xl font-medium tracking-tight">
              {sel.name}
            </h2>
            {sel.description && <p className="text-sm text-fg-2 mt-0.5 max-w-[78ch]">{sel.description}</p>}
          </div>
          <Select value={selected} onChange={(e) => setSelected(e.target.value)} aria-label="Configuration" className="w-64">
            {configs.map((c) => (
              <option key={c.config_id} value={c.config_id}>
                {c.name}
              </option>
            ))}
          </Select>
        </div>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-x-8 gap-y-6">
          {(["Decision", "Answer quality", "Retrieval", "Reasoning", "Temporal", "Latency"] as const).map((g) => (
            <div key={g}>
              <h3 className="text-xs font-medium text-fg-3 mb-1.5">{g}</h3>
              <dl className="divide-y divide-line-soft border-t border-line-soft">
                {METRICS.filter((m) => m.group === g).map((m) => {
                  const v = sel[m.key] as number | undefined;
                  const b = base?.[m.key] as number | undefined;
                  return (
                    <div key={m.key} className="py-2 grid grid-cols-[1fr_auto] gap-x-3" title={m.definition}>
                      <dt className="text-sm text-fg-2">{m.label}</dt>
                      <dd className="text-sm text-fg num text-right">
                        {v == null ? "—" : m.format(v, sel, n)}
                        {v != null && b != null && sel.config_id !== BASELINE_ID && (
                          <div className="text-2xs">
                            <Delta v={v} base={b} better={m.better} pct={PCT_KEYS.has(m.key)} />{" "}
                            <span className="text-fg-3">vs baseline</span>
                          </div>
                        )}
                      </dd>
                      <p className="col-span-2 text-2xs text-fg-3 leading-snug mt-0.5">{m.definition}</p>
                    </div>
                  );
                })}
              </dl>
            </div>
          ))}
        </div>
        <Note tone="warn" className="mt-6 max-w-[90ch]">
          FEIR of 0 means no retrieved or cited evidence post-dated the reference time on this benchmark. It does not show the model
          used no knowledge from pre-training; that is measured separately, below. Decision accuracy counts correct
          answer-or-refuse choices; with exact match at {fmtPct(sel.answerable_exact_match, 0)} and token F1 at{" "}
          {fmtNum(sel.answerable_token_f1, 4)}, answer wording often differs from the reference.
        </Note>
      </section>

      {/* Parametric leakage (Phase 6) */}
      <ParametricLeakageSection data={plr.data} error={plr.error} />

      {/* Charts */}
      <section className="grid grid-cols-1 lg:grid-cols-2 gap-5">
        <ChartFrame
          title="Decision accuracy by configuration"
          caption={`Share of ${n.total} questions with the correct answer-or-refuse decision. Axis starts at 0.`}
          source="phase7_summary.json → configurations"
        >
          <div className="h-[240px]">
            {active && (<ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <BarChart data={decisionData} layout="vertical" margin={{ left: 8, right: 48, top: 4, bottom: 4 }}>
                <CartesianGrid horizontal={false} stroke={t["line-soft"]} />
                <XAxis type="number" domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: t["fg-3"], fontSize: 11 }} stroke={t.line} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fill: t["fg-2"], fontSize: 12 }} stroke={t.line} />
                <Tooltip cursor={{ fill: t["surface-2"] }} contentStyle={tooltipStyle} formatter={(v) => [`${v}%`, "Decision accuracy"]} />
                <Bar dataKey="accuracy" radius={[0, 3, 3, 0]} barSize={18} isAnimationActive={false}>
                  {decisionData.map((d) => (
                    <Cell key={d.id} fill={d.id === selected ? t.accent : t["fg-3"]} fillOpacity={d.id === selected ? 1 : 0.55} />
                  ))}
                  <LabelList dataKey="accuracy" position="right" formatter={(v: unknown) => `${v}%`} style={{ fill: t["fg-2"], fontSize: 11 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>)}
          </div>
        </ChartFrame>

        <ChartFrame
          title="False refusals by configuration"
          caption={`Answerable questions refused, out of ${n.ans}. Lower is better; correct refusals stayed at ${base?.correct_refusals ?? "—"}/${n.unans} in every configuration.`}
          source="phase7_summary.json → configurations"
        >
          <div className="h-[240px]">
            {active && (<ResponsiveContainer width="100%" height="100%" minWidth={0}>
              <BarChart data={decisionData} layout="vertical" margin={{ left: 8, right: 40, top: 4, bottom: 4 }}>
                <CartesianGrid horizontal={false} stroke={t["line-soft"]} />
                <XAxis type="number" domain={[0, 10]} allowDecimals={false} tick={{ fill: t["fg-3"], fontSize: 11 }} stroke={t.line} />
                <YAxis type="category" dataKey="name" width={130} tick={{ fill: t["fg-2"], fontSize: 12 }} stroke={t.line} />
                <Tooltip cursor={{ fill: t["surface-2"] }} contentStyle={tooltipStyle} formatter={(v) => [v as number, "False refusals"]} />
                <Bar dataKey="falseRefusals" radius={[0, 3, 3, 0]} barSize={18} isAnimationActive={false}>
                  {decisionData.map((d) => (
                    <Cell key={d.id} fill={d.id === selected ? t.warn : t["fg-3"]} fillOpacity={d.id === selected ? 1 : 0.55} />
                  ))}
                  <LabelList dataKey="falseRefusals" position="right" style={{ fill: t["fg-2"], fontSize: 11 }} />
                </Bar>
              </BarChart>
            </ResponsiveContainer>)}
          </div>
        </ChartFrame>

        <ChartFrame
          className="lg:col-span-2"
          title="Decision accuracy by question category"
          caption="Frozen baseline compared with the combined Phase 7 configuration. Labels give correct decisions out of questions in the category."
          source={perQ.data ? "per_question_results.csv" : undefined}
        >
          {perQ.error ? (
            <Note tone="warn">
              {perQ.error.isNotFound
                ? "The per-question breakdown needs the /api/v1/benchmark/per-question endpoint. Restart the FastAPI server to load it."
                : perQ.error.message}
            </Note>
          ) : (
            <div className="h-[280px]">
              {active && (<ResponsiveContainer width="100%" height="100%" minWidth={0}>
                <BarChart data={categoryData} margin={{ left: 0, right: 8, top: 20, bottom: 4 }}>
                  <CartesianGrid vertical={false} stroke={t["line-soft"]} />
                  <XAxis dataKey="name" tick={{ fill: t["fg-2"], fontSize: 11 }} stroke={t.line} interval={0} />
                  <YAxis domain={[0, 100]} tickFormatter={(v) => `${v}%`} tick={{ fill: t["fg-3"], fontSize: 11 }} stroke={t.line} width={44} />
                  <Tooltip cursor={{ fill: t["surface-2"] }} contentStyle={tooltipStyle} formatter={(v, name) => [`${v}%`, name]} />
                  <Legend wrapperStyle={{ fontSize: 12, color: t["fg-2"] }} />
                  <Bar name="Frozen baseline" dataKey="baseline" fill={t["fg-3"]} fillOpacity={0.55} radius={[3, 3, 0, 0]} isAnimationActive={false}>
                    <LabelList dataKey="baseLabel" position="top" style={{ fill: t["fg-3"], fontSize: 10 }} />
                  </Bar>
                  <Bar name="Combined (Phase 7)" dataKey="combined" fill={t.accent} radius={[3, 3, 0, 0]} isAnimationActive={false}>
                    <LabelList dataKey="combLabel" position="top" style={{ fill: t["fg-2"], fontSize: 10 }} />
                  </Bar>
                </BarChart>
              </ResponsiveContainer>)}
            </div>
          )}
        </ChartFrame>
      </section>

      {/* Comparison table */}
      <section aria-labelledby="cmp">
        <h2 id="cmp" className="text-xl font-medium tracking-tight">
          Configuration comparison
        </h2>
        <p className="text-sm text-fg-2 mt-0.5 mb-3">All five ablation configurations on the same {n.total} questions.</p>
        <div className="border border-line-soft rounded-md overflow-x-auto">
          <table className="w-full text-sm min-w-[880px]">
            <thead>
              <tr className="text-left text-xs text-fg-3 bg-surface-2/60 border-b border-line-soft">
                <th className="font-medium px-4 py-2.5">Configuration</th>
                {[
                  ["Decision acc.", "decision_accuracy"],
                  ["False refusals", "false_refusals"],
                  ["Correct refusals", "correct_refusals"],
                  ["Answerable F1", "answerable_token_f1"],
                  ["Exact match", "answerable_exact_match"],
                  ["MRR", "mean_mrr"],
                  ["Multi-hop", "multi_hop_path_accuracy"],
                  ["Cross-doc", "cross_doc_coverage"],
                  ["Mean latency", "mean_latency_seconds"],
                ].map(([h]) => (
                  <th key={h} className="font-medium px-3 py-2.5 text-right whitespace-nowrap">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody className="divide-y divide-line-soft">
              {configs.map((c) => {
                const on = c.config_id === selected;
                return (
                  <tr
                    key={c.config_id}
                    onClick={() => setSelected(c.config_id)}
                    className={cx("cursor-pointer", on ? "bg-accent/[0.06]" : "hover:bg-surface-2/50")}
                  >
                    <td className="px-4 py-2.5">
                      <span className={cx(on && "font-medium")}>{c.name}</span>
                    </td>
                    <td className="px-3 py-2.5 text-right num">{fmtPct(c.decision_accuracy, 2)}</td>
                    <td className="px-3 py-2.5 text-right num">
                      {c.false_refusals} <span className="text-fg-3">/ {n.ans}</span>
                    </td>
                    <td className="px-3 py-2.5 text-right num">
                      {c.correct_refusals ?? "—"} <span className="text-fg-3">/ {n.unans}</span>
                    </td>
                    <td className="px-3 py-2.5 text-right num">{fmtNum(c.answerable_token_f1, 4)}</td>
                    <td className="px-3 py-2.5 text-right num">{fmtPct(c.answerable_exact_match, 0)}</td>
                    <td className="px-3 py-2.5 text-right num">{fmtNum(c.mean_mrr, 4)}</td>
                    <td className="px-3 py-2.5 text-right num">{fmtPct(c.multi_hop_path_accuracy, 1)}</td>
                    <td className="px-3 py-2.5 text-right num">{fmtPct(c.cross_doc_coverage, 1)}</td>
                    <td className="px-3 py-2.5 text-right num">{fmtNum(c.mean_latency_seconds, 1)} s</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </section>

      {/* Per-question */}
      <section aria-labelledby="pq">
        <div className="flex flex-wrap items-end justify-between gap-3 mb-3">
          <div>
            <h2 id="pq" className="text-xl font-medium tracking-tight">
              Question-level outcomes
            </h2>
            <p className="text-sm text-fg-2 mt-0.5">Frozen baseline against the combined configuration, question by question.</p>
          </div>
          {perQ.data && (
            <div className="flex flex-wrap items-center gap-2">
              <Select value={category} onChange={(e) => setCategory(e.target.value)} aria-label="Category" className="w-52">
                <option value="ALL">All categories</option>
                {Array.from(new Set(perQ.data.questions.map((q) => q.category))).map((c) => (
                  <option key={c} value={c}>
                    {humanize(c)}
                  </option>
                ))}
              </Select>
              <Segmented
                label="Outcome filter"
                value={outcome}
                onChange={setOutcome}
                options={[
                  { value: "all", label: "All" },
                  { value: "wrong", label: "Incorrect" },
                  { value: "changed", label: "Changed" },
                ]}
              />
            </div>
          )}
        </div>
        {perQ.error ? (
          <Note tone="warn">
            {perQ.error.isNotFound ? "Restart the FastAPI server to load the per-question endpoint." : perQ.error.message}
          </Note>
        ) : !perQ.data ? (
          <Skeleton className="h-40 w-full" />
        ) : (
          <div className="border border-line-soft rounded-md overflow-x-auto">
            <table className="w-full text-sm min-w-[760px]">
              <thead>
                <tr className="text-left text-xs text-fg-3 bg-surface-2/60 border-b border-line-soft">
                  <th className="font-medium px-4 py-2.5">Question</th>
                  <th className="font-medium px-3 py-2.5">Category</th>
                  <th className="font-medium px-3 py-2.5">Expected</th>
                  <th className="font-medium px-3 py-2.5">Baseline</th>
                  <th className="font-medium px-3 py-2.5">Combined</th>
                  <th className="font-medium px-3 py-2.5 text-right">F1</th>
                  <th className="font-medium px-3 py-2.5 text-right">Latency</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line-soft">
                {questions.map((q: PerQuestionItem) => (
                  <tr key={q.question_id}>
                    <td className="px-4 py-2 font-mono-num text-xs">{q.question_id}</td>
                    <td className="px-3 py-2 text-xs text-fg-2">{humanize(q.category)}</td>
                    <td className="px-3 py-2 text-xs text-fg-2">{q.answerable ? "Answer" : "Refuse"}</td>
                    <td className="px-3 py-2 text-xs">
                      <span className={q.baseline_decision_correct ? "text-fg-2" : "text-neg"}>
                        {humanize(q.baseline_decision)} {q.baseline_decision_correct ? "" : "(incorrect)"}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-xs">
                      <span className={q.combined_decision_correct ? "text-fg" : "text-neg"}>
                        {humanize(q.combined_decision)} {q.combined_decision_correct ? "" : "(incorrect)"}
                      </span>
                    </td>
                    <td className="px-3 py-2 text-right num text-xs">{fmtNum(q.combined_token_f1, 3)}</td>
                    <td className="px-3 py-2 text-right num text-xs">{q.combined_latency_sec == null ? "—" : `${fmtNum(q.combined_latency_sec, 1)} s`}</td>
                  </tr>
                ))}
                {questions.length === 0 && (
                  <tr>
                    <td colSpan={7} className="px-4 py-6 text-center text-sm text-fg-3">
                      No questions match these filters.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        )}
        {perQ.data && (
          <p className="text-xs text-fg-3 mt-2 num">
            {questions.length} of {perQ.data.total} questions shown.
          </p>
        )}
      </section>
    </div>
  );
}

function ParametricLeakageSection({
  data,
  error,
}: {
  data: ParametricLeakageResponse | null;
  error: { message: string; status?: number; isNotFound?: boolean } | null;
}) {
  const p = data?.probes;
  const results = p?.details?.plr_results ?? [];
  const leaked = results.filter((r) => r.parametric_leakage_detected).length;
  const cb = data?.phase8_3?.closed_book;
  const g = data?.phase8_3?.grounded;
  const a2 = cb?.A2_as_of;
  const a1 = cb?.A1_unconstrained;
  const ci = (r?: { wilson95_pct: [number, number] | null }) => (r?.wilson95_pct ? `${r.wilson95_pct[0]}–${r.wilson95_pct[1]}%` : "—");
  return (
    <section aria-labelledby="plr">
      <h2 id="plr" className="text-xl font-medium tracking-tight">
        Parametric knowledge leakage
      </h2>
      <p className="text-sm text-fg-2 mt-0.5 max-w-[80ch]">
        Temporal filtering controls which <em>documents</em> reach the model. It cannot remove what the model learned in
        pre-training, so leakage from the model itself is measured separately, with retrieval switched off.
      </p>
      {error ? (
        <Note tone="warn" className="mt-3">
          {error.status === 404
            ? "These results need the /api/v1/benchmark/parametric-leakage endpoint. Restart the FastAPI server to load it."
            : error.message}
        </Note>
      ) : !data ? (
        <Skeleton className="h-28 w-full mt-3" />
      ) : (
        <>
          <div className="card mt-4 p-5">
            <div className="flex flex-wrap items-center gap-2 mb-2">
              <Badge mono>Phase 8.3</Badge>
              <Badge>qwen3.5:4b</Badge>
              <span className="text-xs text-fg-3">Held-out benchmark of facts first published after each question&rsquo;s reference date</span>
            </div>
            {a2 && a1 ? (
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mt-3">
                <div>
                  <div className="text-xs text-fg-3">PLR (as-of prompt)</div>
                  <div className="num text-2xl font-semibold mt-1">{fmtNum(a2.recalled_all.pct, 1)}%</div>
                  <div className="text-xs text-fg-2 num">
                    {a2.recalled_all.k} of {a2.recalled_all.n}, 95% CI {ci(a2.recalled_all)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-fg-3">Future fact recall (no date)</div>
                  <div className="num text-2xl font-semibold mt-1">{fmtNum(a1.recalled_all.pct, 1)}%</div>
                  <div className="text-xs text-fg-2 num">
                    {a1.recalled_all.k} of {a1.recalled_all.n}, 95% CI {ci(a1.recalled_all)}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-fg-3">PLR, facts public before model release</div>
                  <div className="num text-2xl font-semibold mt-1">{fmtNum(a2.recalled_pre_release_window.pct, 1)}%</div>
                  <div className="text-xs text-fg-2 num">
                    {a2.recalled_pre_release_window.k} of {a2.recalled_pre_release_window.n}; control window{" "}
                    {a2.recalled_post_release_control.k} of {a2.recalled_post_release_control.n}
                  </div>
                </div>
                <div>
                  <div className="text-xs text-fg-3">Grounded pipeline, future-claim rate</div>
                  <div className="num text-2xl font-semibold mt-1">
                    {g?.grounded_future_claim_rate?.pct != null ? `${fmtNum(g.grounded_future_claim_rate.pct, 1)}%` : "—"}
                  </div>
                  <div className="text-xs text-fg-2 num">
                    {g ? `${g.grounded_future_claim_rate.k} of ${g.valid_queries}; FEIR ${fmtNum(g.feir_context?.feir ?? null, 3)}` : "Not run"}
                  </div>
                </div>
              </div>
            ) : (
              <p className="text-sm text-fg-3 mt-2">Not yet run. Results appear here once results/phase8_3 has been written.</p>
            )}
            <p className="text-2xs text-fg-3 mt-4 leading-relaxed">
              Measured on this benchmark only; it is not a universal leakage rate for the model. Method and limitations:
              docs/PHASE_8_3_PARAMETRIC_LEAKAGE_EVALUATION.md.
            </p>
          </div>

          {p && (
            <details className="mt-4 card">
              <summary className="cursor-pointer px-5 py-3 text-sm text-fg-2">
                Earlier labelled probes (Phase 6, {p.plr_total_probes ?? results.length} questions, not the production model)
              </summary>
              <div className="px-5 pb-5">
                <Note tone="warn" className="mb-3">
                  {data.phase6_provenance ??
                    "These probes carry manual labels and no stored model responses. They are not a measurement of Qwen3.5:4B."}
                </Note>
                <div className="overflow-x-auto border border-line-soft rounded-md">
                  <table className="w-full text-sm min-w-[560px]">
                    <thead>
                      <tr className="text-left text-xs text-fg-3 bg-surface-2/60 border-b border-line-soft">
                        <th className="font-medium px-3 py-2">Probe</th>
                        <th className="font-medium px-3 py-2">As of</th>
                        <th className="font-medium px-3 py-2">Later fact</th>
                        <th className="font-medium px-3 py-2">Event date</th>
                        <th className="font-medium px-3 py-2">Label</th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-line-soft">
                      {results.map((r) => (
                        <tr key={r.question_id}>
                          <td className="px-3 py-2 font-mono-num text-xs">{r.question_id}</td>
                          <td className="px-3 py-2 font-mono-num text-xs text-fg-2">{fmtDate(r.reference_time)}</td>
                          <td className="px-3 py-2 text-xs text-fg-2">
                            {(r.evidence_of_future_information ?? "").replace(/^Manually verified future fact presence:\s*/, "")}
                          </td>
                          <td className="px-3 py-2 font-mono-num text-xs text-fg-2">{fmtDate(r.future_event_date)}</td>
                          <td className="px-3 py-2 text-xs text-fg-2">{r.parametric_leakage_detected ? "Labelled leaked" : "Labelled not leaked"}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <p className="text-2xs text-fg-3 mt-2">
                  {leaked} of {results.length} labelled as leaked. Source: {data.source}.
                </p>
              </div>
            </details>
          )}
        </>
      )}
    </section>
  );
}

function GlanceCard({
  kind,
  question,
  value,
  sub,
  tone,
  children,
}: {
  kind: string;
  question: string;
  value: React.ReactNode;
  sub: React.ReactNode;
  tone?: "accent" | "pos" | "warn" | "neg";
  children?: React.ReactNode;
}) {
  return (
    <div className="card p-5 flex flex-col">
      <div className="text-xs font-medium text-fg-2">{kind}</div>
      <div className="text-2xs text-fg-3 mt-0.5 leading-snug min-h-[2.2em]">{question}</div>
      <div
        className={cx(
          "num text-[1.75rem] leading-none font-semibold tracking-tight mt-4",
          tone === "accent" ? "text-accent" : tone === "pos" ? "text-pos" : tone === "warn" ? "text-warn" : tone === "neg" ? "text-neg" : "text-fg"
        )}
      >
        {value}
      </div>
      <div className="text-xs text-fg-2 mt-2 num">{sub}</div>
      {children && <div className="text-2xs text-fg-3 mt-auto pt-3 leading-relaxed">{children}</div>}
    </div>
  );
}

function AtAGlance({
  combined,
  n,
  plr,
}: {
  combined: ConfigurationDetail | null;
  n: { total: number; ans: number; unans: number };
  plr: ParametricLeakageResponse | null;
}) {
  if (!combined) return null;
  const p83 = plr?.phase8_3?.closed_book?.A2_as_of ?? null;
  return (
    <section aria-labelledby="glance">
      <h2 id="glance" className="text-xl font-medium tracking-tight">
        Results at a glance
      </h2>
      <p className="text-sm text-fg-2 mt-0.5 mb-4 max-w-[80ch]">
        Five different things were measured. Each answers a different question, so they are reported separately and never combined
        into a single accuracy figure.
      </p>
      <div className="grid grid-cols-1 sm:grid-cols-2 xl:grid-cols-5 gap-4">
        <GlanceCard
          kind="Decision"
          question="Did it correctly choose to answer or refuse?"
          value={fmtPct(combined.decision_accuracy, 2)}
          sub={`${Math.round(combined.decision_accuracy * n.total)} of ${n.total} decisions`}
          tone="accent"
        >
          {combined.correct_refusals}/{n.unans} correct refusals, {combined.false_refusals}/{n.ans} false refusals,{" "}
          {combined.hallucinations_total}/{n.unans} hallucinations.
        </GlanceCard>
        <GlanceCard
          kind="Answer correctness"
          question="How close were answers to the reference wording?"
          value={fmtNum(combined.answerable_token_f1, 4)}
          sub={`Token F1 on ${n.ans} answerable`}
        >
          Exact match {fmtPct(combined.answerable_exact_match, 0)}. Token overlap penalises paraphrase as well as error.
        </GlanceCard>
        <GlanceCard
          kind="Retrieval"
          question="Did the right passage reach the top results?"
          value={fmtPct(combined.mean_hit_at_10 ?? null, 0)}
          sub="Hit@10, chunk level"
        >
          MRR {fmtNum(combined.mean_mrr, 4)}, cross-document coverage {fmtPct(combined.cross_doc_coverage, 1)}.
        </GlanceCard>
        <GlanceCard
          kind="Temporal leakage"
          question="Did evidence from after the reference date get in?"
          value={fmtNum(combined.feir_context, 2)}
          sub="FEIR in context"
          tone="pos"
        >
          FEIR in cited evidence {fmtNum(combined.feir_cited, 2)}. Measures documents only.
        </GlanceCard>
        <GlanceCard
          kind="Parametric leakage"
          question="Does the model know later facts from pre-training?"
          value={p83 && p83.recalled_all.pct != null ? `${fmtNum(p83.recalled_all.pct, 1)}%` : "—"}
          sub={p83 ? `PLR, ${p83.recalled_all.k} of ${p83.recalled_all.n} facts (Qwen3.5:4B)` : "Phase 8.3 not yet run"}
          tone={p83 ? (p83.recalled_all.k > 0 ? "neg" : "pos") : undefined}
        >
          {p83
            ? `Phase 8.3 held-out benchmark, retrieval disabled. 95% CI ${p83.recalled_all.wilson95_pct?.join("–")}%.`
            : "Measured on Qwen3.5:4B in Phase 8.3. Earlier Phase 6 probes were not run on the production model."}
        </GlanceCard>
      </div>
    </section>
  );
}
