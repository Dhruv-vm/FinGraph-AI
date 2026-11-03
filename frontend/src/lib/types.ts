/**
 * TypeScript type definitions for FinGraph AI Terminal.
 * Synchronized with the FastAPI Pydantic schemas in api/schemas/*.py.
 */

export interface ComponentHealth {
  status: "healthy" | "degraded" | "unavailable";
  message?: string | null;
  details: Record<string, any>;
}

export interface HealthResponse {
  status: "healthy" | "degraded" | "unavailable";
  version: string;
  timestamp: string;
  pipeline: ComponentHealth;
  qdrant: ComponentHealth;
  knowledge_graph: ComponentHealth;
  llm: ComponentHealth;
}

export interface QueryRequest {
  query: string;
  reference_time?: string | null;
  top_k?: number;
  candidate_pool_size?: number;
}

export interface CitationItem {
  evidence_id: string;
  document_id?: string | null;
  chunk_id?: string | null;
  source_type?: string | null;
  available_time?: string | null;
  text_snippet?: string | null;
}

export interface EvidenceItem {
  evidence_id: string;
  source_type: string;
  score: number;
  rank: number;
  text: string;
  document_id?: string | null;
  chunk_id?: string | null;
  available_time?: string | null;
  event_time?: string | null;
  publication_date?: string | null;
  source_url?: string | null;
  source_reference?: string | null;
  source_company?: string | null;
  provenance_verified: boolean;
  attribution_status?: string | null;
  metadata: Record<string, any>;
}

export interface SufficiencyDetail {
  sufficient: boolean;
  reason: string;
  missing_entities: string[];
  missing_relationships: string[];
  required_hops: number;
  observed_hops: number;
  evidence_count: number;
  temporal_valid: boolean;
  provenance_complete: boolean;
}

export interface TemporalValidationDetail {
  temporal_valid: boolean;
  reference_time?: string | null;
  context_items_count: number;
  future_items_count: number;
  feir: number;
}

export interface QueryResponse {
  query: string;
  decision: "answer" | "refuse";
  answer: string;
  insufficient_evidence: boolean;
  confidence: number;
  reasoning_summary?: string | null;
  citations: CitationItem[];
  evidence: EvidenceItem[];
  graph_paths: any[];
  sufficiency: SufficiencyDetail;
  temporal_validation: TemporalValidationDetail;
  validation_status: Record<string, any>;
  sources: string[];
  retrieval_rounds: number;
  latency_ms: number;
  timestamp: string;
}

export interface CachedBenchmarkResult {
  decision: "answer" | "refuse";
  token_f1: number;
  generated_answer: string;
  citations: string[];
  latency_sec: number;
  outcome_status: string;
}

export interface DemoQuestionItem {
  question_id: string;
  question: string;
  category: string;
  answerable: boolean;
  expected_hops: number;
  reference_time?: string | null;
  entities: string[];
  ground_truth_answer: string;
  unanswerability_rationale?: string | null;
  cached_phase7_result: CachedBenchmarkResult;
}

export interface DemoQuestionsResponse {
  total_count: number;
  benchmark_baseline: string;
  questions: DemoQuestionItem[];
  note: string;
}

export interface GraphNodeItem {
  id: string;
  label: string;
  type: string;
  properties: Record<string, any>;
  degree: number;
}

export interface GraphEdgeItem {
  id: string;
  source: string;
  target: string;
  relationship: string;
  available_time?: string | null;
  event_time?: string | null;
  confidence?: number | null;
  document_id?: string | null;
  chunk_id?: string | null;
  source_url?: string | null;
  source_reference?: string | null;
  publication_date?: string | null;
  properties: Record<string, any>;
}

export interface SubgraphResponse {
  center_node: string;
  node_count: number;
  edge_count: number;
  nodes: GraphNodeItem[];
  edges: GraphEdgeItem[];
  total_direct_edges?: number | null;
  total_connected_companies?: number | null;
  total_matching?: number | null;
  has_more?: boolean;
  limit?: number | null;
  offset?: number;
  filter_type?: string;
}

export interface CompanyListItem {
  node_id: string;
  ticker: string;
  canonical_name: string;
  edge_count: number;
}

export interface CompanyListResponse {
  total_companies: number;
  companies: CompanyListItem[];
}

export interface CorpusChunkItem {
  chunk_id: string;
  document_id: string;
  company: string;
  document_type?: string | null;
  publication_date?: string | null;
  section?: string | null;
  text: string;
  source_url?: string | null;
  available_time?: string | null;
  event_time?: string | null;
}

export interface CorpusChunkResponse {
  total_returned: number;
  /** Exact count of chunks matching the filters (null if the backend could not count). */
  total_matching?: number | null;
  offset?: string | null;
  chunks: CorpusChunkItem[];
}

export interface AblationConfig {
  config_id: string;
  name: string;
  decision_accuracy: number;
  answerable_token_f1: number;
  answerable_exact_match: number;
  false_refusals: number;
  false_refusal_rate: number;
  correct_answers: number;
  hallucinations: number;
  refusal_recall: number;
  mean_mrr: number;
  cross_doc_coverage: number;
  multi_hop_path_accuracy: number;
  feir_context: number;
  feir_cited: number;
  mean_latency_sec: number;
  median_latency_sec: number;
}

/** Full per-configuration record from results/phase7/phase7_summary.json → configurations. */
export interface ConfigurationDetail {
  config_id: string;
  name: string;
  description?: string;
  decision_accuracy: number;
  refusal_recall: number;
  refusal_precision?: number;
  false_refusal_rate: number;
  false_refusals: number;
  correct_answers_attempted?: number;
  correct_refusals?: number;
  hallucinations_total?: number;
  answerable_token_f1: number;
  answerable_exact_match: number;
  overall_token_f1?: number;
  mean_hit_at_1?: number;
  mean_hit_at_5?: number;
  mean_hit_at_10?: number;
  mean_mrr: number;
  multi_hop_path_accuracy: number;
  cross_doc_coverage: number;
  feir_context: number;
  feir_cited: number;
  mean_latency_seconds?: number;
  median_latency_seconds?: number;
}

export interface BenchmarkSummaryResponse {
  timestamp?: string;
  total_questions?: number;
  answerable_questions?: number;
  unanswerable_questions?: number;
  configurations?: Record<string, ConfigurationDetail>;
  ablation_comparison?: AblationConfig[];
  error?: string;
}

export interface PerQuestionItem {
  question_id: string;
  category: string;
  answerable: boolean;
  expected_hops: number | null;
  baseline_decision: string;
  baseline_token_f1: number | null;
  combined_decision: string;
  combined_token_f1: number | null;
  combined_decision_correct: boolean;
  baseline_decision_correct: boolean;
  outcome_change: string;
  combined_latency_sec: number | null;
}

export interface PerQuestionResponse {
  source: string;
  baseline_label: string;
  combined_label: string;
  total: number;
  questions: PerQuestionItem[];
  error?: string;
}

export interface FilingItem {
  document_id: string;
  company: string;
  document_type?: string | null;
  title?: string | null;
  publication_date?: string | null;
  fiscal_period?: string | null;
  available_time?: string | null;
  source_url?: string | null;
  source_reference?: string | null;
  chunk_count?: number | null;
}

export interface CompanyCoverage {
  company: string;
  filings: number;
  chunks: number;
  by_type: Record<string, number>;
  first_publication_date?: string | null;
  last_publication_date?: string | null;
}

export interface CorpusFilingsResponse {
  total_filings: number;
  total_chunks: number;
  by_type: Record<string, number>;
  companies: CompanyCoverage[];
  filings: FilingItem[];
  source: string;
}

export interface ParametricProbe {
  question_id: string;
  reference_time: string;
  future_information_present: boolean;
  evidence_of_future_information?: string | null;
  verification_method?: string | null;
  future_event_date?: string | null;
  parametric_leakage_detected: boolean;
  retrieval_used: boolean;
  metadata?: Record<string, any>;
}

export interface Phase83Rate {
  k: number;
  n: number;
  pct: number | null;
  wilson95_pct: [number, number] | null;
}

export interface Phase83Metrics {
  benchmark_facts: number;
  closed_book?: Record<string, any> & {
    A1_unconstrained?: { valid_facts_tested: number; recalled_all: Phase83Rate; recalled_pre_release_window: Phase83Rate; recalled_post_release_control: Phase83Rate };
    A2_as_of?: { valid_facts_tested: number; recalled_all: Phase83Rate; recalled_pre_release_window: Phase83Rate; recalled_post_release_control: Phase83Rate };
  };
  grounded?: Record<string, any>;
}

export interface ParametricLeakageResponse {
  source: string;
  phase6_provenance?: string;
  phase8_3_source?: string | null;
  phase8_3?: Phase83Metrics | null;
  timestamp?: string | null;
  report_title?: string | null;
  error?: string;
  probes: {
    retriever_feir?: number | null;
    parametric_plr?: number | null;
    plr_total_probes?: number;
    plr_manually_verified?: number;
    plr_confirmed_non_leakage?: number;
    plr_uncertain?: number;
    feir_stage_metrics?: Record<string, number | null>;
    details?: { plr_results?: ParametricProbe[] };
  } | null;
}
