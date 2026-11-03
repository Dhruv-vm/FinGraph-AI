// Phase 8.3 runner executed inside the FinGraph web page (Playwright on the author's machine).
// The page fetches /__plb/benchmark.json and /__plb/ollama/* which Playwright routes to the benchmark
// file and to the local Ollama daemon (http://localhost:11434). /api/v1/query goes to FastAPI via the
// Next.js rewrite. Results are kept in localStorage (resumable) and exported with storageState.
async function plbRun(cfg, which) {
  const bench = await (await fetch('/__plb/benchmark.json', { cache: 'no-store' })).json();
  const key = which === 'closed' ? 'plb83_closed' : 'plb83_grounded';
  const done = JSON.parse(localStorage.getItem(key) || '[]');
  const seen = new Set(done.map(r => r.fact_id + '|' + (r.condition || 'G')));
  window.__plb = { which, total: which === 'closed' ? bench.facts.length * 3 : bench.facts.length, done: done.length, running: true, started: new Date().toISOString(), last: null };
  const save = () => localStorage.setItem(key, JSON.stringify(done));
  for (const fact of bench.facts) {
    const conds = which === 'closed' ? ['A1_unconstrained', 'A2_as_of', 'A3_forced_estimate'] : ['G'];
    for (const cond of conds) {
      if (seen.has(fact.fact_id + '|' + cond)) continue;
      const t0 = performance.now();
      try {
        if (which === 'closed') {
          const system = cfg.system_prompts[cond].replace('{reference_date}', fact.reference_time.slice(0, 10));
          const body = { model: cfg.model, stream: false, think: cfg.think, format: cfg.format, options: cfg.decoding,
                         messages: [{ role: 'system', content: system }, { role: 'user', content: fact.question }] };
          const r = await fetch('/__plb/ollama/api/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
          const j = await r.json();
          done.push({ fact_id: fact.fact_id, condition: cond, system, user: fact.question, http_status: r.status,
                      raw: j.message ? j.message.content : null, error: j.error || null, model: j.model, created_at: j.created_at,
                      done_reason: j.done_reason, prompt_eval_count: j.prompt_eval_count, eval_count: j.eval_count,
                      latency_ms: Math.round(performance.now() - t0) });
        } else {
          const req = { query: fact.question, reference_time: fact.reference_time, top_k: cfg.grounded.payload.top_k, candidate_pool_size: cfg.grounded.payload.candidate_pool_size };
          const r = await fetch('/api/v1/query', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(req) });
          const j = await r.json();
          if (r.ok) {
            j.evidence = (j.evidence || []).map(e => ({ evidence_id: e.evidence_id, source_type: e.source_type, rank: e.rank, score: e.score,
              document_id: e.document_id, chunk_id: e.chunk_id, available_time: e.available_time, source_company: e.source_company,
              provenance_verified: e.provenance_verified, text: (e.text || '').slice(0, 600) }));
          }
          done.push({ fact_id: fact.fact_id, condition: 'G', request: req, http_status: r.status, response: r.ok ? j : null,
                      error: r.ok ? null : j, latency_ms: Math.round(performance.now() - t0) });
        }
      } catch (e) {
        done.push({ fact_id: fact.fact_id, condition: cond, error: String(e), latency_ms: Math.round(performance.now() - t0) });
      }
      save();
      window.__plb.done = done.length; window.__plb.last = fact.fact_id + '|' + cond;
    }
  }
  window.__plb.running = false; window.__plb.finished = new Date().toISOString();
}
