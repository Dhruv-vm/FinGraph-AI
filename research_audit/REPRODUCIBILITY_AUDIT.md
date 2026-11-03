# FinGraph AI — Research Reproducibility & Integrity Audit

**Audit Date:** 2026-11-01  
**Auditor:** Reproducibility Auditor  

---

## 1. Environment & Dependency Audit

The repository reproducibility was audited against fresh standalone execution standards:
- **Python Version:** 3.12+ (Audited under Python 3.12 / 3.14).
- **Package Management:** `pyproject.toml` and `requirements.txt` correctly specify pinned versions for `qdrant-client`, `sentence-transformers`, `fastapi`, `pydantic>=2.0`, `beautifulsoup4`, `networkx`.
- **Hardware Prerequisites:** Runs entirely on commodity CPU/Apple Silicon hardware. Embedded Qdrant operates locally without external Docker requirements. Ollama runs locally with `qwen3.5:4b` quantized model (~2.8 GB VRAM/RAM).

---

## 2. Step-by-Step Reproduction Script

An independent researcher can reproduce the Phase 7 benchmark results in four deterministic steps:

```bash
# 1. Clone repository and install dependencies
git clone https://github.com/dhruv/FinGraph-AI.git
cd FinGraph-AI
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. Verify local Qdrant collection (8,181 vectors)
python3 -c "
from src.retrieval.vector import VectorStore
vs = VectorStore()
print('Collection points count:', vs.client.get_collection('fingraph_chunks').points_count)
"
# Expected output: Collection points count: 8181

# 3. Verify Ollama local model availability
ollama list | grep qwen3.5:4b

# 4. Execute automated Phase 7 benchmark evaluation
python3 scripts/run_phase7_evaluation.py
```

---

## 3. Determinism & Seed Integrity

- **Embedding Determinism:** `sentence-transformers/all-MiniLM-L6-v2` produces deterministic 384-dimensional floating point vectors across platform architectures.
- **RRF & Reranker Determinism:** RRF ranking and deterministic heuristic reranking are strictly mathematical and permutation-invariant for fixed inputs.
- **LLM Sampling Temperature:** Generator defaults to `temperature=0.0`, enforcing deterministic token generation for all factual benchmark responses.
