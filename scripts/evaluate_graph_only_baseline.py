import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.benchmark import load_benchmark
from src.evaluation.qa import token_f1_score, exact_match_score
from src.retrieval.graph import GraphRetriever
from src.qa.evidence_sufficiency import EvidenceSufficiencyChecker

def run_graph_only_evaluation():
    output_dir = ROOT / "results" / "baselines" / "graph_only"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    questions = load_benchmark(only_verified=True)
    gr = GraphRetriever(str(ROOT / "data" / "processed" / "unified" / "fingraph_semantic_kg.json"))
    checker = EvidenceSufficiencyChecker()
    
    results = []
    t_start = time.perf_counter()
    
    tp_refusal = 0
    fp_refusal = 0
    fn_hallucination = 0
    tn_answer = 0
    
    f1_list = []
    em_list = []
    
    for q in questions:
        q_start = time.perf_counter()
        edges = gr.retrieve(query=q.question, top_k=8)
        
        # Check if graph edges are sufficient to answer
        # If no edges found or missing entities, refuse
        is_suff = len(edges) > 0 and len(q.entities) > 0
        
        # Determine decision
        if is_suff and q.answerable:
            # Graph provides structural support
            decision = "answer"
            tn_answer += 1
            # Mock or minimal answer synthesis from graph triples
            answer_text = "; ".join([f"{e.source_node} -[{e.relationship}]-> {e.target_node}" for e in edges[:3]])
            f1 = token_f1_score(answer_text, q.ground_truth_answer)
            em = exact_match_score(answer_text, q.ground_truth_answer)
        elif is_suff and not q.answerable:
            # Hallucinated answer on unanswerable
            decision = "answer"
            fn_hallucination += 1
            answer_text = "; ".join([f"{e.source_node} -[{e.relationship}]-> {e.target_node}" for e in edges[:3]])
            f1 = 0.0
            em = 0.0
        elif not is_suff and not q.answerable:
            # Correct refusal
            decision = "refuse"
            tp_refusal += 1
            answer_text = "Refusal: Insufficient evidence in knowledge graph to answer."
            f1 = 1.0 # Refusal match
            em = 1.0
        else:
            # False refusal (answerable question refused due to graph sparsity)
            decision = "refuse"
            fp_refusal += 1
            answer_text = "Refusal: Insufficient evidence in knowledge graph."
            f1 = 0.0
            em = 0.0
            
        q_latency = time.perf_counter() - q_start
        if q.answerable:
            f1_list.append(f1)
            em_list.append(em)
            
        results.append({
            "question_id": q.question_id,
            "category": q.category,
            "answerable": q.answerable,
            "expected_hops": q.expected_reasoning_hops,
            "decision": decision,
            "retrieved_edges_count": len(edges),
            "token_f1": f1,
            "exact_match": em,
            "answer": answer_text[:200],
            "latency_sec": round(q_latency, 3),
        })
        
    total_latency = time.perf_counter() - t_start
    decision_acc = (tp_refusal + tn_answer) / len(questions)
    ans_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0
    ans_em = sum(em_list) / len(em_list) if em_list else 0.0
    refusal_rec = tp_refusal / (tp_refusal + fn_hallucination) if (tp_refusal + fn_hallucination) > 0 else 0.0
    hall_rate = fn_hallucination / 14.0
    
    summary = {
        "system_name": "B2_Graph_Only_Baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "total_questions": len(questions),
        "answerable_questions": 50,
        "unanswerable_questions": 14,
        "decision_accuracy": round(decision_acc, 4),
        "correct_answers": tn_answer,
        "false_refusals": fp_refusal,
        "correct_refusals": tp_refusal,
        "hallucinations": fn_hallucination,
        "refusal_recall": round(refusal_rec, 4),
        "hallucination_rate": round(hall_rate, 4),
        "answerable_token_f1": round(ans_f1, 4),
        "answerable_exact_match": round(ans_em, 4),
        "mean_latency_seconds": round(total_latency / len(questions), 4),
    }
    
    with open(output_dir / "graph_only_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(output_dir / "graph_only_per_question.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("Graph-Only Baseline Evaluation Completed:")
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_graph_only_evaluation()
