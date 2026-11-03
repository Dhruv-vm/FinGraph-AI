# FinGraph AI — Copy-Pasteable LaTeX Equations Reference

This reference provides production-ready LaTeX snippets pre-formatted for IEEE conference double-column format (`IEEEtran.cls`) and ACM format (`acmart.cls`).

---

## Section: Retrieval Methodology

```latex
% Equation: Vector Cosine Similarity
\begin{equation}
\text{sim}_{\text{cos}}(\mathbf{q}, \mathbf{d}) = \frac{\phi(q) \cdot \phi(d)}{\|\phi(q)\|_2 \|\phi(d)\|_2}
\label{eq:cosine}
\end{equation}

% Equation: Reciprocal Rank Fusion
\begin{equation}
\text{RRF}(d) = \sum_{m \in \mathcal{M}} \frac{w_m}{k + \text{rank}_m(d)}
\label{eq:rrf}
\end{equation}

% Equation: Entity-Balanced Quota Allocation
\begin{equation}
\mathcal{S}_{\text{alloc}} = \bigcup_{E_i \in \mathcal{E}_Q} \text{Top}\Big(\mathcal{A}(E_i), \, \min\big(k_{\min}, |\mathcal{A}(E_i)|\big)\Big)
\label{eq:eb_rrf}
\end{equation}
```

---

## Section: Temporal Knowledge Graph & Leakage Mitigation

```latex
% Equation: Point-in-Time Temporal Snapshot
\begin{equation}
G(t_{\text{as\_of}}) = \big( \{v \in V \mid t_{\text{avail}}(v) \le t_{\text{as\_of}}\}, \, \{e \in E \mid t_{\text{avail}}(e) \le t_{\text{as\_of}}\} \big)
\label{eq:snapshot}
\end{equation}

% Equation: Future Evidence Infiltration Rate (FEIR)
\begin{equation}
\text{FEIR}(\mathcal{E}, t_{\text{ref}}) = \frac{1}{|\mathcal{E}|} \sum_{e \in \mathcal{E}} \mathbb{I}\big(t_{\text{available}}(e) > t_{\text{ref}}\big)
\label{eq:feir}
\end{equation}
```

---

## Section: Evaluation Metrics

```latex
% Equation: Token F1
\begin{equation}
\text{F1}_{\text{token}} = \frac{2 \cdot \text{Prec}_{\text{token}} \cdot \text{Rec}_{\text{token}}}{\text{Prec}_{\text{token}} + \text{Rec}_{\text{token}}}
\label{eq:token_f1}
\end{equation}

% Equation: Decision Accuracy
\begin{equation}
\text{Acc}_{\text{decision}} = \frac{\text{TP}_{\text{refusal}} + \text{TN}_{\text{answerable}}}{|\mathcal{Q}|}
\label{eq:decision_acc}
\end{equation}
```
