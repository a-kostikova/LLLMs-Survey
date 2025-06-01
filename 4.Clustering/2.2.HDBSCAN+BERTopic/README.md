# HDBSCAN + BERTopic Clustering Pipeline

- Implemented in the main script (`UMAPInlierDist.py`, see source).
- Combines grid/random search over UMAP and HDBSCAN hyperparameters.
- Input: Cleaned text from `Evidence_Llama-3.1-70b` + extracted keyphrases.
- Embeddings obtained using `text-embedding-3-large` via OpenAI API.

---

## Outputs

Each run produces:

- `clustered_evidence_*.json`: Per-paper topic assignment.
- `topic_info_*.csv`: Topic ID, label, top keywords, and representative documents.
- `histogram*.png`: Distribution of distances (used to optionally reassign outliers).
- `result.json`: Basic stats.
