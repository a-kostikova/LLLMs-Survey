# 📚 Clustering LLM Limitation Papers

This folder contains the full pipeline for clustering high-rated LLM-related papers using two methods: **LlooM** (Lam et al., 2024) and **HDBSCAN**.

---

## 1. Preparation

- **KeyphraseGeneration.py**  
  Filters papers rated 3–5 and generates keyphrases  
  (following [Viswanathan et al., 2023](https://aclanthology.org/2024.tacl-1.18/)).

  → Outputs:
  - `ACL_3-4-5-PapersKeyphrases.json`
  - `arxiv_3-4-5-PapersKeyphrases.json`

---

## 2. Clustering

- **LlooM**: Concept-guided clustering using LLM-based evaluation prompts.
- **HDBSCAN+BERTopic**: Unsupervised clustering.

See the subfolders for implementation.