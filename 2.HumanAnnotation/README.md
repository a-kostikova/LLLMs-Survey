# Human Annotation

This directory contains human-labeled data and scripts for rating annotation agreement and evidence extraction.

---

## Dataset

- **`FullGS445.csv`**  
  Human-annotated dataset with 445 samples. Each entry includes:
  - Limitation rating labels from four annotators (`annotator1_Label`, etc.)
  - Highlighted evidence spans (`annotator1_evidence`, etc.)
  - Final labels (`FinalLabel`), defined by rounding the average of annotators' ratings.

---

## Agreement Evaluation (`1.AgreementCheck/FullGS`)

- **`1.1.CohensKappaHumans.py`**  
  Computes inter-annotator agreement using:
  - Standard Cohen’s Kappa
  - Quadratic weighted Kappa
  - Aggregated label Kappa

- **`1.2.ConfusionMatrix.py`**  
  Generates pairwise confusion matrices.
---

## Evidence Extraction Evaluation (`2.HighlightingEvidenceCheck`)

- **`EvaluatingEvidenceOutputs.py`**  
  Calculates token-level F1 scores using BIO tagging across annotator pairs.

- **`EvidenceEvaluationResults_445.xlsx`**  
  Excel file with pairwise F1 scores for evidence agreement.

---

## Stats

- **`FullGS-Stats.py`**  
  Computes basic dataset statistics (e.g., venue distribution, label frequency, year breakdown).
