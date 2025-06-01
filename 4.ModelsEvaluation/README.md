# Models Evaluation

This folder contains scripts and results for evaluating the performance of LLMs on limitations rating and evidence extraction.

---

## 1.0 PromptExperiments

Contains scripts and prompt configurations used to run LLMs:

- `GPT.py`, `Llama.py`, `Mistral.py`
- `prompts.json`: 3 prompt templates used across models and tasks.

---

## 1.1 PromptExperimentsResults

Evaluation outputs from GPT-4, LLaMA-3.1-70B, and Mistral-7B used in the evaluation:

- `FullGS445_<Model>_Prompt1-3.csv`: Predicted limitation labels per prompt.
- `FullGS445_<Model>_Prompt1-3_EvidenceCheck.csv`: Token-level evidence extraction results using BIO tagging.
- `<Model>_FinalLabel_Full.pdf`: Confusion matrices.

---

## 2.1 AgreementCheck

Scripts for computing models-vs-humans agreement:

- `1.CohensKappa_MacroF1_Models.py`: Computes macro F1 and Cohen’s Kappa between models and human annotations.
- `2.ConfusionMatrices.py`: Generates confusion matrices for predicted vs. true labels.

---

## 2.2 HighlightingEvidenceCheck

- `EvaluatingEvidenceOutputs_F1.py`: Calculates token-level F1 scores for model-extracted evidence spans using BIO tagging.
