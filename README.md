# Limitations of LLMs Survey

This repository accompanies the paper  
**LLLMs: A Data-Driven Survey of Evolving Research on Limitations of Large Language Models**  
[arXiv:2505.19240](https://arxiv.org/abs/2505.19240)

<p align="center">
  <img src="ExperimentalPipeline.png" alt="Experimental Pipeline" width="500px"/>
</p>

---

## Repository Structure

### 1.1. Data Crawling
Scripts and outputs for collecting LLM-related papers from **ACL Anthology** and **arXiv**.

### 1.2. Filtering Dataset
Filtering LLM-relevant papers using keyword-based method (scripts and outputs).

### 2. Human Annotation
Ground-truth annotation of limitations ratings and evidence by human annotators used for models evaluation in Step 3.1.

### 3.1. Model Evaluation
Results and experiment scripts for evaluating models (GPT-4o, Llama-3.1-70B, Mistral). Includes prompt outputs and evaluation scripts.

### 3.2. Final Llama Classification
Final classification of the full dataset using Llama (best model as identified in 3.1). Includes rated papers dataset.

### 4. Clustering
Clustering of high-rated LLM limitation papers using:
- **LlooM**
- **HDBSCAN + BERTopic**

These include both scripts and clustered datasets.

---

Each folder contains the necessary **data and scripts** for that step.

For questions or contributions, feel free to open an issue.