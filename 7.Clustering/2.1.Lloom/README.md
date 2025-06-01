# LlooM

We adapt the original [LlooM method](https://dl.acm.org/doi/10.1145/3613904.3642830) with minor modifications, most notably using **Llama-3.1-70B-Instruct** for concept scoring.  
We provide our batching script for concept scoring (`Llama_Lloom_Batching.py`), but for all other implementation details, refer to the official [LlooM GitHub repository](https://github.com/michelle123lam/lloom).
---

## Folder Structure

### `Llama_Lloom_ACL/` and `Llama_Lloom_arxiv/`
- LlooM clustering outputs for ACL and arXiv papers.
- Includes:
  - `*_cluster_prompts.json`: Prompt set used for concept scoring step.
  - `*_3-4-5-PapersKeyphrases_LlooM.json`: Clustered topics.
  - `Plots/`: Visualizations.

### `Stats/`
- CSV files summarizing topic distribution across datasets:
  - `*_topic_distribution_lloom.csv`

### `TopicCoOccurrence/`
- Co-occurrence matrix generation:
  - `TopicCooccurrence.py`
