# Keyword Extraction Pipeline for LLM Paper Filtering

This directory contains scripts and data for extracting and refining LLM-related keywords from paper abstracts using an iterative, LLR-based approach.

## Contents

- `1.Initial2SetsCreation.py` — Creates initial labeled sets of LLM vs. non-LLM papers using seed keywords.
- `LRR_TNT_KID.py` — Computes log-likelihood ratio (LLR) scores for keywords based on TNT-KID output.
- `2.Iterative2SetsCreation_automatic.py` — Iteratively expands the dataset and refines keyword selection.
- `3.FinalFilteringDataset.py` — Final pass to filter the full dataset using the constructed keyword list.
- `keywords_metadata.json` — Tracks keyword inclusion, stats, and selection iteration.
- `run_iterations.sh` — Shell script for running iterative refinement across multiple steps.
- `tnt_output/` — Folder with TNT-KID keyword outputs for all papers.
- `TEMP_Iterations/` — Stores intermediate results from each iteration.

## Usage

1. Run `1.Initial2SetsCreation.py` to generate the initial LLM/non-LLM paper sets.
2. Use `run_iterations.sh` to automate keyword refinement via:
   - `LRR_TNT_KID.py` → keyword scoring
   - `2.Iterative2SetsCreation_automatic.py` → filtering and set updates
3. Run `3.FinalFilteringDataset.py` to apply the final keyword set to the full dataset.

## Output

- Refined keyword list stored in `keywords_metadata.json`
- Filtered paper sets for each iteration saved in `TEMP_Iterations/` (you can discard this folder after the processing is complete)
- Final filtered paper sets in `2.Filtering_dataset`

