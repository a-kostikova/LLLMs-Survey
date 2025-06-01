import json
import re
import os
import sys
import random
from pathlib import Path
import argparse
import string

def normalize_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

def load_papers(directory):
    papers = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            file_path = os.path.join(directory, filename)
            with open(file_path, encoding='utf-8') as f:
                loaded_papers = json.load(f)
                for paper in loaded_papers:
                    paper['source_file'] = filename  # Keep track of the source file
                papers.extend(loaded_papers)
    return papers

def filter_papers_by_keywords(papers, keyword_pattern, inverse=False):
    """Filter papers based on the presence or absence of LLM keywords."""
    filtered_papers = []
    for paper in papers:
        title = normalize_text(paper.get('title', ''))
        summary = normalize_text(paper.get('summary', ''))
        match = keyword_pattern.search(title) or keyword_pattern.search(summary)
        if (match and not inverse) or (not match and inverse):
            filtered_papers.append(paper)
    return filtered_papers

def select_papers(papers, total_required, existing_titles):
    valid_papers = [paper for paper in papers if paper['title'] not in existing_titles]
    selected_papers = random.sample(valid_papers, min(total_required, len(valid_papers)))
    return selected_papers

def load_keywords(file_path, base_threshold, metadata_path, iteration):
    import json
    import os

    threshold_state_path = 'github/2.1.ExtractingKeywords/threshold_state.json'
    default_threshold = base_threshold
    min_gap_between_increases = 10  # iterations to wait before next threshold increase

    # Load or initialize threshold state
    if os.path.exists(threshold_state_path):
        with open(threshold_state_path, 'r') as f:
            threshold_state = json.load(f)
    else:
        threshold_state = {"threshold": default_threshold, "last_increase_iteration": 0}

    threshold = threshold_state["threshold"]
    last_increase = threshold_state["last_increase_iteration"]

    mandatory_keywords = ["llm", "llms", "large language model", "large language models"]
    stop_words = {"the", "of", "in", "and", "to", "with", "on", "models", "model", "language", "large"}

    # Load existing metadata
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    else:
        metadata = {}

    keywords = mandatory_keywords[:]  # Start with mandatory keywords
    fresh_keywords_with_scores = []

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if ': ' not in line:
                continue

            keyword, stats_part = line.split(': ', 1)
            try:
                stats = eval(stats_part)  # Parse the tuple (LLR, accepted_count, rejected_count)
                relevance_score, accepted_count, rejected_count = stats
                if rejected_count > 0.5:  # Exclude noisy or overly generic terms
                    continue

                keyword = keyword.strip().rstrip('.')
                keyword_words = keyword.split()
                keyword = ' '.join(word for word in keyword_words if word not in stop_words)

                if keyword in metadata:
                    continue

                if relevance_score >= threshold:
                    keywords.append(keyword)
                    fresh_keywords_with_scores.append((keyword, relevance_score))
                    metadata[keyword] = {"iteration": iteration, "tuple": stats, "included": True}
            except (SyntaxError, ValueError) as e:
                print(f"Error processing stats for keyword '{keyword}': {e}")

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=4)

    # Decide whether to increase the threshold
    new_keywords_count = len(fresh_keywords_with_scores)
    total_keywords_considered = len(metadata) + new_keywords_count
    ratio_new = new_keywords_count / total_keywords_considered if total_keywords_considered > 0 else 0

    if ratio_new < 0.05 and (iteration - last_increase) >= min_gap_between_increases:
        new_threshold = int(threshold * 1.05)
        print(f"Raising threshold from {threshold} to {new_threshold} due to low novelty ({ratio_new:.2%})")
        threshold_state["threshold"] = new_threshold
        threshold_state["last_increase_iteration"] = iteration

        # Save updated threshold state
        with open(threshold_state_path, 'w') as f:
            json.dump(threshold_state, f, indent=4)

    for keyword, score in fresh_keywords_with_scores:
        print(f"Keyword: {keyword}, Relevance Score: {score}")

    return keywords

def load_existing_papers(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def compile_keyword_pattern(keywords):
    pattern = '|'.join([re.escape(keyword).replace(r'\ ', r'[\s-]*') for keyword in keywords])
    return re.compile(r'\b(?:' + pattern + r')\b', re.IGNORECASE)

def main(args):
    # Paths
    metadata_path = 'github/2.1.ExtractingKeywords/keywords_metadata.json'
    base_threshold = 10

    keywords = load_keywords(
        f'{args.input_folder}/combined_keywords.txt',
        base_threshold,
        metadata_path,
        int(args.input_iteration),
    )

    llm_keyword_pattern = compile_keyword_pattern(keywords)

    acl_papers = load_papers('github/2.1.ExtractingKeywords/tnt_output/acl_data_tnt_kid')
    arxiv_papers = load_papers('github/2.1.ExtractingKeywords/tnt_output/arxiv_data_tnt_kid')

    all_papers = acl_papers + arxiv_papers
    llm_papers = filter_papers_by_keywords(all_papers, llm_keyword_pattern)
    non_llm_papers = filter_papers_by_keywords(all_papers, llm_keyword_pattern, inverse=True)

    existing_llm_papers = load_existing_papers(f'{args.input_folder}/{args.input_iteration}.filtered_llm_papers.json')
    existing_non_llm_papers = load_existing_papers(f'{args.input_folder}/{args.input_iteration}.filtered_non_llm_papers.json')

    existing_titles = {paper['title'] for paper in existing_llm_papers + existing_non_llm_papers}

    # Select 100 LLM and 100 non-LLM papers
    selected_llm_papers = select_papers(llm_papers, 100, existing_titles)
    selected_non_llm_papers = select_papers(non_llm_papers, 100, existing_titles)

    # Append new papers to the existing ones
    updated_llm_papers = existing_llm_papers + selected_llm_papers
    updated_non_llm_papers = existing_non_llm_papers + selected_non_llm_papers

    # Define output paths and save the updated papers
    iteration_dir = Path(args.output_folder)
    iteration_dir.mkdir(parents=True, exist_ok=True)
    output_path_llm = iteration_dir / f'{args.output_iteration}.filtered_llm_papers.json'
    output_path_non_llm = iteration_dir / f'{args.output_iteration}.filtered_non_llm_papers.json'
    with open(output_path_llm, 'w', encoding='utf-8') as f:
        json.dump(updated_llm_papers, f, ensure_ascii=False, indent=4)
    with open(output_path_non_llm, 'w', encoding='utf-8') as f:
        json.dump(updated_non_llm_papers, f, ensure_ascii=False, indent=4)

    print(f"Selected {len(selected_llm_papers)} new LLM papers and {len(selected_non_llm_papers)} new non-LLM papers.")
    print(f"Total LLM papers: {len(updated_llm_papers)}")
    print(f"Total non-LLM papers: {len(updated_non_llm_papers)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Iterative paper filtering script.")
    parser.add_argument('--input-folder', required=True, help="Input folder containing combined_keywords.txt and previous JSON files.")
    parser.add_argument('--input-iteration', required=True, help="Prefix of the input iteration (e.g., '2').")
    parser.add_argument('--output-folder', required=True, help="Output folder to save new filtered JSON files.")
    parser.add_argument('--output-iteration', required=True, help="Prefix for the output iteration (e.g., '3').")
    args = parser.parse_args()
    main(args)

