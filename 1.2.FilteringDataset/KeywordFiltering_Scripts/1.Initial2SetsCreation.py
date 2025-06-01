import json
import re
import os
import random

# seed LLM-related keywords
llm_keywords = [
    "llm", "llms", "large language model", "large language models", "language model", "language models"
]
llm_keyword_pattern = re.compile(r'\b(?:' + '|'.join(map(re.escape, llm_keywords)) + r')\b', re.IGNORECASE)

acl_dir = '2.1.ExtractingKeywords/tnt_output/acl_data_tnt_kid'
arxiv_dir = '2.1.ExtractingKeywords/tnt_output/arxiv_data_tnt_kid'

def normalize_text(text):
    return text.replace('\n', ' ').replace('\r', ' ').strip()

def parse_filename(filename):
    parts = filename.split('.')[0].split('_')
    year = re.search(r'\d{4}', filename).group(0)
    venue = ''.join(filter(str.isalpha, parts[0])).upper()
    return venue, year

def load_papers(directory):
    papers = []
    for filename in os.listdir(directory):
        if filename.endswith('.json'):
            venue, year = parse_filename(filename)
            file_path = os.path.join(directory, filename)
            with open(file_path, encoding='utf-8') as f:
                loaded_papers = json.load(f)
                for paper in loaded_papers:
                    paper['venue'] = venue
                    paper['year'] = year
                    paper['source_file'] = filename
                papers.extend(loaded_papers)
    return papers

def filter_papers_by_keywords(papers, keyword_pattern, inverse=False):
    filtered_papers = []
    for paper in papers:
        title = normalize_text(paper.get('title', ''))
        summary = normalize_text(paper.get('summary', ''))
        match = keyword_pattern.search(title) or keyword_pattern.search(summary)
        if (match and not inverse) or (not match and inverse):
            filtered_papers.append(paper)
    return filtered_papers

def select_balanced_papers(papers, total_required):
    selected_papers = []
    papers_by_year = {str(year): [] for year in range(2022, 2025)}
    for paper in papers:
        if paper['year'] in papers_by_year:
            papers_by_year[paper['year']].append(paper)

    per_year = total_required // len(papers_by_year)
    for year, papers_list in papers_by_year.items():
        if len(papers_list) > per_year:
            selected_papers.extend(random.sample(papers_list, per_year))
        else:
            selected_papers.extend(papers_list)

    remaining_papers = total_required - len(selected_papers)
    while remaining_papers > 0:
        for papers_list in papers_by_year.values():
            if papers_list and remaining_papers > 0:
                selected_papers.append(papers_list.pop())
                remaining_papers -= 1

    return selected_papers[:total_required]

def print_source_file_statistics(papers):
    """Print statistics for 'source_file' showing how many papers were selected from each file."""
    source_file_stats = {}
    for paper in papers:
        if paper['source_file'] in source_file_stats:
            source_file_stats[paper['source_file']] += 1
        else:
            source_file_stats[paper['source_file']] = 1
    for file, count in source_file_stats.items():
        print(f"From {file}: {count} papers")

acl_papers = load_papers(acl_dir)
arxiv_papers = load_papers(arxiv_dir)

llm_acl = filter_papers_by_keywords(acl_papers, llm_keyword_pattern)
llm_arxiv = filter_papers_by_keywords(arxiv_papers, llm_keyword_pattern)
non_llm_acl = filter_papers_by_keywords(acl_papers, llm_keyword_pattern, inverse=True)
non_llm_arxiv = filter_papers_by_keywords(arxiv_papers, llm_keyword_pattern, inverse=True)

selected_llm_acl = select_balanced_papers(llm_acl, 25)
selected_llm_arxiv = select_balanced_papers(llm_arxiv, 25)
selected_non_llm_acl = select_balanced_papers(non_llm_acl, 25)
selected_non_llm_arxiv = select_balanced_papers(non_llm_arxiv, 25)

final_llm_papers = selected_llm_acl + selected_llm_arxiv
final_non_llm_papers = selected_non_llm_acl + selected_non_llm_arxiv

output_path_llm = '2.1.ExtractingKeywords/TEMP_Iterations/1.filtered_llm_papers.json'
output_path_non_llm = '2.1.ExtractingKeywords/TEMP_Iterations/1.filtered_non_llm_papers.json'

with open(output_path_llm, 'w', encoding='utf-8') as f:
    json.dump(final_llm_papers, f, ensure_ascii=False, indent=4)
with open(output_path_non_llm, 'w', encoding='utf-8') as f:
    json.dump(final_non_llm_papers, f, ensure_ascii=False, indent=4)

# Print source file statistics
print("Statistics for LLM papers:")
print_source_file_statistics(final_llm_papers)
print("Statistics for Non-LLM papers:")
print_source_file_statistics(final_non_llm_papers)

print(f"Selected {len(final_llm_papers)} LLM papers.")
print(f"Selected {len(final_non_llm_papers)} Non-LLM papers.")
