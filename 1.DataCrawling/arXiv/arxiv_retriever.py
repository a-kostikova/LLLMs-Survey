#!/usr/bin/env python
import collections
from collections import defaultdict
import json
import arxiv
import time
import seaborn as sb
from tqdm import tqdm
import numpy as np
from os.path import basename
from datetime import datetime

def search(queries=[], field="all", cats=["cs.CL", "cs.LG", 'cs.AI', 'cs.CV'], start='202411010000', end='202503280000'):
    query_string, client = "", arxiv.Client(num_retries=40, page_size=1000)
    if queries:
        query_string += "(" + " OR ".join(f"{field}:{query}" for query in queries) + ")"
    if cats:
        if query_string:
            query_string += " AND "
        query_string += "(" + " OR ".join(f"cat:{cat}" for cat in cats) + ")"
    query_string += f" AND submittedDate:[{start} TO {end}]"
    print(query_string)
    return client.results(arxiv.Search(
        query=query_string,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    ))

def get_papers(cached=False, start='202411010000', end='202503280000'):
    papers = []

    print("Downloading papers.")
    progress = 0
    print("Progress:", 0)
    count = 1
    for result in search(start=start, end=end):
        published_date = result.published
        pdf_link = f"https://arxiv.org/pdf/{basename(result.entry_id)}.pdf"
        paper_info = {
            "title": result.title,
            "authors": [author.name for author in result.authors],
            "published": published_date.strftime('%Y-%m-%dT%H:%M:%SZ'),
            "summary": result.summary,
            "pdf_link": pdf_link,
            "categories": result.categories
        }
        papers.append(paper_info)
        progress += 1
        if progress % 500 == 0:
            print(f"Progress{count}: {progress}")
            count += 1
        print(json.dumps(paper_info, indent=2, ensure_ascii=False))

    print("Final Progress:", progress)
    return papers

sb.set()

# Date range: 1st Nov 2024 to 28th Mar 2025
start_date = '202411010000'
end_date = '202503280000'
papers = get_papers(cached=False, start=start_date, end=end_date)
json_path = "1.Data_crawling/arXiv_data/data_collected_Nov2024-Mar2025.json"
with open(json_path, 'w', encoding='utf-8') as jsonf:
    json.dump(papers, jsonf, ensure_ascii=False, indent=4)
