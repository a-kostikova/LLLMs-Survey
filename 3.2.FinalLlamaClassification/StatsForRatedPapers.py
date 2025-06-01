import os
import json
from collections import Counter

def count_ratings(folder_path):
    rating_counts = Counter()
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                papers = json.load(f)
                for paper in papers:
                    rating = paper.get("Rate_Llama-3.1-70b")
                    if rating is not None:
                        rating_counts[rating] += 1
    return rating_counts

acl_folder = "5.FinalLlamaClassification/LlamaRatedPapers/acl_llama_limitation_ratings"
arxiv_folder = "5.FinalLlamaClassification/LlamaRatedPapers/arxiv_llama_limitation_ratings"

acl_ratings = count_ratings(acl_folder)
arxiv_ratings = count_ratings(arxiv_folder)

print("ACL Ratings Count:")
for rating, count in sorted(acl_ratings.items()):
    print(f"Rating {rating}: {count}")

print("\nArXiv Ratings Count:")
for rating, count in sorted(arxiv_ratings.items()):
    print(f"Rating {rating}: {count}")
