import os
import json
from collections import defaultdict

acl_folder_path = "github/ForUpload/2.Filtering_dataset/acl_filtered_data"
arxiv_folder_path = "github/ForUpload/2.Filtering_dataset/arxiv_filtered_data"

conference_counts = defaultdict(int)

for filename in os.listdir(acl_folder_path):
    if filename.endswith(".json"):
        conference_name = filename.replace(".json", "").upper()
        file_path = os.path.join(acl_folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                conference_counts[conference_name] = len(data)
            except json.JSONDecodeError:
                print(f"Error reading {filename}, skipping...")

arxiv_counts = defaultdict(int)

for filename in os.listdir(arxiv_folder_path):
    if filename.endswith(".json"):
        file_path = os.path.join(arxiv_folder_path, filename)

        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
                for paper in data:
                    if "published" in paper:
                        year = paper["published"][:4]  # Extract year from the date
                        arxiv_counts[year] += 1
            except json.JSONDecodeError:
                print(f"Error reading {filename}, skipping...")

print("\nConference Papers Count:")
for conf, count in sorted(conference_counts.items()):
    print(f"{conf}: {count} papers")

# Print ArXiv results
print("\nArXiv Papers Count by Year:")
for year, count in sorted(arxiv_counts.items()):
    print(f"{year}: {count} papers")
