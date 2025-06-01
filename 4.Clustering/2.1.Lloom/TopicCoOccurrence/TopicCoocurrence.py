import json
import pandas as pd
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# --- CONFIG ---
source_type = "arxiv"  # or "acl"
short_source_name = source_type.upper()

input_path_map = {
    "arxiv": "7.Clustering/2.1.Lloom/Llama_Lloom_ACL/ACL_3-4-5-PapersKeyphrases_LlooM.json",
    "acl": "7.Clustering/2.1.Lloom/Llama_Lloom_arxiv/arxiv_3-4-5-PapersKeyphrases_LlooM.json"
}
input_path = input_path_map[source_type]

with open(input_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

topics_per_paper = [
    list(entry.get("Final_Topic", {}).keys())
    for entry in data if entry.get("Final_Topic")
]

pair_counter = Counter()
for topics in topics_per_paper:
    for combo in combinations(sorted(topics), 2):
        pair_counter[combo] += 1

all_topics = sorted({topic for topics in topics_per_paper for topic in topics})
co_occurrence = pd.DataFrame(0, index=all_topics, columns=all_topics)

for (t1, t2), count in pair_counter.items():
    co_occurrence.loc[t1, t2] = count
    co_occurrence.loc[t2, t1] = count

topic_counts = Counter(topic for topics in topics_per_paper for topic in topics)
for topic, count in topic_counts.items():
    co_occurrence.loc[topic, topic] = count

sorted_topics = [topic for topic, _ in topic_counts.most_common()]
co_occurrence = co_occurrence.loc[sorted_topics, sorted_topics]

num_topics = len(sorted_topics)
figsize = (0.5 * num_topics, 0.5 * num_topics)

plt.figure(figsize=figsize)
mask = np.tril(np.ones_like(co_occurrence, dtype=bool), 0)

sns.set(style="whitegrid", font_scale=1.2)

sns.heatmap(
    co_occurrence,
    mask=mask,
    annot=True,
    fmt="d",
    cmap="Blues",
    square=True,
    cbar=False,
    annot_kws={"size": 9},
    xticklabels=sorted_topics,
    yticklabels=sorted_topics
)

plt.title(f"{short_source_name} Topic Co-occurrence (LlooM)", fontsize=11, pad=10)
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout(pad=0.6)

out_dir = os.path.dirname(input_path)
out_name = f"{short_source_name}_LlooM_CooccurrenceMatrix.pdf"
out_path = os.path.join(out_dir, out_name)

plt.savefig(out_path, format="pdf", dpi=300, bbox_inches='tight')
plt.show()

print(f"Saved: {out_path}")

num_topics_per_paper = [len(topics) for topics in topics_per_paper]
topic_count_distribution = Counter(num_topics_per_paper)

print("\nDistribution of topic counts per paper:")
for n_topics in sorted(topic_count_distribution):
    print(f"  {n_topics} topic(s): {topic_count_distribution[n_topics]} paper(s)")

print("\nPapers with X topics:\n")
for entry in data:
    topics = entry.get("Final_Topic", {})
    if len(topics) == 8:
        title = entry.get("title", "[No title]")
        evidence = entry.get("evidence", "[No evidence]")
        print(f"Title: {title}\nTopics: {list(topics.keys())}\nEvidence: {evidence}\n---\n")
