import os
import pickle
import pandas as pd
import numpy as np
import spacy
import re
import nltk
from nltk.corpus import stopwords
from openai import OpenAI
import datetime
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from bertopic import BERTopic
import umap
import hdbscan

nlp = spacy.load("en_core_web_sm")
nltk_stopwords = set(stopwords.words('english'))
custom_stopwords = {'model', 'language', 'large', 'task', 'method', 'system', 'data', 'result', 'llm', 'output',
                    'generate', 'input', 'perform', 'answer', 'knowledge', 'gpt', 'performance',
                    'limitation', 'challenge', 'issue', 'evaluation', 'gap', 'problem', 'chatgpt', 'zero-shot', 'few-shot', 'zeroshot', 'fewshot'}
all_stopwords = nltk_stopwords.union(custom_stopwords)

client = OpenAI(
    api_key="")

json_file = "arxiv_3-4-5-PapersKeyphrases.json"
df = pd.read_json(json_file)

# Preprocessing
def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower().strip()
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if token.text not in all_stopwords and token.lemma_ not in all_stopwords])

df['Evidence'] = df.apply(lambda row: preprocess_text(row['Evidence_Llama-3.1-70b'] + ' ' + ' '.join(row['Keyphrases'])), axis=1)

def get_embeddings(texts):
    embeddings = []
    try:
        for text in texts:
            response = client.embeddings.create(input=text, model="text-embedding-3-large")
            embedding = response.data[0].embedding
            embeddings.append(embedding)
        return np.array(embeddings)
    except Exception as e:
        print("Error fetching embeddings:", e)
        return np.array([])

# Load or generate embeddings
embedding_file = "7.Clustering/embeddings_EvidenceKeywordsArxiv.pkl"
if os.path.exists(embedding_file):
    with open(embedding_file, "rb") as f:
        embeddings = pickle.load(f)
    if len(embeddings) != len(df):
        print("Mismatch in dataset size. Regenerating embeddings...")
        embeddings = get_embeddings(df['Evidence'].tolist())
        with open(embedding_file, "wb") as f:
            pickle.dump(embeddings, f)
else:
    embeddings = get_embeddings(df['Evidence'].tolist())
    with open(embedding_file, "wb") as f:
        pickle.dump(embeddings, f)

print("Data and embeddings are ready!")

# -----------------------------
# Utility functions (distances, etc.)
# -----------------------------
"""def get_distance_from_cluster(topic_model, embeddings):
    if not hasattr(topic_model, "topic_embeddings_") or topic_model.topic_embeddings_ is None:
        return [np.nan] * len(embeddings)

    topic_centroids = np.array(topic_model.topic_embeddings_)
    if topic_centroids.shape[0] == 0:
        return [np.nan] * len(embeddings)

    # Euclidean distance from each point to the nearest topic centroid
    distances = [np.min(np.linalg.norm(topic_centroids - emb, axis=1)) for emb in embeddings]
    return distances"""

"""def compute_umap_centroid_distance(umap_embeddings, clusters):
    centroids = {}
    for cluster in set(clusters):
        if cluster == -1:
            continue
        indices = np.where(clusters == cluster)[0]
        centroids[cluster] = np.mean(umap_embeddings[indices], axis=0)
    distances = []
    for emb in umap_embeddings:
        dists = [np.linalg.norm(emb - centroids[c]) for c in centroids]
        distances.append(np.min(dists))
    return np.array(distances)"""

def compute_umap_inlier_distance(umap_embeddings, clusters):
    distances = []
    for i, emb in enumerate(umap_embeddings):
        dists = []
        for cluster in set(clusters):
            if cluster == -1:
                continue
            indices = np.where(clusters == cluster)[0]
            if clusters[i] == cluster:
                # exclude itself if it belongs to this cluster
                other_indices = indices[indices != i]
                if len(other_indices) == 0:
                    continue
                cluster_embeddings = umap_embeddings[other_indices]
            else:
                cluster_embeddings = umap_embeddings[indices]
            d = np.linalg.norm(cluster_embeddings - emb, axis=1)
            dists.append(np.min(d))
        distances.append(np.min(dists) if dists else np.nan)
    return np.array(distances)

"""def compute_cosine_centroid_distance(embeddings, clusters):
    from sklearn.metrics.pairwise import cosine_distances
    centroids = {}
    for cluster in set(clusters):
        if cluster == -1:
            continue
        indices = np.where(clusters == cluster)[0]
        centroids[cluster] = np.mean(embeddings[indices], axis=0)
    distances = []
    for emb in embeddings:
        dists = []
        for c in centroids:
            d = cosine_distances([emb], [centroids[c]])[0][0]
            dists.append(d)
        distances.append(np.min(dists))
    return np.array(distances)"""

"""def compute_cosine_inlier_distance(embeddings, clusters):
    from sklearn.metrics.pairwise import cosine_distances
    distances = []
    for i, emb in enumerate(embeddings):
        dists = []
        for cluster in set(clusters):
            if cluster == -1:
                continue
            indices = np.where(clusters == cluster)[0]
            if clusters[i] == cluster:
                other_indices = indices[indices != i]
                if len(other_indices) == 0:
                    continue
                cluster_embeddings = embeddings[other_indices]
            else:
                cluster_embeddings = embeddings[indices]
            dd = cosine_distances([emb], cluster_embeddings)[0]
            dists.append(np.min(dd))
        distances.append(np.min(dists) if dists else np.nan)
    return np.array(distances)"""

"""def compute_probability_distance(topic_probabilities):
    distances = []
    for probs in topic_probabilities:
        if len(probs) == 0:
            distances.append(1.0)
        else:
            distances.append(1.0 - max(probs))
    return distances"""

def get_best_topic(probs):
    """Return the index of the highest probability, or -1 if empty."""
    return np.argmax(probs) if len(probs) > 0 else -1

def get_representative_docs(topic_model, df_, top_n=3):
    """
    Retrieve the top representative documents for each topic.
    Assumes df_ has a 'Final_Topic' column.
    """
    doc_representatives = {}
    for tpc in topic_model.get_topics():
        if tpc != -1:
            indices = np.where(df_['Final_Topic'] == tpc)[0]
            docs = df_.iloc[indices]['Evidence'].tolist()
            doc_representatives[tpc] = " || ".join(docs[:top_n])
    return pd.DataFrame(doc_representatives.items(), columns=['Topic', 'Representative_Docs'])

# -----------------------------
# Main script
# -----------------------------
def run_full_pipeline(df, embeddings, param_combinations):

    cluster_index = 0
    num_docs = len(df)

    for params in tqdm(param_combinations, desc="Grid Search Progress"):
        try:
            n_neighbors, n_components, min_dist, min_cluster_size, min_samples, min_topic_size = params

            umap_model = umap.UMAP(
                n_neighbors=n_neighbors,
                n_components=n_components,
                min_dist=min_dist,
                metric='cosine'
            )
            hdbscan_model = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean',
                prediction_data=True
            )
            vectorizer = TfidfVectorizer(ngram_range=(1, 3), stop_words='english')

            topic_model = BERTopic(
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                vectorizer_model=vectorizer,
                min_topic_size=min_topic_size,
                calculate_probabilities=True,
            )

            topics, probs = topic_model.fit_transform(df['Evidence'], embeddings)
            if topics is None or len(topics) == 0:
                print(f"No topics found for {params}. Skipping...")
                continue

            df['Topic'] = topics
            df['Topic_Probabilities'] = probs.tolist() if probs is not None else [[]] * len(df)

            num_clusters = len(set(topics)) - (1 if -1 in topics else 0)
            num_outliers_initial = (df['Topic'] == -1).sum()
            outlier_percentage = num_outliers_initial / len(df)

            if not (5 <= num_clusters <= 17 and outlier_percentage < 0.30):
                print(f"Skipping params {params} (Clusters: {num_clusters}, Outliers: {outlier_percentage:.2%})")
                continue

            print(f"Found valid params: {params}")

            # Create a unique directory for this cluster
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            cluster_index += 1
            cluster_dir = f"{timestamp}_{cluster_index}"
            os.makedirs(cluster_dir, exist_ok=True)

            # Save params.json
            param_data = {
                "n_neighbors": n_neighbors,
                "n_components": n_components,
                "min_dist": min_dist,
                "min_cluster_size": min_cluster_size,
                "min_samples": min_samples,
                "min_topic_size": min_topic_size,
            }
            with open(os.path.join(cluster_dir, "params.json"), "w") as f:
                json.dump(param_data, f, indent=4)

            umap_embeddings = umap_model.transform(embeddings)
            clusters = np.array(df['Topic'])

            # Distances
            #dist_a = compute_umap_centroid_distance(umap_embeddings, clusters)
            dist_b = compute_umap_inlier_distance(umap_embeddings, clusters)
            #dist_c = compute_cosine_centroid_distance(embeddings, clusters)
            #dist_d = compute_cosine_inlier_distance(embeddings, clusters)
            #dist_e = compute_probability_distance(probs)

            # Save the BEFORE files
            df_before = df.copy()
            df_before['Final_Topic'] = df_before['Topic']  # no reassignment yet

            before_json_path = os.path.join(cluster_dir, "clustered_evidence_before.json")
            df_before.to_json(before_json_path, orient='records', indent=4)

            # Generate topic_info_before
            topic_counts_before = df_before['Final_Topic'].value_counts().reset_index()
            topic_counts_before.columns = ['Topic', 'Count']
            topic_labels_before = topic_model.get_topic_info()[['Topic', 'Name']]

            topic_representations_before = {
                tpc: ", ".join([word for word, _ in words])
                for tpc, words in topic_model.get_topics().items()
                if tpc != -1
            }
            rep_df_before = pd.DataFrame(topic_representations_before.items(), columns=['Topic', 'Representation'])
            rep_docs_df_before = get_representative_docs(topic_model, df_before)

            corrected_topic_info_before = (
                topic_counts_before
                .merge(topic_labels_before, on='Topic', how='left')
                .merge(rep_df_before, on='Topic', how='left')
                .merge(rep_docs_df_before, on='Topic', how='left')
            )
            before_topicinfo_path = os.path.join(cluster_dir, "topic_info_before.csv")
            corrected_topic_info_before.to_csv(before_topicinfo_path, index=False)

            # Plot the original distribution histogram
            fig, axs = plt.subplots(3, 2, figsize=(12, 15))
            axs = axs.flatten()

            # a) UMAP Centroid Distance
            """axs[0].hist(dist_a, bins=50, color='green', alpha=0.4, label='All Points')
            axs[0].hist(dist_a[clusters != -1], bins=50, color='blue', alpha=0.6, label='Inliers')
            axs[0].hist(dist_a[clusters == -1], bins=50, color='orange', alpha=0.6, label='Outliers')
            axs[0].set_title('UMAP Centroid Distance')
            axs[0].set_xlabel('Distance')
            axs[0].set_ylabel('Frequency')
            axs[0].legend()"""

            # b) UMAP Inlier Distance
            axs[1].hist(dist_b, bins=50, color='green', alpha=0.4, label='All Points')
            axs[1].hist(dist_b[clusters != -1], bins=50, color='blue', alpha=0.6, label='Inliers')
            axs[1].hist(dist_b[clusters == -1], bins=50, color='orange', alpha=0.6, label='Outliers')
            axs[1].set_title('UMAP Inlier Distance')
            axs[1].set_xlabel('Distance')
            axs[1].set_ylabel('Frequency')
            axs[1].legend()

            # c) Cosine Centroid Distance
            """axs[2].hist(dist_c, bins=50, color='green', alpha=0.4, label='All Points')
            axs[2].hist(dist_c[clusters != -1], bins=50, color='blue', alpha=0.6, label='Inliers')
            axs[2].hist(dist_c[clusters == -1], bins=50, color='orange', alpha=0.6, label='Outliers')
            axs[2].set_title('Cosine Centroid Distance')
            axs[2].set_xlabel('Distance')
            axs[2].set_ylabel('Frequency')
            axs[2].legend()"""

            # d) Cosine Inlier Distance
            """axs[3].hist(dist_d, bins=50, color='green', alpha=0.4, label='All Points')
            axs[3].hist(dist_d[clusters != -1], bins=50, color='blue', alpha=0.6, label='Inliers')
            axs[3].hist(dist_d[clusters == -1], bins=50, color='orange', alpha=0.6, label='Outliers')
            axs[3].set_title('Cosine Inlier Distance')
            axs[3].set_xlabel('Distance')
            axs[3].set_ylabel('Frequency')
            axs[3].legend()"""

            # e) Probability Distance
            """dist_e = np.array(dist_e)
            axs[4].hist(dist_e, bins=50, color='green', alpha=0.4, label='All Points')
            axs[4].hist(dist_e[clusters != -1], bins=50, color='blue', alpha=0.6, label='Inliers')
            axs[4].hist(dist_e[clusters == -1], bins=50, color='orange', alpha=0.6, label='Outliers')
            axs[4].set_title('Prob Distance (1 - max probability)')
            axs[4].set_xlabel('Distance')
            axs[4].set_ylabel('Frequency')
            axs[4].legend()"""

            axs[1].axis('off')
            plt.tight_layout()
            hist_path = os.path.join(cluster_dir, "histogram.png")
            plt.savefig(hist_path)
            #plt.show()

            result_data = {}
            result_data['before_inliers'] = int((df_before['Final_Topic'] != -1).sum())
            result_data['before_outliers'] = int((df_before['Final_Topic'] == -1).sum())
            result_data['before_inliers_ratio'] = float(result_data['before_inliers'] / num_docs)
            result_data['before_outliers_ratio'] = float(result_data['before_outliers'] / num_docs)

            threshold_values = np.arange(
                0.2, 0.3, 0.46)

            for t in threshold_values:
                t_str = str(t).replace(".", "")
                df_after = df_before.copy()  # always reassign from 'before' state

                # outlier mask based on original clusters
                outlier_mask = (clusters == -1)
                reassign_mask = outlier_mask & (dist_b < t)
                inlier_mask = (clusters != -1)

                # reassign
                df_after.loc[reassign_mask, 'Final_Topic'] = df_after.loc[reassign_mask, 'Topic_Probabilities'].apply(get_best_topic)

                # count inliers / outliers
                num_inliers_after = int((df_after['Final_Topic'] != -1).sum())
                num_outliers_after = int((df_after['Final_Topic'] == -1).sum())

                result_data[f"th_{t_str}_inliers"] = num_inliers_after
                result_data[f"th_{t_str}_outliers"] = num_outliers_after
                result_data[f"th_{t_str}_inliers_ratio"] = float(num_inliers_after / num_docs)
                result_data[f"th_{t_str}_outliers_ratio"] = float(num_outliers_after / num_docs)

                # Save topic_info_after & evidence_after
                after_json_path = os.path.join(cluster_dir, f"clustered_evidence_after_{t_str}.json")
                df_after.to_json(after_json_path, orient='records', indent=4)

                # Build topic info
                topic_counts_after = df_after['Final_Topic'].value_counts().reset_index()
                topic_counts_after.columns = ['Topic', 'Count']
                topic_labels_after = topic_model.get_topic_info()[['Topic', 'Name']]

                topic_representations_after = {
                    tpc: ", ".join([word for word, _ in words])
                    for tpc, words in topic_model.get_topics().items()
                    if tpc != -1
                }
                rep_df_after = pd.DataFrame(topic_representations_after.items(), columns=['Topic', 'Representation'])
                rep_docs_df_after = get_representative_docs(topic_model, df_after)

                corrected_topic_info_after = (
                    topic_counts_after
                    .merge(topic_labels_after, on='Topic', how='left')
                    .merge(rep_df_after, on='Topic', how='left')
                    .merge(rep_docs_df_after, on='Topic', how='left')
                )

                after_topicinfo_path = os.path.join(cluster_dir, f"topic_info_after_{t_str}.csv")
                corrected_topic_info_after.to_csv(after_topicinfo_path, index=False)

                # Also plot histogram_{t_str}.png for reassign distribution
                # Compare original inliers vs. the newly reassigned outliers
                #dist_a_reassigned = dist_a[reassign_mask]
                dist_b_reassigned = dist_b[reassign_mask]
                #dist_c_reassigned = dist_c[reassign_mask]
                #dist_d_reassigned = dist_d[reassign_mask]
                #dist_e_reassigned = dist_e[reassign_mask]

                fig2, axs2 = plt.subplots(2, 3, figsize=(15, 10))
                axs2 = axs2.flatten()

                """axs2[0].hist(dist_a[inlier_mask], bins=50, color='green', alpha=0.5, label='Original Inliers')
                axs2[0].hist(dist_a_reassigned, bins=50, color='blue', alpha=0.5, label='Reassigned Outliers')
                axs2[0].set_title('UMAP Centroid Distance')
                axs2[0].legend()"""

                axs2[1].hist(dist_b[inlier_mask], bins=50, color='green', alpha=0.5, label='Original Inliers')
                axs2[1].hist(dist_b_reassigned, bins=50, color='blue', alpha=0.5, label='Reassigned Outliers')
                axs2[1].set_title(f'UMAP Inlier Distance (Th = {t})')
                axs2[1].legend()

                """axs2[2].hist(dist_c[inlier_mask], bins=50, color='green', alpha=0.5, label='Original Inliers')
                axs2[2].hist(dist_c_reassigned, bins=50, color='blue', alpha=0.5, label='Reassigned Outliers')
                axs2[2].set_title('Cosine Centroid Distance')
                axs2[2].legend()"""

                """axs2[3].hist(dist_d[inlier_mask], bins=50, color='green', alpha=0.5, label='Original Inliers')
                axs2[3].hist(dist_d_reassigned, bins=50, color='blue', alpha=0.5, label='Reassigned Outliers')
                axs2[3].set_title('Cosine Inlier Distance')
                axs2[3].legend()"""

                """axs2[4].hist(dist_e[inlier_mask], bins=50, color='green', alpha=0.5, label='Original Inliers')
                axs2[4].hist(dist_e_reassigned, bins=50, color='blue', alpha=0.5, label='Reassigned Outliers')
                axs2[4].set_title('Probability Distance')
                axs2[4].legend()"""

                axs2[5].axis('off')
                hist_thresh_path = os.path.join(cluster_dir, f"histogram_{t_str}.png")
                plt.savefig(hist_thresh_path)
                #plt.show()

            # Finally save the result.json
            result_path = os.path.join(cluster_dir, "result.json")
            with open(result_path, "w") as f:
                json.dump(result_data, f, indent=4)

        except Exception as e:
            print(f"Error with params {params}: {e}")

    print("Completed search. All valid clusters processed.")


# -----------------------------
# Example usage
# -----------------------------
# param_combinations = [...]  # define your param combinations
# run_full_pipeline(df, embeddings, param_combinations)
# grid search
import itertools

umap_params = {"n_neighbors": [15, 25, 30, 50],
               "n_components": [5, 10],
               "min_dist": [0.05, 0.1]}
hdbscan_params = {"min_cluster_size": [35, 40, 50], "min_samples": [10, 15, 20]}
bertopic_params = {"min_topic_size": [20, 25], "nr_topics": [11, 12, 13, 14, 15]}

param_combinations = list(itertools.product(
    umap_params["n_neighbors"],
    umap_params["n_components"],
    umap_params["min_dist"],
    hdbscan_params["min_cluster_size"],
    hdbscan_params["min_samples"],
    bertopic_params["min_topic_size"],
    bertopic_params["nr_topics"]
))
run_full_pipeline(df, embeddings, param_combinations)

"""# random search

import itertools
import random

# Define hyperparameter search space
umap_params = {
    "n_neighbors": [15, 25, 30, 50],
    "n_components": [5, 10],
    "min_dist": [0.05, 0.1]
}

hdbscan_params = {
    "min_cluster_size": [35, 40, 50],
    "min_samples": [10, 15, 20]
}

bertopic_params = {
    "min_topic_size": [20, 25],
    "nr_topics": [15, 16]
}

# Generate all possible parameter combinations
all_param_combinations = list(itertools.product(
    umap_params["n_neighbors"],
    umap_params["n_components"],
    umap_params["min_dist"],
    hdbscan_params["min_cluster_size"],
    hdbscan_params["min_samples"],
    bertopic_params["min_topic_size"],
    bertopic_params["nr_topics"]
))

# Set the number of random samples (e.g., 20 random configurations)
num_samples = min(20, len(all_param_combinations))  # Adjust as needed

# Randomly select a subset of parameter combinations
random_param_combinations = random.sample(all_param_combinations, num_samples)

# Run the pipeline with random search
run_full_pipeline(df, embeddings, random_param_combinations)
"""