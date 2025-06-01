import pandas as pd
from sklearn.metrics import f1_score
from nltk.tokenize import word_tokenize
import itertools

# Load data
file_path = '4.ModelsEvaluation/PromptExperimentsResults/Llama/FullGS445_Llama-3.1-70B_Prompt1-3.csv'
data = pd.read_csv(file_path, delimiter=";")

# Row ranges for each annotator
annotator_row_ranges = {
    "annotator1": range(1, 446),  # All rows
    "annotator2": range(1, 106),  # Rows 1–105
    "annotator3": range(1, 251),  # Rows 1–250
    "annotator4": range(75, 106)  # Rows 75–105
}

def tokenize_text(text):
    return word_tokenize(str(text).lower())

def generate_bio_tags(tokens, evidence_tokens_set):
    tags = ['O'] * len(tokens)
    for i, token in enumerate(tokens):
        if token in evidence_tokens_set:
            tags[i] = 'B-EVID' if (i == 0 or tags[i - 1] == 'O') else 'I-EVID'
    return tags

def convert_highlights_to_set(highlights):
    return set(word_tokenize(highlights.lower())) if pd.notna(highlights) and highlights.strip() else set()

def calculate_f1(tags1, tags2):
    if not tags1 or not tags2:
        return None
    if all(t == 'O' for t in tags1) and all(t == 'O' for t in tags2):
        return None
    return round(f1_score(tags1, tags2, labels=['B-EVID', 'I-EVID'], average='weighted', zero_division=0), 2)

annotator_cols = [col for col in data.columns if col.startswith("annotator") and col.endswith("_evidence")]
prompt_cols = [col for col in data.columns if col.startswith("Evidence_Prompt")]

results = []
for index, row in data.iterrows():
    text = row['Abstract']
    title = row['Title']
    tokens = tokenize_text(text)

    f1_results = {'title': title, 'summary': text}

    for annotator_col in annotator_cols:
        annotator = annotator_col.replace("_evidence", "")
        valid_range = annotator_row_ranges.get(annotator, range(1, len(data) + 1))

        # Skip if this row is outside the annotator's assigned range
        if index + 1 not in valid_range:
            continue

        annotator_tokens = convert_highlights_to_set(row[annotator_col])
        annotator_tags = generate_bio_tags(tokens, annotator_tokens)

        for prompt_col in prompt_cols:
            rate_col = f"Rate_{prompt_col.replace('Evidence_', '')}"
            rate = int(row[rate_col]) if rate_col in row and pd.notna(row[rate_col]) else None
            if rate is not None and rate in [0, 1, 2]:
                continue

            prompt_tokens = convert_highlights_to_set(row[prompt_col])
            prompt_tags = generate_bio_tags(tokens, prompt_tokens)

            col_name = f"{annotator} vs {prompt_col}"
            f1_results[col_name] = calculate_f1(annotator_tags, prompt_tags)

    results.append(f1_results)

results_df = pd.DataFrame(results)

f1_columns = [col for col in results_df.columns if "vs" in col]
average_f1 = {'title': 'AVG', 'summary': ''}
for col in f1_columns:
    valid_scores = results_df[col].dropna()
    average_f1[col] = round(valid_scores.mean(), 2) if not valid_scores.empty else None
results_df = pd.concat([results_df, pd.DataFrame([average_f1])], ignore_index=True)

output_path = '4.ModelsEvaluation/PromptExperimentsResults/Llama/FullGS445_Llama-3.1-70B_Prompt1-3_EvidenceCheck.csv'
results_df.to_csv(output_path, index=False)
