import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score
from nltk.tokenize import word_tokenize
from itertools import combinations

file_path = '3.Human_Annotation/1.AgreementCheck/FullGS/FullGS445.csv'
data = pd.read_csv(file_path, delimiter = ";")

def tokenize_text(text):
    return word_tokenize(text.lower())

def generate_bio_tags(tokens, evidence_tokens_set):
    """ Generate BIO tags based on whether tokens appear in any evidence set. """
    tags = ['O'] * len(tokens)
    for i, token in enumerate(tokens):
        if token in evidence_tokens_set:
            tags[i] = 'B-EVID' if (i == 0 or tags[i - 1] == 'O') else 'I-EVID'
    return tags

def evaluate_annotations(tokens, bio1, bio2):
    if set(bio1) == {'O'} and set(bio2) == {'O'}:
        return float('nan'), float('nan'), float('nan')
    else:
        precision = precision_score(bio1, bio2, labels=['B-EVID', 'I-EVID'], average='macro', zero_division=0)
        recall = recall_score(bio1, bio2, labels=['B-EVID', 'I-EVID'], average='macro', zero_division=0)
        f1 = f1_score(bio1, bio2, labels=['B-EVID', 'I-EVID'], average='macro', zero_division=0)
        return precision, recall, f1

def convert_highlights_to_set(highlights):
    return set(word_tokenize(highlights.lower())) if pd.notna(highlights) else set()

annotators = [col for col in data.columns if col.startswith("annotator") and col.endswith("_evidence")]

### Returns True if the annotator was assigned to this sample (e.g. annotator2 was only
# annotating evidence for samples 0–104). ###

def is_annotator_expected(index, annotator):
    if annotator == 'annotator1_evidence':
        return True  # annotated all 445
    elif annotator == 'annotator2_evidence':
        return index < 105
    elif annotator == 'annotator3_evidence':
        return index < 250
    elif annotator == 'annotator4_evidence':
        return 75 <= index < 105
    else:
        return False

results = []
for index, row in data.iterrows():
    text = row['Abstract']
    title = row['Title']
    tokens = tokenize_text(text)

    bio_tags = {}
    for annotator in annotators:
        evidence = row[annotator]
        if pd.notna(evidence) and evidence.strip() != "":
            evidence_tokens_set = convert_highlights_to_set(evidence)
            bio_tags[annotator] = generate_bio_tags(tokens, evidence_tokens_set)
        else:
            bio_tags[annotator] = ['O'] * len(tokens)  # still assign 'O's (penalizable)

    result_entry = {'Title': title, 'Abstract': text}
    annotation_pairs = list(combinations(annotators, 2))

    for annotator1, annotator2 in annotation_pairs:
        expected1 = is_annotator_expected(index, annotator1)
        expected2 = is_annotator_expected(index, annotator2)

        if expected1 and expected2:
            tags1 = bio_tags[annotator1]
            tags2 = bio_tags[annotator2]
            precision, recall, f1 = evaluate_annotations(tokens, tags1, tags2)
            key_prefix = f"{annotator1} vs {annotator2}"
            result_entry[f'{key_prefix} - Precision'] = precision
            result_entry[f'{key_prefix} - Recall'] = recall
            result_entry[f'{key_prefix} - F1 Score'] = f1
        else:
            # Leave blank if one or both weren't supposed to annotate this sample
            key_prefix = f"{annotator1} vs {annotator2}"
            result_entry[f'{key_prefix} - Precision'] = ''
            result_entry[f'{key_prefix} - Recall'] = ''
            result_entry[f'{key_prefix} - F1 Score'] = ''

    results.append(result_entry)

results_df = pd.DataFrame(results)
print(results_df)
results_df.to_excel('3.Human_Annotation/2.HighlightingEvidenceCheck/EvidenceEvaluationResults_445.xlsx', index=False)
