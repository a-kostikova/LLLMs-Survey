import pandas as pd
from sklearn.metrics import cohen_kappa_score

def calculate_pairwise_kappa(columns, data, weights=None):
    kappa_scores = {}
    kappa_list = []

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):
            col1, col2 = columns[i], columns[j]
            filtered_data = data[[col1, col2]].dropna()

            if not filtered_data.empty:
                score = cohen_kappa_score(filtered_data[col1], filtered_data[col2], weights=weights)
                kappa_scores[f'{col1} vs {col2}'] = score
                kappa_list.append(score)
            else:
                kappa_scores[f'{col1} vs {col2}'] = 'Not computable (no data)'

    avg_kappa = sum(kappa_list) / len(kappa_list) if kappa_list else 'Not computable'
    return kappa_scores, avg_kappa

def clean_labels(label):
    if pd.isna(label):
        return None
    label = str(label).strip()
    if label.lower() in ['nan', 'not extracted', 'missing']:
        return None
    try:
        return int(float(label))
    except ValueError:
        return label

def aggregate_labels(value):
    if pd.isna(value):
        return value
    if value in [2, 3]:
        return 2
    elif value in [4, 5]:
        return 3
    else:
        return value

def main():
    file_path = '3.Human_Annotation/1.AgreementCheck/FullGS/FullGS445.csv'
    data = pd.read_csv(file_path, delimiter=';')

    label_columns = [col for col in data.columns if col.endswith('_Label') and col.startswith('annotator')]

    if not label_columns:
        print("No valid annotator label columns found. Exiting.")
        return

    print(f"Found label columns: {label_columns}")

    for col in label_columns:
        data[col] = data[col].apply(clean_labels)

    print("\n=== Non-Aggregated Cohen's Kappa ===")
    non_agg_kappa, avg_non_agg = calculate_pairwise_kappa(label_columns, data, weights=None)
    for pair, score in non_agg_kappa.items():
        print(f'{pair}: {score}')
    print(f'Average Non-Aggregated Kappa: {avg_non_agg}\n')

    for col in label_columns:
        data[col] = data[col].apply(aggregate_labels)

    print("\n=== Aggregated Cohen's Kappa ===")
    agg_kappa, avg_agg = calculate_pairwise_kappa(label_columns, data, weights=None)
    for pair, score in agg_kappa.items():
        print(f'{pair}: {score}')
    print(f'Average Aggregated Kappa: {avg_agg}\n')

    print("\n=== Quadratic Weighted Cohen's Kappa ===")
    weighted_kappa, avg_weighted = calculate_pairwise_kappa(label_columns, data, weights='quadratic')
    for pair, score in weighted_kappa.items():
        print(f'{pair}: {score}')
    print(f'Average Weighted Kappa: {avg_weighted}\n')

if __name__ == "__main__":
    main()
