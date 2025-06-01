import pandas as pd
from sklearn.metrics import cohen_kappa_score, f1_score, precision_recall_fscore_support

def calculate_pairwise_kappa(columns, reference_col, data, aggregated=False):
    """
    Calculate Cohen's Kappa between a reference column and a list of columns.
    Supports both unweighted and weighted (quadratic) kappa scores.
    """
    kappa_scores = {}
    for col in columns:
        filtered_data = data[[reference_col, col]].dropna()
        filtered_data = filtered_data[(filtered_data[reference_col].notnull()) & (filtered_data[col].notnull())]

        if not filtered_data.empty:
            if aggregated:
                filtered_data[reference_col] = filtered_data[reference_col].astype('object')
                filtered_data[col] = filtered_data[col].astype('object')

                filtered_data[reference_col] = filtered_data[reference_col].replace({
                    2: '2-3', 3: '2-3', 4: '4-5', 5: '4-5'
                })
                filtered_data[col] = filtered_data[col].replace({
                    2: '2-3', 3: '2-3', 4: '4-5', 5: '4-5'
                })

                filtered_data[reference_col] = pd.Categorical(
                    filtered_data[reference_col], categories=['2-3', '4-5'], ordered=True
                ).codes
                filtered_data[col] = pd.Categorical(
                    filtered_data[col], categories=['2-3', '4-5'], ordered=True
                ).codes

            print(f"\nUnique labels for Cohen's Kappa ({reference_col} vs {col}):")
            print(f"{reference_col}: {filtered_data[reference_col].unique()}")
            print(f"{col}: {filtered_data[col].unique()}")

            unweighted_kappa = cohen_kappa_score(filtered_data[reference_col], filtered_data[col])

            weighted_kappa = cohen_kappa_score(
                filtered_data[reference_col],
                filtered_data[col],
                weights='quadratic'
            )

            kappa_scores[f'{reference_col} vs {col}'] = {
                'Unweighted': unweighted_kappa,
                'Weighted (Quadratic)': weighted_kappa
            }
        else:
            kappa_scores[f'{reference_col} vs {col}'] = 'Not computable (no data)'
    return kappa_scores


def calculate_macro_f1(columns, reference_col, data, aggregated=False):
    """
    Calculate Macro Precision, Recall, and F1 score between a reference column and a list of columns.
    Supports both non-aggregated and aggregated cases.
    """
    macro_metrics = {}
    for col in columns:
        filtered_data = data[[reference_col, col]].dropna()
        filtered_data = filtered_data[(filtered_data[reference_col].notnull()) & (filtered_data[col].notnull())]

        if not filtered_data.empty:
            if aggregated:
                filtered_data[reference_col] = filtered_data[reference_col].astype('object')
                filtered_data[col] = filtered_data[col].astype('object')

                filtered_data[reference_col] = filtered_data[reference_col].replace({
                    2: '2-3', 3: '2-3', 4: '4-5', 5: '4-5'
                })
                filtered_data[col] = filtered_data[col].replace({
                    2: '2-3', 3: '2-3', 4: '4-5', 5: '4-5'
                })

                filtered_data[reference_col] = pd.Categorical(
                    filtered_data[reference_col], categories=['2-3', '4-5'], ordered=True
                ).codes
                filtered_data[col] = pd.Categorical(
                    filtered_data[col], categories=['2-3', '4-5'], ordered=True
                ).codes

            precision, recall, f1, _ = precision_recall_fscore_support(
                filtered_data[reference_col], filtered_data[col],
                average='macro', zero_division=0
            )

            macro_metrics[f'{reference_col} vs {col}'] = {
                'Precision': precision,
                'Recall': recall,
                'Macro F1': f1
            }
        else:
            macro_metrics[f'{reference_col} vs {col}'] = 'Not computable (no data)'
    return macro_metrics

def calculate_label_metrics(columns, reference_col, data):
    """
    Calculate precision, recall, and F1-score for each label.
    """
    label_metrics = {}
    for col in columns:
        filtered_data = data[[reference_col, col]].dropna()
        filtered_data = filtered_data[(filtered_data[reference_col].notnull()) & (filtered_data[col].notnull())]

        if not filtered_data.empty:
            precision, recall, f1, _ = precision_recall_fscore_support(
                filtered_data[reference_col], filtered_data[col],
                average=None, labels=sorted(filtered_data[reference_col].unique()), zero_division=0
            )

            label_metrics[col] = {
                label: {
                    'Precision': precision[i],
                    'Recall': recall[i],
                    'F1-Score': f1[i]
                } for i, label in enumerate(sorted(filtered_data[reference_col].unique()))
            }
        else:
            label_metrics[col] = 'Not computable (no data)'
    return label_metrics

def clean_labels(label):
    if pd.isna(label) or str(label).strip().lower() in ['nan', 'not extracted', 'missing']:
        return None
    try:
        return int(float(label))
    except ValueError:
        return str(label).strip()

def main():
    file_path = '4.ModelsEvaluation/PromptExperimentsResults/GPT/FullGS445_GPT4_Prompt1-3.csv'
    data = pd.read_csv(file_path, delimiter=";")

    data = data.iloc[:]

    # Specify columns for agreement calculation
    reference_col = "FinalLabel"  # Reference column
    gpt_cols_rate = ["Rate_GPT4_Prompt2"]

    # Clean and convert all labels in relevant columns
    columns_to_clean = [reference_col] + gpt_cols_rate
    for col in columns_to_clean:
        if col in data.columns:
            data[col] = data[col].apply(clean_labels)
            data[col] = data[col].astype('Int64')
            unique_labels = data[col].dropna().unique()
            print(f"{col}: {unique_labels}")
        else:
            print(f"{col}: Column not found in data.")

    print("Unique labels in columns for comparison after cleaning:")

    # Non-aggregated Cohen's Kappa and Macro F1 scores
    kappa_non_aggregated = calculate_pairwise_kappa(gpt_cols_rate, reference_col, data, aggregated=False)
    f1_non_aggregated = calculate_macro_f1(gpt_cols_rate, reference_col, data, aggregated=False)

    # Aggregated Cohen's Kappa and Macro F1 scores
    kappa_aggregated = calculate_pairwise_kappa(gpt_cols_rate, reference_col, data, aggregated=True)
    f1_aggregated = calculate_macro_f1(gpt_cols_rate, reference_col, data, aggregated=True)

    # Label-specific metrics
    label_metrics = calculate_label_metrics(gpt_cols_rate, reference_col, data)

    print("\nNon-Aggregated Cohen's Kappa scores (Unweighted and Weighted):")
    for pair, scores in kappa_non_aggregated.items():
        print(f'{pair}:')
        for k, v in scores.items():
            print(f"  {k}: {v:.3f}")

    print("\nNon-Aggregated Macro F1 scores:")
    for pair, score in f1_non_aggregated.items():
        print(f'{pair}: {score}')

    print("\nAggregated Cohen's Kappa scores:")
    for pair, score in kappa_aggregated.items():
        print(f'{pair}: {score}')

    print("\nAggregated Macro F1 scores:")
    for pair, score in f1_aggregated.items():
        print(f'{pair}: {score}')

    print("\nLabel-specific Precision, Recall, and F1-Score:")
    for col, metrics in label_metrics.items():
        if isinstance(metrics, str):
            print(f'{col}: {metrics}')
        else:
            print(f'{col}:')
            for label, scores in metrics.items():
                print(f"  Label {label}: {scores}")

if __name__ == "__main__":
    main()
