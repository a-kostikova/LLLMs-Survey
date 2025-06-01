import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from itertools import combinations

df = pd.read_csv('github/3.Human_Annotation/1.AgreementCheck/FullGS/FullGS445.csv', delimiter=';')
label_columns = [col for col in df.columns if col.endswith('_Label') and col.startswith('annotator')]

if not label_columns:
    raise ValueError("No valid 'annotatorX_Label' columns found in the dataset.")

unique_full_labels = [0, 1, 2, 3, 4, 5]

unique_aggregated_labels = ['0', '1', '2-3', '4-5']

def aggregate_labels(series):
    return series.replace({
        0: '0', 1: '1',
        2: '2-3', 3: '2-3',
        4: '4-5', 5: '4-5'
    })

def calculate_confusion_matrix(df, columns, labels):
    aggregated_matrix = np.zeros((len(labels), len(labels)), dtype=int)

    for annotator_1, annotator_2 in combinations(columns, 2):
        temp_df = df[[annotator_1, annotator_2]].dropna()
        confusion = confusion_matrix(temp_df[annotator_1], temp_df[annotator_2], labels=labels)
        aggregated_matrix += confusion

    symmetrized_matrix = (np.triu(aggregated_matrix) + np.triu(aggregated_matrix, 1).T) // 2
    return symmetrized_matrix

def visualize_matrix(matrix, labels, filename):
    plt.figure(figsize=(3.8, 3.5))
    mask = np.tril(np.ones_like(matrix, dtype=bool), -1)
    sns.set_theme(style="white")
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", mask=mask,
                xticklabels=labels, yticklabels=labels, cbar=False,
                annot_kws={"size": 12})
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

# ================= Full Label Confusion Matrix =================
symmetrized_full_matrix = calculate_confusion_matrix(df, label_columns, unique_full_labels)
visualize_matrix(symmetrized_full_matrix, unique_full_labels,
                 'github/3.Human_Annotation/1.AgreementCheck/FullGS/FullLabelCM_annotators.pdf')

# ================= Aggregated Label Confusion Matrix =================
df_aggregated = df.copy()
for col in label_columns:
    df_aggregated[col] = aggregate_labels(df_aggregated[col])
symmetrized_aggregated_matrix = calculate_confusion_matrix(df_aggregated, label_columns, unique_aggregated_labels)
visualize_matrix(symmetrized_aggregated_matrix, unique_aggregated_labels,
                 'github/3.Human_Annotation/1.AgreementCheck/FullGS/AggregatedLabelCM_annotators.pdf')
