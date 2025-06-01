import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load the combined annotations file
df = pd.read_csv('github/4.ModelsEvaluation/PromptExperimentsResults/GPT/FullGS445_GPT4_Prompt1-3.csv', delimiter=";")

unique_full_labels = [0, 1, 2, 3, 4, 5]

unique_aggregated_labels = ['0', '1', '2-3', '4-5']

def aggregate_labels(series):
    return series.replace({
        0: '0',
        1: '1',
        2: '2-3',
        3: '2-3',
        4: '4-5',
        5: '4-5'
    })

def calculate_confusion_matrix_single(y_true, y_pred, labels):
    temp_df = df[[y_true, y_pred]].dropna()
    return confusion_matrix(temp_df[y_true], temp_df[y_pred], labels=labels)

def visualize_matrix(matrix, labels, #title,
                     filename):
    plt.figure(figsize=(3.8, 3.5))
    sns.set_theme(style="white")

    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, #linewidths=0.5, linecolor="gray",
                annot_kws={"size": 12})
    plt.xlabel("Predicted", fontsize=11)
    plt.ylabel("True", fontsize=11)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.88, bottom=0.12)
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight', dpi=300)
    plt.show()

# ================= Model vs. Final Label: Non-Aggregated =================
model_full_matrix = calculate_confusion_matrix_single("FinalLabel", "Rate_GPT4_Prompt3", unique_full_labels)

# Save non-aggregated confusion matrix
visualize_matrix(model_full_matrix, unique_full_labels,
                 'GPT4_FinalLabel_Full.pdf')

# ================= Model vs. Final Label: Aggregated =================
df['Final_Label_Aggregated'] = aggregate_labels(df['FinalLabel'])
df['Model_Aggregated'] = aggregate_labels(df['Rate Limitations of LLMs_Prompt3'])

model_aggregated_matrix = calculate_confusion_matrix_single("Final_Label_Aggregated", "Model_Aggregated", unique_aggregated_labels)

# Save aggregated confusion matrix
visualize_matrix(model_aggregated_matrix, unique_aggregated_labels,
                 'GPT4_FinalLabel_Aggregated.pdf')
