import numpy as np
from sklearn.metrics import (
    f1_score,
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def compute_metrics(y_true, y_pred, class_names: List[str]) -> Dict:
    accuracy = accuracy_score(y_true, y_pred)

    macro_f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    per_class_metrics = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics = {
        'accuracy': float(accuracy),
        'macro_f1': float(macro_f1),
        'weighted_f1': float(weighted_f1),
        'per_class': {
            class_names[i]: {
                'precision': float(per_class_metrics[0][i]),
                'recall': float(per_class_metrics[1][i]),
                'f1': float(per_class_metrics[2][i]),
                'support': int(per_class_metrics[3][i])
            }
            for i in range(len(class_names))
        }
    }

    return metrics


def print_metrics(metrics: Dict, title: str = "Evaluation Metrics"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"Weighted F1: {metrics['weighted_f1']:.4f}")

    print("\nPer-Class Metrics:")
    print("-" * 80)
    print(f"{'Class':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 80)

    per_class = metrics['per_class']
    for class_name, class_metrics in per_class.items():
        print(
            f"{class_name:<10} "
            f"{class_metrics['precision']:>10.4f} "
            f"{class_metrics['recall']:>10.4f} "
            f"{class_metrics['f1']:>10.4f} "
            f"{class_metrics['support']:>10.0f}"
        )
    print("=" * 80)


def plot_confusion_matrix(
    y_true,
    y_pred,
    class_names: List[str],
    save_path: str = None,
    normalize: bool = True
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm,
        annot=True,
        fmt='.2f' if normalize else 'd',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={'label': 'Normalized Count' if normalize else 'Count'}
    )
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()

    plt.close()


def save_classification_report(
    y_true,
    y_pred,
    class_names: List[str],
    save_path: str
):
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0
    )

    with open(save_path, 'w') as f:
        f.write(report)

    print(f"Classification report saved to {save_path}")
