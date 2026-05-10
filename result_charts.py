import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def conf_matrix(y_test, y_pred, le, output_prefix=""):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_)

    plt.title('Confusion Matrix', fontsize=14)
    plt.ylabel('True Class', fontsize=12)
    plt.xlabel('Predicted Class', fontsize=12)

    file_name = f"{output_prefix}confusion_matrix.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

    print(f'Chart saved successfully as {file_name}')

# only for rf
def feature_importance(best_model, X, output_prefix=""):
    importances = best_model.feature_importances_
    features = X.columns

    # Sort features from most to least important
    indices = np.argsort(importances)[::-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances[indices], y=features[indices], palette='viridis', hue=features[indices], legend=False)

    plt.title('Random Forest Feature Importances', fontsize=14)
    plt.xlabel('Importance (Feature Importance)', fontsize=12)
    plt.ylabel('Feature', fontsize=12)

    file_name = f"{output_prefix}feature_importance.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

    print(f'Chart saved successfully as {file_name}')


def roc_auc(X_test, y_test, le, best_model, output_prefix=""):
    class_nums = le.transform(le.classes_)
    y_test_bin = label_binarize(y_test, classes=class_nums)
    y_score = best_model.predict_proba(X_test)

    plt.figure(figsize=(10, 8))

    for i, class_name in enumerate(class_nums):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc_value = auc(fpr, tpr)

        plt.plot(fpr, tpr, color=plt.cm.tab10(i), lw=2,
                 label=f'ROC Curve: {class_name} (AUC = {roc_auc_value:.3f})')

    plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random guessing')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)', fontsize=12)
    plt.ylabel('True Positive Rate (TPR)', fontsize=12)
    plt.title('Multiclass ROC Curve', fontsize=14)
    plt.legend(loc="lower right")

    file_name = f"{output_prefix}roc_curve.png"
    plt.savefig(file_name, bbox_inches='tight')
    plt.close()

    print(f"ROC plot generated and saved successfully as {file_name}")
