from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, precision_score, balanced_accuracy_score, roc_auc_score, \
    classification_report
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)

# Create imbalanced dataset
X0 = np.random.normal(0, 1, (900, 2))
X1 = np.random.normal(2, 1, (100, 2))

X = np.vstack([X0, X1])
y = np.array([0] * 900 + [1] * 100)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

# Train base model
model_base = RandomForestClassifier(random_state=42, n_estimators=100)
model_base.fit(X_train, y_train)

# Base model evaluation
y_pred_base = model_base.predict(X_test)
y_prob_base = model_base.predict_proba(X_test)[:, 1]
cm_base = confusion_matrix(y_test, y_pred_base)
print(cm_base)
print(f"Recall: {recall_score(y_test, y_pred_base):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_base):.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_base):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_base):.4f}")
print("-" * 80)

print("\n" + "=" * 80)
print("SMOTE")
print("=" * 80)

# Apply SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"Training samples after SMOTE: {len(X_train_smote)}")
print(f"Class distribution after SMOTE: {np.bincount(y_train_smote)}\n")

# Train model with SMOTE
model_smote = RandomForestClassifier(random_state=42, n_estimators=100)
model_smote.fit(X_train_smote, y_train_smote)

# SMOTE model evaluation
y_pred_smote = model_smote.predict(X_test)
y_prob_smote = model_smote.predict_proba(X_test)[:, 1]

print("SMOTE Results:")
print("Confusion Matrix:")
cm_smote = confusion_matrix(y_test, y_pred_smote)
print(cm_smote)
recall_smote = recall_score(y_test, y_pred_smote)
precision_smote = precision_score(y_test, y_pred_smote)
print(f"Recall: {recall_smote:.4f}")
print(f"Precision: {precision_smote:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_smote):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_smote):.4f}")

# Compare with base model
print("\nSMOTE vs Base Model:")
print(f"Recall change: {recall_smote - recall_score(y_test, y_pred_base):+.4f}")
print(f"Precision change: {precision_smote - precision_score(y_test, y_pred_base):+.4f}")
print("-" * 80)

print("\n" + "=" * 80)
print("Threshold Tuning")
print("=" * 80)

# Different thresholds
thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
threshold = 0.3

y_pred_custom = (y_prob_smote >= threshold).astype(int)
cm_custom = confusion_matrix(y_test, y_pred_custom)

print(f"Results with threshold {threshold}:")
print("Confusion Matrix:")
print(cm_custom)
recall_custom = recall_score(y_test, y_pred_custom)
precision_custom = precision_score(y_test, y_pred_custom)
print(f"Recall: {recall_custom:.4f}")
print(f"Precision: {precision_custom:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_custom):.4f}")

print(f"\nComparison with default threshold 0.5:")
print(f"Recall: {recall_custom:.4f} vs {recall_smote:.4f}")
print(f"Precision: {precision_custom:.4f} vs {precision_smote:.4f}")

# Show threshold impact
print("\nEffect of different thresholds on Recall and Precision:")
for thresh in thresholds[::2]:
    y_tmp = (y_prob_smote >= thresh).astype(int)
    rec = recall_score(y_test, y_tmp)
    prec = precision_score(y_test, y_tmp)
    print(f"   Threshold={thresh}: Recall={rec:.4f}, Precision={prec:.4f}")

print("-" * 80)

print("\n" + "=" * 80)
print("Drift Simulation")
print("=" * 80)

# Apply drift to test data
print("Applying Feature Drift: Adding 1.5 to feature means")
X_test_drift = X_test.copy()
X_test_drift[:, 0] += 1.5
X_test_drift[:, 1] += 1.5

# Evaluate on drifted data
y_pred_drift = model_smote.predict(X_test_drift)
y_prob_drift = model_smote.predict_proba(X_test_drift)[:, 1]

print("\nSMOTE model performance on drifted data:")
cm_drift = confusion_matrix(y_test, y_pred_drift)
print("Confusion Matrix:")
print(cm_drift)
recall_drift = recall_score(y_test, y_pred_drift)
precision_drift = precision_score(y_test, y_pred_drift)
print(f"Recall: {recall_drift:.4f}")
print(f"Precision: {precision_drift:.4f}")
print(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred_drift):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_prob_drift):.4f}")

print(f"\nPerformance drop after drift:")
print(f"Recall drop: {recall_smote - recall_drift:.4f}")
print(
    f"Balanced Accuracy drop: {balanced_accuracy_score(y_test, y_pred_smote) - balanced_accuracy_score(y_test, y_pred_drift):.4f}")

# Answers to questions
print("\nDrift Analysis:")
print("1. Type of drift:")
print("   Feature Drift (Covariate Shift)")
print("   Reason: Input feature distribution changed, but relationship between X and y remained constant")
print("   (p(x) changed but p(y|x) is stable)")

print("\n2. Why SMOTE doesn't help with drift:")
print("   SMOTE only balances class distribution in training data")
print("   SMOTE does not make model robust to feature distribution changes")
print("   Drift problem is about input distribution change, not class imbalance")
print("   Solutions: Retraining, Domain Adaptation, or using newer data")

print("\n" + "=" * 80)
print("Summary")
print("=" * 80)
print("""
1. SMOTE:
   - Improves Recall and Balanced Accuracy
   - Reduces Precision (increases False Positives)

2. Threshold Tuning:
   - Can adjust trade-off between Recall and Precision
   - Lower threshold increases Recall
   - Lower threshold decreases Precision

3. Drift:
   - SMOTE does not help with drift
   - Model performance drops significantly
   - Need other solutions like retraining or domain adaptation
""")

# Visualization
try:
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Original test data
    axes[0, 0].scatter(X_test[y_test == 0, 0], X_test[y_test == 0, 1], alpha=0.5, label='Class 0', c='blue')
    axes[0, 0].scatter(X_test[y_test == 1, 0], X_test[y_test == 1, 1], alpha=0.5, label='Class 1', c='red')
    axes[0, 0].set_title('Original Test Data')
    axes[0, 0].legend()

    # Drifted test data
    axes[0, 1].scatter(X_test_drift[y_test == 0, 0], X_test_drift[y_test == 0, 1], alpha=0.5, label='Class 0', c='blue')
    axes[0, 1].scatter(X_test_drift[y_test == 1, 0], X_test_drift[y_test == 1, 1], alpha=0.5, label='Class 1', c='red')
    axes[0, 1].set_title('Test Data with Drift')
    axes[0, 1].legend()

    # Recall comparison
    axes[1, 0].bar(['Base', 'SMOTE', 'SMOTE+Threshold', 'Drift'],
                   [recall_score(y_test, y_pred_base), recall_smote, recall_custom, recall_drift])
    axes[1, 0].set_title('Recall Comparison')
    axes[1, 0].set_ylabel('Recall')

    # Precision comparison
    axes[1, 1].bar(['Base', 'SMOTE', 'SMOTE+Threshold', 'Drift'],
                   [precision_score(y_test, y_pred_base), precision_smote, precision_custom, precision_drift])
    axes[1, 1].set_title('Precision Comparison')
    axes[1, 1].set_ylabel('Precision')

    plt.tight_layout()
    plt.savefig('smote_drift_analysis.png', dpi=100)
    plt.show()
    print("\nCharts saved to smote_drift_analysis.png")
except:
    print("\nPlease install matplotlib and seaborn for visualization")