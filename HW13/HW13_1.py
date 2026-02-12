import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, \
    roc_curve

np.random.seed(42)

X0 = np.random.normal(0, 1, (900, 1))
X1 = np.random.normal(2, 1, (100, 1))

X = np.vstack([X0, X1])
y = np.array([0] * 900 + [1] * 100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)

model = LogisticRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
y_prob = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.1, 1.0, 0.1)
precisions, recalls = [], []

for t in thresholds:
    y_t = (y_prob >= t).astype(int)
    precisions.append(precision_score(y_test, y_t))
    recalls.append(recall_score(y_test, y_t))

plt.plot(recalls, precisions, marker='o')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Tradeoff")
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_prob)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], '--')
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.title("ROC Curve")
plt.show()

print("AUC:", roc_auc_score(y_test, y_prob))
