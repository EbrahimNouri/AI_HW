from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, recall_score, balanced_accuracy_score, roc_auc_score

np.random.seed(42)

X0 = np.random.normal(0, 1, (900, 1))
X1 = np.random.normal(2, 1, (100, 1))

X = np.vstack([X0, X1])
y = np.array([0]*900 + [1]*100)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, stratify=y, random_state=42
)
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("Balanced Acc:", balanced_accuracy_score(y_test, y_pred))
print("AUC:", roc_auc_score(y_test, model.predict_proba(X_test)[:,1]))
y_prob = model.predict_proba(X_test)[:,1]
y_custom = (y_prob >= 0.3).astype(int)

print(confusion_matrix(y_test, y_custom))
X_test_drift = X_test + 1.0
y_pred_drift = model.predict(X_test_drift)

print("Recall after drift:", recall_score(y_test, y_pred_drift))
