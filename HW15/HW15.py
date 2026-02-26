import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from torch.onnx.symbolic_opset9 import relu
from sklearn.metrics import accuracy_score, confusion_matrix

from Utils import map_clusters_to_labels

column_names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']
seed = 42
df_iris = pd.read_csv("../HW4/resource/iris.data", header=None, names=column_names)
species_mapping = {
    'Iris-setosa': 0,
    'Iris-versicolor': 1,
    'Iris-virginica': 2
}
y_true = df_iris['species'].map(species_mapping)
x = df_iris.drop('species', axis=1)
# print(df_iris.head())

scaler = StandardScaler()
X_scaled = scaler.fit_transform(x)

print(2 * ((80 * '=') + "\n"))

kmeans = KMeans(n_clusters=3, random_state=seed, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)
best_acc_kmeans, best_mapping_kmeans, _ = map_clusters_to_labels(y_true, y_kmeans)
print(f"K-Means accuracy score: {best_acc_kmeans:.3f}")
print("\nتوزیع خوشه‌های K-Means:")
print(pd.Series(y_kmeans).value_counts().sort_index())

print("\nمقایسه با برچسب‌های واقعی:")
comparison = pd.DataFrame({
    'KMeans_Cluster': y_kmeans,
    'True_Species': y_true
})
print(pd.crosstab(comparison['True_Species'],
                  comparison['KMeans_Cluster']))

print(2 * ((80 * '=') + "\n"))

hierarchical = AgglomerativeClustering(n_clusters=3)
y_hier = hierarchical.fit_predict(X_scaled)

best_acc_hier, best_mapping_hier, _ = map_clusters_to_labels(y_true, y_hier)
print(f"Hierarchical accuracy score: {best_acc_hier:.3f}")
print("\nتوزیع خوشه‌های Hierarchical:")
print(pd.Series(y_hier).value_counts().sort_index())

print("\nمقایسه با برچسب‌های واقعی:")
comparison = pd.DataFrame({
    'Hierarchical_Cluster': y_hier,
    'True_Species': y_true
})
print(pd.crosstab(comparison['True_Species'],
                  comparison['Hierarchical_Cluster']))

print(2 * ((80 * '=') + "\n"))

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_kmeans, test_size=0.2, random_state=seed)
mlp = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    max_iter=1000,
    random_state=seed,
    activation='relu',
    solver='adam'
)

mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"دقت MLP روی داده‌های آزمایشی: {acc:.3f}")

conf = confusion_matrix(y_test, y_pred)
print(f"دقت MLP روی داده‌های آزمایشی: {conf}")
