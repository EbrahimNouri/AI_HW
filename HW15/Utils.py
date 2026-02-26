from itertools import permutations
from sklearn.metrics import accuracy_score
import numpy as np


def map_clusters_to_labels(y_true, y_clusters):
    unique_labels = np.unique(y_true)
    n_clusters = len(unique_labels)
    best_acc = 0
    best_mapping = None
    best_mapped = None

    for perm in permutations(unique_labels):
        # perm یک تاپل مثل (0,1,2) یا (0,2,1) و ...
        mapped = [perm[label] for label in y_clusters]
        acc = accuracy_score(y_true, mapped)
        if acc > best_acc:
            best_acc = acc
            best_mapping = perm
            best_mapped = mapped

    return best_acc, best_mapping, best_mapped
