from itertools import permutations
from sklearn.metrics import accuracy_score
import numpy as np


def map_clusters_to_labels(y_true, y_cluster):
    """
    Find the best permutation of cluster labels that maximizes accuracy.
    Returns (best_accuracy, best_mapping, mapped_labels)
    """
    unique_true = np.unique(y_true)
    unique_cluster = np.unique(y_cluster)
    assert len(unique_true) == len(unique_cluster), "Number of clusters must match number of classes"

    best_acc = 0
    best_mapping = None
    for perm in permutations(unique_cluster):
        mapping = {orig: new for orig, new in zip(unique_cluster, perm)}
        y_mapped = np.array([mapping[x] for x in y_cluster])
        acc = accuracy_score(y_true, y_mapped)
        if acc > best_acc:
            best_acc = acc
            best_mapping = mapping
    # Return best accuracy, mapping dict, and mapped labels
    y_best = np.array([best_mapping[x] for x in y_cluster])
    return best_acc, best_mapping, y_best
