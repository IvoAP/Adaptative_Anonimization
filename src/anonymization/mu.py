import numpy as np
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif


class MutualInformationAnalyzer:
    def __init__(self, random_state=42):
        self.random_state = random_state

    def calculate_feature_importance(self, X, y=None, task_type='classification'):
        importance_scores = {}
        n_features = X.shape[1]

        if y is not None:
            if task_type == 'classification':
                mi_func = mutual_info_classif
            else:
                mi_func = mutual_info_regression

            mi_scores = mi_func(X, y, random_state=self.random_state)

            for i in range(n_features):
                importance_scores[i] = mi_scores[i]
        else:
            for i in range(n_features):
                importance_scores[i] = np.var(X[:, i])

        max_score = max(importance_scores.values()) if importance_scores.values() else 1.0
        if max_score > 0:
            for key in importance_scores:
                importance_scores[key] = importance_scores[key] / max_score
        else:
            for key in importance_scores:
                importance_scores[key] = 1.0 / n_features

        return importance_scores

    def calculate_feature_redundancy(self, X, correlation_threshold=0.7):
        n_features = X.shape[1]
        correlation_matrix = np.corrcoef(X.T)

        redundancy_groups = []
        processed = set()

        for i in range(n_features):
            if i in processed:
                continue

            group = [i]
            for j in range(i + 1, n_features):
                if j not in processed and abs(correlation_matrix[i, j]) > correlation_threshold:
                    group.append(j)
                    processed.add(j)
            redundancy_groups.append(group)
            processed.add(i)

        return redundancy_groups, correlation_matrix