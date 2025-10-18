import numpy as np
import time
from sklearn.cluster import KMeans

from .anon_main import MIAdaptiveDPAnonymizer
from .dp_mechanism import apply_simple_noise


def mi_adaptive_dp_clustering(data, y, k, epsilon=1.0, mi_weight=0.8,
                             correlation_threshold=0.7, noise_type='laplace'):
    start_time = time.time()

    data = data.astype(np.float64)

    if data.shape[0] < 3:
        print("Dataset too small for MI-adaptive clustering")
        return data, y

    k = min(k, data.shape[0] // 2)
    k = max(k, 1)

    print(f"Applied MI-Adaptive DP (ε={epsilon}, MI weight={mi_weight})")

    clusters = find_clusters(data, k)

    indices = {}
    for i, cluster_id in enumerate(clusters):
        if cluster_id not in indices:
            indices[cluster_id] = []
        indices[cluster_id].append(i)

    epsilon_per_cluster = epsilon / len(indices)

    data_anonymized = None
    y_in_new_order = None

    for cluster_id in indices.keys():
        cluster_indices = indices[cluster_id]
        cluster_data = data[cluster_indices]
        cluster_y = y[cluster_indices]

        if cluster_data.shape[0] < 3:
            anonymized_cluster = apply_simple_noise(cluster_data, epsilon_per_cluster, noise_type)
        else:
            anonymizer = MIAdaptiveDPAnonymizer(epsilon=epsilon_per_cluster,
                                              mi_weight=mi_weight,
                                              correlation_threshold=correlation_threshold,
                                              noise_type=noise_type)

            task_type = 'classification' if len(np.unique(cluster_y)) < cluster_data.shape[0] * 0.5 else 'regression'
            anonymized_cluster = anonymizer.anonymize(cluster_data, cluster_y, task_type)

        if data_anonymized is None:
            data_anonymized = anonymized_cluster
            y_in_new_order = cluster_y
        else:
            data_anonymized = np.concatenate([data_anonymized, anonymized_cluster], axis=0)
            y_in_new_order = np.concatenate([y_in_new_order, cluster_y], axis=0)

    end_time = time.time()
    print(f"MI-Adaptive DP clustering time: {end_time - start_time:.4f} seconds")
    print(f"Privacy guarantee: ε={epsilon}-differential privacy with MI adaptation")

    return data_anonymized, y_in_new_order


def find_clusters(X, k, random_state=42):
    X = X.astype(np.float64)
    n_samples = X.shape[0]
    k = min(k, n_samples)

    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=random_state, init='k-means++')
    kmeans.fit(X)
    return kmeans.labels_