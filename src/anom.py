import numpy as np
import scipy.linalg as la
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import time
import warnings
warnings.filterwarnings('ignore')


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


class AdaptiveNoiseAllocator:
    def __init__(self, mi_weight=0.8, min_epsilon_ratio=0.05):
        self.mi_weight = mi_weight
        self.min_epsilon_ratio = min_epsilon_ratio
        
    def allocate_epsilon_budget(self, importance_scores, total_epsilon, redundancy_groups):
        n_features = len(importance_scores)
        epsilon_allocation = {}
        
        base_epsilon = total_epsilon * self.min_epsilon_ratio
        remaining_epsilon = total_epsilon - (base_epsilon * n_features)
        
        total_importance = sum(importance_scores.values())
        
        for feature_idx in importance_scores:
            if total_importance > 0:
                importance_ratio = importance_scores[feature_idx] / total_importance
                adaptive_epsilon = remaining_epsilon * (1.0 - importance_ratio * self.mi_weight)
            else:
                adaptive_epsilon = remaining_epsilon / n_features
                
            epsilon_allocation[feature_idx] = base_epsilon + adaptive_epsilon
        
        group_adjustments = {}
        for group in redundancy_groups:
            if len(group) > 1:
                avg_epsilon = np.mean([epsilon_allocation[idx] for idx in group])
                for idx in group:
                    group_adjustments[idx] = avg_epsilon
                    
        epsilon_allocation.update(group_adjustments)
        
        total_allocated = sum(epsilon_allocation.values())
        if total_allocated > 0:
            scale_factor = total_epsilon / total_allocated
            for key in epsilon_allocation:
                epsilon_allocation[key] *= scale_factor
                
        return epsilon_allocation


class CorrelationAwareDP:
    def __init__(self):
        pass
        
    def calculate_feature_sensitivity(self, X, feature_idx):
        feature_data = X[:, feature_idx]
        data_range = np.max(feature_data) - np.min(feature_data)
        sensitivity = data_range / len(feature_data)
        return max(sensitivity, 1e-8)
    
    def add_correlated_noise(self, X, redundancy_groups, epsilon_allocation, noise_type='laplace'):
        X_noisy = X.copy()
        
        for group in redundancy_groups:
            if len(group) == 1:
                feature_idx = group[0]
                sensitivity = self.calculate_feature_sensitivity(X, feature_idx)
                epsilon_feature = epsilon_allocation[feature_idx]
                
                if noise_type == 'laplace':
                    scale = sensitivity / epsilon_feature
                    noise = np.random.laplace(0, scale, X.shape[0])
                else:
                    sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / epsilon_feature
                    noise = np.random.normal(0, sigma, X.shape[0])
                    
                X_noisy[:, feature_idx] += noise
            else:
                group_sensitivities = [self.calculate_feature_sensitivity(X, idx) for idx in group]
                group_epsilons = [epsilon_allocation[idx] for idx in group]
                
                correlation_matrix = np.corrcoef([X[:, idx] for idx in group])
                
                for i, feature_idx in enumerate(group):
                    sensitivity = group_sensitivities[i]
                    epsilon_feature = group_epsilons[i]
                    
                    if noise_type == 'laplace':
                        scale = sensitivity / epsilon_feature
                        base_noise = np.random.laplace(0, scale, X.shape[0])
                    else:
                        sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / epsilon_feature
                        base_noise = np.random.normal(0, sigma, X.shape[0])
                    
                    correlation_factor = 0.0
                    for j, other_idx in enumerate(group):
                        if i != j and abs(correlation_matrix[i, j]) > 0.5:
                            correlation_factor += correlation_matrix[i, j] * 0.1
                    
                    adjusted_noise = base_noise * (1.0 + correlation_factor)
                    X_noisy[:, feature_idx] += adjusted_noise
                    
        return X_noisy


class MIAdaptiveDPAnonymizer:
    def __init__(self, epsilon=1.0, mi_weight=0.8, correlation_threshold=0.7, 
                 min_epsilon_ratio=0.05, noise_type='laplace'):
        self.epsilon = epsilon
        self.mi_weight = mi_weight
        self.correlation_threshold = correlation_threshold
        self.min_epsilon_ratio = min_epsilon_ratio
        self.noise_type = noise_type
        
        self.mi_analyzer = MutualInformationAnalyzer()
        self.noise_allocator = AdaptiveNoiseAllocator(mi_weight, min_epsilon_ratio)
        self.correlation_dp = CorrelationAwareDP()
        
    def anonymize(self, X, y=None, task_type='classification'):
        X = X.astype(np.float64)
        
        if X.shape[0] < 3 or X.shape[1] < 2:
            return self._apply_simple_noise(X)
        
        epsilon_stats = self.epsilon * 0.3
        epsilon_transform = self.epsilon * 0.4
        epsilon_features = self.epsilon * 0.3
        
        importance_scores = self.mi_analyzer.calculate_feature_importance(X, y, task_type)
        redundancy_groups, correlation_matrix = self.mi_analyzer.calculate_feature_redundancy(
            X, self.correlation_threshold)
        epsilon_allocation = self.noise_allocator.allocate_epsilon_budget(
            importance_scores, epsilon_features, redundancy_groups)
        
        X_dp_features = self.correlation_dp.add_correlated_noise(
            X, redundancy_groups, epsilon_allocation, self.noise_type)
        
        mean_noisy = self._dp_mean(X_dp_features, epsilon_stats * 0.4)
        cov_noisy = self._dp_covariance(X_dp_features, mean_noisy, epsilon_stats * 0.6)
        
        X_centered = X_dp_features - mean_noisy
        
        try:
            eigenvals, eigenvecs = la.eigh(cov_noisy)
            eigenvals = np.maximum(eigenvals, 1e-8)
            cov_fixed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T
            
            eigenvals, eigenvecs = la.eigh(cov_fixed)
            idx = np.argsort(eigenvals)[::-1]
            eigenvecs = eigenvecs[:, idx]
            
            X_transformed = X_centered @ eigenvecs
            
            rotation_matrix = self._generate_random_rotation(eigenvecs.shape[1])
            X_rotated = X_transformed @ rotation_matrix
            
            sensitivity_transform = np.max(np.linalg.norm(X_rotated, axis=1)) / X_rotated.shape[0]
            if epsilon_transform > 0:
                scale = sensitivity_transform / epsilon_transform
                if self.noise_type == 'laplace':
                    transform_noise = np.random.laplace(0, scale, X_rotated.shape)
                else:
                    sigma = sensitivity_transform * np.sqrt(2 * np.log(1.25)) / epsilon_transform
                    transform_noise = np.random.normal(0, sigma, X_rotated.shape)
                X_rotated += transform_noise
            
            X_reconstructed = X_rotated @ rotation_matrix.T @ eigenvecs.T
            X_final = X_reconstructed + mean_noisy
            
        except Exception:
            X_final = self._apply_simple_noise(X_dp_features)
            
        return X_final
    
    def _dp_mean(self, X, epsilon):
        if epsilon <= 0:
            return np.mean(X, axis=0)
            
        true_mean = np.mean(X, axis=0)
        sensitivity = (np.max(X, axis=0) - np.min(X, axis=0)) / X.shape[0]
        
        if self.noise_type == 'laplace':
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, true_mean.shape)
        else:
            sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / epsilon
            noise = np.random.normal(0, sigma, true_mean.shape)
            
        return true_mean + noise
    
    def _dp_covariance(self, X, mean_dp, epsilon):
        if epsilon <= 0:
            return np.cov(X.T)
            
        X_centered = X - mean_dp
        true_cov = np.cov(X_centered.T)
        
        max_range = np.max(np.max(X, axis=0) - np.min(X, axis=0))
        sensitivity = (max_range ** 2) / X.shape[0]
        
        if self.noise_type == 'laplace':
            scale = sensitivity / epsilon
            noise = np.random.laplace(0, scale, true_cov.shape)
        else:
            sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / epsilon
            noise = np.random.normal(0, sigma, true_cov.shape)
            
        noisy_cov = true_cov + noise
        return (noisy_cov + noisy_cov.T) / 2
    
    def _generate_random_rotation(self, n):
        random_matrix = np.random.randn(n, n)
        q, r = np.linalg.qr(random_matrix)
        if np.linalg.det(q) < 0:
            q[:, 0] = -q[:, 0]
        return q
    
    def _apply_simple_noise(self, X):
        sensitivity = np.max(np.linalg.norm(X, axis=1)) / X.shape[0] if X.shape[0] > 0 else 1.0
        
        if self.noise_type == 'laplace':
            scale = sensitivity / self.epsilon
            noise = np.random.laplace(0, scale, X.shape)
        else:
            sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / self.epsilon
            noise = np.random.normal(0, sigma, X.shape)
            
        return X + noise


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
            anonymizer = MIAdaptiveDPAnonymizer(epsilon=epsilon_per_cluster, 
                                              mi_weight=mi_weight,
                                              correlation_threshold=correlation_threshold,
                                              noise_type=noise_type)
            anonymized_cluster = anonymizer._apply_simple_noise(cluster_data)
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


def test_mi_adaptive_dp():
    np.random.seed(42)
    n_samples = 1000
    n_features = 8
    
    X = np.random.randn(n_samples, n_features)
    
    X[:, 1] = X[:, 0] * 2 + np.random.randn(n_samples) * 0.1
    X[:, 2] = X[:, 0] * 0.5 + X[:, 1] * 0.3 + np.random.randn(n_samples) * 0.2
    X[:, 3] = np.random.randn(n_samples) * 0.1
    X[:, 4] = X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 0.3
    
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    
    print("=== MI-ADAPTIVE DIFFERENTIAL PRIVACY TEST ===")
    print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Original correlations:")
    corr_matrix = np.corrcoef(X.T)
    for i in range(X.shape[1]):
        for j in range(i+1, X.shape[1]):
            if abs(corr_matrix[i, j]) > 0.3:
                print(f"  Feature {i} - Feature {j}: {corr_matrix[i, j]:.3f}")
    
    epsilons = [0.5, 1.0, 2.0]
    mi_weights = [0.2, 0.5, 0.8]
    
    for eps in epsilons:
        for mi_w in mi_weights:
            print(f"\n--- ε={eps}, MI weight={mi_w} ---")
            
            anonymizer = MIAdaptiveDPAnonymizer(epsilon=eps, mi_weight=mi_w)
            X_anon = anonymizer.anonymize(X, y)
            
            correlations = []
            for i in range(X.shape[1]):
                corr = np.corrcoef(X[:, i], X_anon[:, i])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
            
            if correlations:
                print(f"Avg correlation with original: {np.mean(correlations):.3f}")
                print(f"Std correlation: {np.std(correlations):.3f}")
            
            utility_score = np.mean([abs(np.corrcoef(X[:, i], y)[0, 1] - 
                                       np.corrcoef(X_anon[:, i], y)[0, 1]) 
                                   for i in range(X.shape[1]) 
                                   if not np.isnan(np.corrcoef(X[:, i], y)[0, 1])])
            print(f"Target correlation preservation: {1 - utility_score:.3f}")


if __name__ == "__main__":
    test_mi_adaptive_dp()