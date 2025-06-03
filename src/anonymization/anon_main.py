# anonymization/anon_main.py
import numpy as np
import scipy.linalg as la
import warnings
warnings.filterwarnings('ignore')

from .mu import MutualInformationAnalyzer  # Importação corrigida
from .noise_alocation import AdaptiveNoiseAllocator  # Importação corrigida
from .dp_mechanism import CorrelationAwareDP, dp_mean, dp_covariance, generate_random_rotation, apply_simple_noise  # Importação corrigida


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
            return apply_simple_noise(X, self.epsilon, self.noise_type)

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

        mean_noisy = dp_mean(X_dp_features, epsilon_stats * 0.4, self.noise_type)
        cov_noisy = dp_covariance(X_dp_features, mean_noisy, epsilon_stats * 0.6, self.noise_type)

        X_centered = X_dp_features - mean_noisy

        try:
            eigenvals, eigenvecs = la.eigh(cov_noisy)
            eigenvals = np.maximum(eigenvals, 1e-8)
            cov_fixed = eigenvecs @ np.diag(eigenvals) @ eigenvecs.T

            eigenvals, eigenvecs = la.eigh(cov_fixed)
            idx = np.argsort(eigenvals)[::-1]
            eigenvecs = eigenvecs[:, idx]

            X_transformed = X_centered @ eigenvecs

            rotation_matrix = generate_random_rotation(eigenvecs.shape[1])
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
            X_final = apply_simple_noise(X_dp_features, self.epsilon, self.noise_type)

        return X_final