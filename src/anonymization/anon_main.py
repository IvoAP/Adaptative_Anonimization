import numpy as np
import scipy.linalg as la
import warnings
warnings.filterwarnings('ignore')

from .mu import MutualInformationAnalyzer
from .noise_alocation import AdaptiveNoiseAllocator
from .dp_mechanism import CorrelationAwareDP, apply_simple_noise

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

        epsilon_features = self.epsilon

        importance_scores = self.mi_analyzer.calculate_feature_importance(X, y, task_type)
        redundancy_groups, correlation_matrix = self.mi_analyzer.calculate_feature_redundancy(
            X, self.correlation_threshold)

        epsilon_allocation = self.noise_allocator.allocate_epsilon_budget(
            importance_scores, epsilon_features, redundancy_groups)

        X_final = self.correlation_dp.add_correlated_noise(
            X, redundancy_groups, epsilon_allocation, self.noise_type)

        return X_final