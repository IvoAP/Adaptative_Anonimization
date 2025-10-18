import numpy as np
import scipy.linalg as la


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

                if epsilon_feature <= 0:
                    continue

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

                group_data = X[:, group]
                correlation_matrix = np.corrcoef(group_data.T + np.random.rand(*group_data.T.shape) * 1e-9)

                for i, feature_idx in enumerate(group):
                    sensitivity = group_sensitivities[i]
                    epsilon_feature = group_epsilons[i]

                    if epsilon_feature <= 0:
                        continue

                    if noise_type == 'laplace':
                        scale = sensitivity / epsilon_feature
                        base_noise = np.random.laplace(0, scale, X.shape[0])
                    else:
                        sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / epsilon_feature
                        base_noise = np.random.normal(0, sigma, X.shape[0])

                    correlation_factor = 0.0
                    for j, other_idx in enumerate(group):
                        if i != j and abs(correlation_matrix[i, j]) > 0.5:
                            correlation_factor += abs(correlation_matrix[i, j]) * 0.1

                    adjusted_noise = base_noise * (1.0 + correlation_factor)
                    X_noisy[:, feature_idx] += adjusted_noise

        return X_noisy


def apply_simple_noise(X, epsilon, noise_type):
    if X.shape[0] == 0 or epsilon <= 0:
        return X

    sensitivity = np.max(np.linalg.norm(X, axis=1)) / X.shape[0] if X.shape[0] > 0 else 1.0
    sensitivity = max(sensitivity, 1e-8)

    if noise_type == 'laplace':
        scale = sensitivity / epsilon
        noise = np.random.laplace(0, scale, X.shape)
    else:
        sigma = sensitivity * np.sqrt(2 * np.log(1.25)) / epsilon
        noise = np.random.normal(0, sigma, X.shape)

    return X + noise