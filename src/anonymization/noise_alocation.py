import numpy as np


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