import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from anonymization.clustering import mi_adaptive_dp_clustering # Importação corrigida


def cross_validate_k_fold(X, y, anon_training, anon_test, model, model_name, n_clusters,
                         noise_factor=0.01, mi_weight=0.8, correlation_threshold=0.7,
                         noise_type='laplace'):
    kf = StratifiedKFold(n_splits=3)
    scaler = StandardScaler()

    accuracy, precision, recall, f1 = [], [], [], []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if anon_training:
           X_train, y_train = mi_adaptive_dp_clustering(
               X_train, y_train, n_clusters,
               epsilon=noise_factor, mi_weight=mi_weight,
               correlation_threshold=correlation_threshold,
               noise_type=noise_type
           )

        if anon_test:
            X_test, y_test = mi_adaptive_dp_clustering(
                X_test, y_test, n_clusters,
                epsilon=noise_factor, mi_weight=mi_weight,
                correlation_threshold=correlation_threshold,
                noise_type=noise_type
            )

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        accuracy.append(accuracy_score(y_test, y_pred))
        precision.append(precision_score(y_test, y_pred, average='weighted'))
        recall.append(recall_score(y_test, y_pred, average='weighted'))
        f1.append(f1_score(y_test, y_pred, average='weighted'))

    results = {
        'accuracy': np.array(accuracy),
        'precision': np.array(precision),
        'recall': np.array(recall),
        'f1_score': np.array(f1)
    }

    print(f"{model_name}, anon_train={anon_training}, anon_test={anon_test}, ε={noise_factor}, MI_w={mi_weight}")
    for k_val in results.keys(): # Changed 'k' to 'k_val' to avoid conflict with function parameter 'k'
        print(f"{k_val} ---> mean: {results[k_val].mean():.4f}, std: {results[k_val].std():.4f}")

    return [anon_training, anon_test, results['accuracy'].mean(), results['precision'].mean(), results['recall'].mean(), results['f1_score'].mean()]