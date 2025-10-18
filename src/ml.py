import numpy as np
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from anonymization.clustering import mi_adaptive_dp_clustering
import time


def cross_validate_k_fold(X, y, anon_training, anon_test, model_instance, model_name, n_clusters,
                         noise_factor=0.01, mi_weight=0.8, correlation_threshold=0.7,
                         noise_type='laplace', verbose=False):
    kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    scaler = StandardScaler()

    accuracy, precision, recall, f1 = [], [], [], []
    anon_train_times, anon_test_times, model_train_times = [], [], []

    for fold_idx, (train_index, test_index) in enumerate(kf.split(X, y)):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        anon_start_time_train = time.time()
        if anon_training:
           X_train, y_train = mi_adaptive_dp_clustering(
               X_train, y_train, n_clusters,
               epsilon=noise_factor, mi_weight=mi_weight,
               correlation_threshold=correlation_threshold,
               noise_type=noise_type
           )
        anon_end_time_train = time.time()
        anon_train_times.append(anon_end_time_train - anon_start_time_train)

        anon_start_time_test = time.time()
        if anon_test:
            X_test, y_test = mi_adaptive_dp_clustering(
                X_test, y_test, n_clusters,
                epsilon=noise_factor, mi_weight=mi_weight,
                correlation_threshold=correlation_threshold,
                noise_type=noise_type
            )
        anon_end_time_test = time.time()
        anon_test_times.append(anon_end_time_test - anon_start_time_test)

        if X_train.shape[0] == 0 or X_test.shape[0] == 0:
            if verbose:
                print(f"Fold {fold_idx}: Skipped due to empty (train/test) data after anonymization.")
            accuracy.append(0.0)
            precision.append(0.0)
            recall.append(0.0)
            f1.append(0.0)
            model_train_times.append(0.0)
            continue

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        try:
            model_train_start_time = time.time()
            model_instance.fit(X_train, y_train)
            model_train_end_time = time.time()
            model_train_times.append(model_train_end_time - model_train_start_time)

            y_pred = model_instance.predict(X_test)

            accuracy.append(accuracy_score(y_test, y_pred))
            precision.append(precision_score(y_test, y_pred, average='weighted', zero_division=0))
            recall.append(recall_score(y_test, y_pred, average='weighted', zero_division=0))
            f1.append(f1_score(y_test, y_pred, average='weighted', zero_division=0))
        except Exception as e:
            if verbose:
                print(f"Fold {fold_idx}: Model training/prediction failed: {e}")
            accuracy.append(0.0)
            precision.append(0.0)
            recall.append(0.0)
            f1.append(0.0)
            model_train_times.append(0.0)

    if not accuracy:
        return [anon_training, anon_test, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    mean_accuracy = np.mean(accuracy)
    mean_precision = np.mean(precision)
    mean_recall = np.mean(recall)
    mean_f1 = np.mean(f1)
    mean_anon_train_time = np.mean(anon_train_times)
    mean_anon_test_time = np.mean(anon_test_times)
    mean_model_train_time = np.mean(model_train_times)


    if verbose:
        print(f"{model_name}, anon_train={anon_training}, anon_test={anon_test}, ε={noise_factor}, MI_w={mi_weight}")
        print(f"accuracy ---> mean: {mean_accuracy:.4f}, std: {np.std(accuracy):.4f}")
        print(f"precision ---> mean: {mean_precision:.4f}, std: {np.std(precision):.4f}")
        print(f"recall ---> mean: {mean_recall:.4f}, std: {np.std(recall):.4f}")
        print(f"f1_score ---> mean: {mean_f1:.4f}, std: {np.std(f1):.4f}")
        print(f"anon_train_time ---> mean: {mean_anon_train_time:.4f}, std: {np.std(anon_train_times):.4f}")
        print(f"anon_test_time ---> mean: {mean_anon_test_time:.4f}, std: {np.std(anon_test_times):.4f}")
        print(f"model_train_time ---> mean: {mean_model_train_time:.4f}, std: {np.std(model_train_times):.4f}")


    return [anon_training, anon_test, mean_accuracy, mean_precision, mean_recall,
            mean_f1, mean_anon_train_time, mean_anon_test_time, mean_model_train_time]