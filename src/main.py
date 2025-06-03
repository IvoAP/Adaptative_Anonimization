import numpy as np
import pandas as pd
import os
import sys
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2,
                                       f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from file_utils import get_file, list_available_datasets
from ml import cross_validate_k_fold
from anonymization.anon_main import MIAdaptiveDPAnonymizer # Importação corrigida


def feature_selection(X, y, method, k=None):
    if method == 'chi2':
        X = X.astype(np.float64)

        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)

        selector = SelectKBest(chi2, k=k)
        X_new = selector.fit_transform(X_scaled, y)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    elif method == 'extra_trees':
        X = X.astype(np.float64)

        model = ExtraTreesClassifier(n_estimators=100)
        model.fit(X, y)
        selector = SelectFromModel(model, prefit=True)
        X_new = selector.transform(X)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    else:
        raise ValueError(f"Feature selection method not supported: {method}")


def get_result(model, X, y, model_name, n_clusters, feature_method, k,
               noise_factor=0.01, mi_weight=0.8, correlation_threshold=0.7, noise_type='laplace'):
    bol = [True, False]
    results_columns = ['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score']
    results = pd.DataFrame(columns=results_columns)
    selected_features_all = []

    for i in range(0, 2):
        for j in range(0, 2):
            X_new, selected_features_idx = feature_selection(X, y, feature_method, k)
            selected_features_all.append({
                'anonymized_train': bol[i],
                'anonymized_test': bol[j],
                'model': model_name,
                'feature_method': feature_method,
                'num_features': k,
                'selected_features_idx': selected_features_idx.tolist()
            })
            cross_val_results = cross_validate_k_fold(
                X_new, y, bol[i], bol[j], model, model_name, n_clusters,
                noise_factor, mi_weight, correlation_threshold, noise_type
            )
            new_df = pd.DataFrame([[
                model_name,
                bol[i],
                bol[j],
                cross_val_results[2],
                cross_val_results[3],
                cross_val_results[4],
                cross_val_results[5]
            ]], columns=results_columns)
            results = pd.concat([results, new_df], ignore_index=True)

    return results, selected_features_all

def experiment(X, y, feature_method, k, noise_factor=0.01, mi_weight=0.8,
               correlation_threshold=0.7, noise_type='laplace'):
    all_results = pd.DataFrame(columns=['model', 'anonymized_train', 'anonymized_test',
                                       'accuracy', 'precision', 'recall', 'f1_score',
                                       'selected_features', 'feature_method', 'num_features'])
    models = [
        (KNeighborsClassifier(n_neighbors=5), 'KNN'),
        (RandomForestClassifier(n_estimators=100), 'Random Forest'),
        (GaussianNB(var_smoothing=1e-02), 'GaussianNB'),
        (MLPClassifier(
            hidden_layer_sizes=(100, 50),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            early_stopping=True,
            validation_fraction=0.1
        ), 'Multilayer Perceptron'),
        (AdaBoostClassifier(n_estimators=100, learning_rate=1.0), 'AdaBoost'),
        (LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='auto'), 'Logistic Regression')
    ]

    for model, model_name in models:
        results, selected_features = get_result(
            model, X, y, model_name, 3, feature_method, k,
            noise_factor, mi_weight, correlation_threshold, noise_type
        )
        best_results = find_best_results(results, selected_features, feature_method, k)
        all_results = pd.concat([all_results, best_results], ignore_index=True)

    return all_results

def find_best_results(results, selected_features, feature_method, k):
    scenarios = [(True, True), (True, False), (False, True), (False, False)]
    best_results = []

    for model_name in results['model'].unique():
        for scenario in scenarios:
            anonymized_train, anonymized_test = scenario
            model_results = results[
                (results['model'] == model_name) &
                (results['anonymized_train'] == anonymized_train) &
                (results['anonymized_test'] == anonymized_test)
            ]
            if not model_results.empty:
                best_result = model_results.loc[model_results['accuracy'].idxmax()]
                selected_feature_info = [s for s in selected_features if
                                          s['anonymized_train'] == anonymized_train and
                                          s['anonymized_test'] == anonymized_test and
                                          s['model'] == model_name and
                                          s['feature_method'] == feature_method and
                                          s['num_features'] == k]
                if selected_feature_info:
                    best_result.loc['selected_features'] = selected_feature_info[0]['selected_features_idx']
                best_result.loc['feature_method'] = feature_method
                best_result.loc['num_features'] = k
                best_results.append(best_result)

    return pd.DataFrame(best_results)

def run_mi_adaptive_experiments(X, y, dataset_name, feature_method,
                               noise_factor=0.01, mi_weight=0.8,
                               correlation_threshold=0.7, noise_type='laplace'):
    all_best_results = []

    num_features = X.shape[1]
    print(f"Running {feature_method} with MI-Adaptive DP")
    print(f"Features: {num_features}, ε: {noise_factor}, MI weight: {mi_weight}")
    print(f"Correlation threshold: {correlation_threshold}, Noise: {noise_type}")

    for i in range(2, num_features, 1):
        best_results = experiment(X, y, feature_method, i, noise_factor,
                                mi_weight, correlation_threshold, noise_type)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)

    os.makedirs('results', exist_ok=True)

    filename = f'mi_adaptive_{feature_method}_{dataset_name}_eps_{noise_factor:.2f}_miw_{mi_weight:.2f}_{noise_type}.csv'
    absolute_path = os.path.join(os.getcwd(), 'results', filename)
    final_best_results_df.to_csv(absolute_path, index=False)
    print(f"MI-Adaptive results saved at: {absolute_path}")
    print(final_best_results_df.head())

def Chi2(X, y, dataset_name, noise_factor=0.01, mi_weight=0.8,
         correlation_threshold=0.7, noise_type='laplace'):
    run_mi_adaptive_experiments(X, y, dataset_name, 'chi2', noise_factor,
                               mi_weight, correlation_threshold, noise_type)

def ExtraTree(X, y, dataset_name, noise_factor=0.01, mi_weight=0.8,
              correlation_threshold=0.7, noise_type='laplace'):
    run_mi_adaptive_experiments(X, y, dataset_name, 'extra_trees', noise_factor,
                               mi_weight, correlation_threshold, noise_type)

def MutualInfo(X, y, dataset_name, noise_factor=0.01, mi_weight=0.8,
               correlation_threshold=0.7, noise_type='laplace'):
    run_mi_adaptive_experiments(X, y, dataset_name, 'mutual_info', noise_factor,
                               mi_weight, correlation_threshold, noise_type)

def main():
    np.random.seed(7)

    dataset_name = None
    noise_factor = 1.0
    mi_weight = 0.8
    correlation_threshold = 0.7
    noise_type = 'laplace'

    for i, arg in enumerate(sys.argv[1:], 1):
        if arg.startswith('--epsilon='):
            try:
                noise_factor = float(arg.split('=')[1])
                print(f"Using epsilon: {noise_factor}")
            except (ValueError, IndexError):
                print(f"Invalid epsilon format. Using default: {noise_factor}")
        elif arg.startswith('--mi_weight='):
            try:
                mi_weight = float(arg.split('=')[1])
                print(f"Using MI weight: {mi_weight}")
            except (ValueError, IndexError):
                print(f"Invalid MI weight format. Using default: {mi_weight}")
        elif arg.startswith('--correlation_threshold='):
            try:
                correlation_threshold = float(arg.split('=')[1])
                print(f"Using correlation threshold: {correlation_threshold}")
            except (ValueError, IndexError):
                print(f"Invalid correlation threshold format. Using default: {correlation_threshold}")
        elif arg.startswith('--noise_type='):
            noise_type = arg.split('=')[1]
            if noise_type not in ['laplace', 'gaussian']:
                print(f"Invalid noise type. Using default: laplace")
                noise_type = 'laplace'
            else:
                print(f"Using noise type: {noise_type}")
        elif i == 1 and not arg.startswith('--'):
            dataset_name = arg

    dataset, label_column = get_file(dataset_name)

    dataset_name = dataset_name or "cahousing"

    print(f"Total columns in dataset: {len(dataset.columns)}")
    print(f"Using '{label_column}' as target variable")

    y = np.array(dataset[label_column])
    dataset = dataset.drop(columns=[label_column])
    X = np.array(dataset)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"Number of unique classes: {len(np.unique(y))}")

    print(f"MI-Adaptive DP parameters:")
    print(f"  Epsilon: {noise_factor}")
    print(f"  MI Weight: {mi_weight}")
    print(f"  Correlation Threshold: {correlation_threshold}")
    print(f"  Noise Type: {noise_type}")

    Chi2(X, y, dataset_name, noise_factor, mi_weight, correlation_threshold, noise_type)
    ExtraTree(X, y, dataset_name, noise_factor, mi_weight, correlation_threshold, noise_type)
    MutualInfo(X, y, dataset_name, noise_factor, mi_weight, correlation_threshold, noise_type)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['-h', '--help', 'help']:
        print("\nUsage: python main.py [dataset_name] [options]")
        print("\nOptions:")
        print("  dataset_name                   Name of the dataset to use")
        print("  --epsilon=VALUE               Epsilon for differential privacy (default: 1.0)")
        print("  --mi_weight=VALUE             MI weight for adaptive allocation (default: 0.8)")
        print("  --correlation_threshold=VALUE Correlation threshold (default: 0.7)")
        print("  --noise_type=TYPE             Noise type: laplace or gaussian (default: laplace)")
        print("\nAvailable datasets:")
        list_available_datasets()
        sys.exit(0)

    main()