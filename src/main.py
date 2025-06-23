import numpy as np
import pandas as pd
import os
import sys
import optuna
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

from file_utils import get_file, list_available_datasets
from ml import cross_validate_k_fold
from anonymization.anon_main import MIAdaptiveDPAnonymizer


def feature_selection(X, y, method, k=None):
    if method == 'chi2':
        X = X.astype(np.float64)
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_scaled, y)
        X_new = selector.transform(X_scaled)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    elif method == 'extra_trees':
        X = X.astype(np.float64)
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        selector = SelectFromModel(model, prefit=True)
        X_new = selector.transform(X)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    else:
        raise ValueError(f"Feature selection method not supported: {method}")


def objective_knn(trial, X, y, anon_training, anon_test, n_clusters, noise_factor, mi_weight, correlation_threshold, noise_type):
    n_neighbors = trial.suggest_int('n_neighbors', 3, 15)
    weights = trial.suggest_categorical('weights', ['uniform', 'distance'])
    p = trial.suggest_int('p', 1, 2)

    model = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, p=p)
    results = cross_validate_k_fold(X, y, anon_training, anon_test, model, 'KNN', n_clusters,
                                    noise_factor, mi_weight, correlation_threshold, noise_type, verbose=False)
    return results[2]

def objective_random_forest(trial, X, y, anon_training, anon_test, n_clusters, noise_factor, mi_weight, correlation_threshold, noise_type):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    min_samples_leaf = trial.suggest_int('min_samples_leaf', 1, 5)
    criterion = trial.suggest_categorical('criterion', ['gini', 'entropy'])

    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                   min_samples_leaf=min_samples_leaf, criterion=criterion,
                                   random_state=42, n_jobs=-1)
    results = cross_validate_k_fold(X, y, anon_training, anon_test, model, 'Random Forest', n_clusters,
                                    noise_factor, mi_weight, correlation_threshold, noise_type, verbose=False)
    return results[2]

def objective_gaussian_nb(trial, X, y, anon_training, anon_test, n_clusters, noise_factor, mi_weight, correlation_threshold, noise_type):
    var_smoothing = trial.suggest_float('var_smoothing', 1e-09, 1e-02, log=True)

    model = GaussianNB(var_smoothing=var_smoothing)
    results = cross_validate_k_fold(X, y, anon_training, anon_test, model, 'GaussianNB', n_clusters,
                                    noise_factor, mi_weight, correlation_threshold, noise_type, verbose=False)
    return results[2]

def objective_mlp(trial, X, y, anon_training, anon_test, n_clusters, noise_factor, mi_weight, correlation_threshold, noise_type):
    n_layers = trial.suggest_int('n_layers', 1, 3)
    hidden_layer_sizes = tuple(trial.suggest_int(f'n_units_l{i}', 30, 150) for i in range(n_layers))
    activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
    solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
    alpha = trial.suggest_float('alpha', 1e-05, 1e-02, log=True)
    learning_rate_init = trial.suggest_float('learning_rate_init', 1e-04, 1e-02, log=True)

    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        learning_rate='adaptive',
        learning_rate_init=learning_rate_init,
        max_iter=500,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
    results = cross_validate_k_fold(X, y, anon_training, anon_test, model, 'Multilayer Perceptron', n_clusters,
                                    noise_factor, mi_weight, correlation_threshold, noise_type, verbose=False)
    return results[2]

def objective_adaboost(trial, X, y, anon_training, anon_test, n_clusters, noise_factor, mi_weight, correlation_threshold, noise_type):
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    learning_rate = trial.suggest_float('learning_rate', 0.01, 1.0, log=True)

    model = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    results = cross_validate_k_fold(X, y, anon_training, anon_test, model, 'AdaBoost', n_clusters,
                                    noise_factor, mi_weight, correlation_threshold, noise_type, verbose=False)
    return results[2]

def objective_logistic_regression(trial, X, y, anon_training, anon_test, n_clusters, noise_factor, mi_weight, correlation_threshold, noise_type):
    C = trial.suggest_float('C', 1e-03, 10.0, log=True)
    solver = trial.suggest_categorical('solver', ['lbfgs', 'liblinear'])
    
    model = LogisticRegression(max_iter=1000, C=C, solver=solver, multi_class='auto', random_state=42)
    results = cross_validate_k_fold(X, y, anon_training, anon_test, model, 'Logistic Regression', n_clusters,
                                    noise_factor, mi_weight, correlation_threshold, noise_type, verbose=False)
    return results[2]


model_objectives = {
    'KNN': objective_knn,
    'Random Forest': objective_random_forest,
    'GaussianNB': objective_gaussian_nb,
    'Multilayer Perceptron': objective_mlp,
    'AdaBoost': objective_adaboost,
    'Logistic Regression': objective_logistic_regression
}


def get_result(model_name, X, y, n_clusters, feature_method, k,
               noise_factor=0.01, mi_weight=0.8, correlation_threshold=0.7, noise_type='laplace', n_trials=20):
    bol = [True, False]
    results_columns = ['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score',
                       'anon_train_time', 'anon_test_time', 'model_train_time', 'best_params']
    results_df = pd.DataFrame(columns=results_columns)
    selected_features_all = []

    for anon_train in bol:
        for anon_test in bol:
            X_new, selected_features_idx = feature_selection(X, y, feature_method, k)
            
            selected_features_all.append({
                'anonymized_train': anon_train,
                'anonymized_test': anon_test,
                'model': model_name,
                'feature_method': feature_method,
                'num_features': k,
                'selected_features_idx': selected_features_idx.tolist()
            })

            objective = model_objectives.get(model_name)
            if objective is None:
                raise ValueError(f"Objective function not defined for model: {model_name}")

            print(f"Starting Optuna optimization for {model_name} with anon_train={anon_train}, anon_test={anon_test}, k={k}")

            study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42))
            study.optimize(lambda trial: objective(trial, X_new, y, anon_train, anon_test, n_clusters,
                                                    noise_factor, mi_weight, correlation_threshold, noise_type),
                           n_trials=n_trials, show_progress_bar=True)

            best_trial = study.best_trial
            best_accuracy = best_trial.value
            best_params = best_trial.params
            
            if model_name == 'KNN':
                final_model = KNeighborsClassifier(**best_params)
            elif model_name == 'Random Forest':
                final_model = RandomForestClassifier(random_state=42, n_jobs=-1, **best_params)
            elif model_name == 'GaussianNB':
                final_model = GaussianNB(**best_params)
            elif model_name == 'Multilayer Perceptron':
                if 'n_layers' in best_params:
                    hidden_layer_sizes_final = tuple(best_params[f'n_units_l{i}'] for i in range(best_params['n_layers']))
                    mlp_params = {k: v for k, v in best_params.items() if not k.startswith('n_units_l') and k != 'n_layers'}
                    final_model = MLPClassifier(hidden_layer_sizes=hidden_layer_sizes_final, random_state=42, **mlp_params)
                else:
                    final_model = MLPClassifier(random_state=42, **best_params)
            elif model_name == 'AdaBoost':
                final_model = AdaBoostClassifier(random_state=42, **best_params)
            elif model_name == 'Logistic Regression':
                final_model = LogisticRegression(max_iter=1000, multi_class='auto', random_state=42, **best_params)
            else:
                raise ValueError("Model not recognized for final instantiation after Optuna.")


            cross_val_results = cross_validate_k_fold(
                X_new, y, anon_train, anon_test, final_model, model_name, n_clusters,
                noise_factor, mi_weight, correlation_threshold, noise_type, verbose=True
            )
            
            new_row = pd.DataFrame([[
                model_name,
                anon_train,
                anon_test,
                cross_val_results[2],
                cross_val_results[3],
                cross_val_results[4],
                cross_val_results[5],
                cross_val_results[6], # anon_train_time
                cross_val_results[7], # anon_test_time
                cross_val_results[8], # model_train_time
                str(best_params)
            ]], columns=results_columns)
            results_df = pd.concat([results_df, new_row], ignore_index=True)

    return results_df, selected_features_all


def experiment(X, y, feature_method, k, noise_factor=0.01, mi_weight=0.8,
               correlation_threshold=0.7, noise_type='laplace', n_trials=20):
    all_results = pd.DataFrame(columns=['model', 'anonymized_train', 'anonymized_test',
                                       'accuracy', 'precision', 'recall', 'f1_score',
                                       'anon_train_time', 'anon_test_time', 'model_train_time',
                                       'selected_features', 'feature_method', 'num_features', 'best_params'])
    models_to_run = [
        'KNN',
        'Random Forest',
        'GaussianNB',
        'Multilayer Perceptron',
        'AdaBoost',
        'Logistic Regression'
    ]

    for model_name in models_to_run:
        results, selected_features = get_result(
            model_name, X, y, 3, feature_method, k,
            noise_factor, mi_weight, correlation_threshold, noise_type, n_trials
        )
        best_results = find_best_results(results, selected_features, feature_method, k)
        all_results = pd.concat([all_results, best_results], ignore_index=True)

    return all_results

def find_best_results(results, selected_features, feature_method, k):
    scenarios = [(True, True), (True, False), (False, True), (False, False)]
    best_results_list = []

    for model_name in results['model'].unique():
        for scenario in scenarios:
            anonymized_train, anonymized_test = scenario
            model_results = results[
                (results['model'] == model_name) &
                (results['anonymized_train'] == anonymized_train) &
                (results['anonymized_test'] == anonymized_test)
            ]
            if not model_results.empty:
                best_result_row = model_results.loc[model_results['accuracy'].idxmax()].copy()
                
                selected_feature_info = [s for s in selected_features if
                                          s['anonymized_train'] == anonymized_train and
                                          s['anonymized_test'] == anonymized_test and
                                          s['model'] == model_name and
                                          s['feature_method'] == feature_method and
                                          s['num_features'] == k]
                if selected_feature_info:
                    best_result_row['selected_features'] = selected_feature_info[0]['selected_features_idx']
                else:
                    best_result_row['selected_features'] = []
                
                best_result_row['feature_method'] = feature_method
                best_result_row['num_features'] = k
                best_results_list.append(best_result_row)

    return pd.DataFrame(best_results_list)


def run_mi_adaptive_experiments(X, y, dataset_name, feature_method,
                               noise_factor=0.01, mi_weight=0.8,
                               correlation_threshold=0.7, noise_type='laplace', n_trials=20):
    all_best_results = []

    num_features = X.shape[1]
    print(f"Running {feature_method} with MI-Adaptive DP")
    print(f"Features: {num_features}, ε: {noise_factor}, MI weight: {mi_weight}")
    print(f"Correlation threshold: {correlation_threshold}, Noise: {noise_type}")

    max_features_to_test = min(num_features, 20)
    
    for k_val in range(2, max_features_to_test + 1):
        print(f"\n--- Testing with {k_val} selected features ---")
        best_results = experiment(X, y, feature_method, k_val, noise_factor,
                                mi_weight, correlation_threshold, noise_type, n_trials)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)

    os.makedirs('results', exist_ok=True)

    filename = f'mi_adaptive_{feature_method}_{dataset_name}_eps_{noise_factor:.2f}_miw_{mi_weight:.2f}_{noise_type}_optuna.csv'
    absolute_path = os.path.join(os.getcwd(), 'results', filename)
    final_best_results_df.to_csv(absolute_path, index=False)
    print(f"\nMI-Adaptive results saved at: {absolute_path}")
    print(final_best_results_df.head())


def Chi2(X, y, dataset_name, noise_factor=0.01, mi_weight=0.8,
         correlation_threshold=0.7, noise_type='laplace', n_trials=20):
    run_mi_adaptive_experiments(X, y, dataset_name, 'chi2', noise_factor,
                               mi_weight, correlation_threshold, noise_type, n_trials)

def ExtraTree(X, y, dataset_name, noise_factor=0.01, mi_weight=0.8,
              correlation_threshold=0.7, noise_type='laplace', n_trials=20):
    run_mi_adaptive_experiments(X, y, dataset_name, 'extra_trees', noise_factor,
                               mi_weight, correlation_threshold, noise_type, n_trials)


def main():
    np.random.seed(7)

    dataset_name = None
    noise_factor = 1.0
    mi_weight = 0.8
    correlation_threshold = 0.7
    noise_type = 'laplace'
    n_trials = 20

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
        elif arg.startswith('--n_trials='):
            try:
                n_trials = int(arg.split('=')[1])
                print(f"Using n_trials for Optuna: {n_trials}")
            except (ValueError, IndexError):
                print(f"Invalid n_trials format. Using default: {n_trials}")
        elif i == 1 and not arg.startswith('--'):
            dataset_name = arg

    if dataset_name is None:
        print("No dataset name provided. Listing available datasets.")
        list_available_datasets()
        sys.exit(1)

    dataset, label_column = get_file(dataset_name)

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
    print(f"  Optuna Trials per scenario: {n_trials}")

    Chi2(X, y, dataset_name, noise_factor, mi_weight, correlation_threshold, noise_type, n_trials)
    ExtraTree(X, y, dataset_name, noise_factor, mi_weight, correlation_threshold, noise_type, n_trials)
    

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].lower() in ['-h', '--help', 'help']:
        print("\nUsage: python main.py [dataset_name] [options]")
        print("\nOptions:")
        print("  dataset_name                   Name of the dataset to use")
        print("  --epsilon=VALUE               Epsilon for differential privacy (default: 1.0)")
        print("  --mi_weight=VALUE             MI weight for adaptive allocation (default: 0.8)")
        print("  --correlation_threshold=VALUE Correlation threshold (default: 0.7)")
        print("  --noise_type=TYPE             Noise type: laplace or gaussian (default: laplace)")
        print("  --n_trials=VALUE              Number of Optuna trials per model/scenario (default: 20)")
        print("\nAvailable datasets:")
        list_available_datasets()
        sys.exit(0)

    main()