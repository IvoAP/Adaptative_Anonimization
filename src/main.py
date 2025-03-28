import numpy as np
import pandas as pd
import os
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.feature_selection import (SelectFromModel, SelectKBest, chi2,
                                       f_classif, mutual_info_classif)
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import MinMaxScaler

from file_utils import get_file
from ml import cross_validate_k_fold


# Function that select the technique of feature selection
def feature_selection(X, y, method, k=None):
    if method == 'chi2':
        # For chi2, we need to ensure all values are non-negative
        # First convert to float64 to avoid type issues
        X = X.astype(np.float64)
        
        # Apply MinMaxScaler to ensure all values are non-negative
        scaler = MinMaxScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Now apply chi2
        selector = SelectKBest(chi2, k=k)
        X_new = selector.fit_transform(X_scaled, y)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    elif method == 'extra_trees':
        # Convert to float64 to avoid type issues
        X = X.astype(np.float64)
        
        model = ExtraTreesClassifier(n_estimators=100)
        model.fit(X, y)
        selector = SelectFromModel(model, prefit=True)
        X_new = selector.transform(X)
        selected_features_idx = selector.get_support(indices=True)
        return X_new, selected_features_idx
    else:
        raise ValueError(f"Método de seleção de características não suportado: {method}")


# Function to  get the results
def get_result(model, X, y, model_name, n_clusters, feature_method, k):
    bol = [True, False]
    results_columns = ['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score']
    results = pd.DataFrame(columns=results_columns)
    selected_features_all = []

    for i in range(0, 2):
        for j in range(0, 2):
            # Feature Selection
            X_new, selected_features_idx = feature_selection(X, y, feature_method, k)
            selected_features_all.append({
                'anonymized_train': bol[i],
                'anonymized_test': bol[j],
                'model': model_name,
                'feature_method': feature_method,
                'num_features': k,
                'selected_features_idx': selected_features_idx.tolist()
            })
            cross_val_results = cross_validate_k_fold(X_new, y, bol[i], bol[j], model, model_name, n_clusters)
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

# Function where the experiment are maded
def experiment(X, y, feature_method, k):
    all_results = pd.DataFrame(columns=['model', 'anonymized_train', 'anonymized_test', 'accuracy', 'precision', 'recall', 'f1_score', 'selected_features', 'feature_method', 'num_features'])
    models = [
        (KNeighborsClassifier(n_neighbors=5), 'KNN'),
        # (DecisionTreeClassifier(), 'Decision Tree'),
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
        ), 'Mutilayer Perceptron'),
        (AdaBoostClassifier(n_estimators=100, learning_rate=1.0), 'AdaBoost'),
        (LogisticRegression(max_iter=1000, C=1.0, solver='lbfgs', multi_class='auto'), 'Logistic Regression')
    ]

    for model, model_name in models:
        results, selected_features = get_result(model, X, y, model_name, 3, feature_method, k)
        # Find the best resutlt for which scenario
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

def Chi2(X, y):
    all_best_results = []


    for i in range(2, 80, 5):
        best_results = experiment(X, y, 'chi2', i)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save results with absolute path for clarity
    absolute_path = os.path.join(os.getcwd(), 'results', 'best_results_chi2.csv')
    final_best_results_df.to_csv(absolute_path, index=False)
    print(f"Resultados Chi2 salvos em: {absolute_path}")
    print(final_best_results_df.head())



def ExtraTree(X, y):
    all_best_results = []

    for i in range(2, 80, 5):
        best_results = experiment(X, y, 'extra_trees', i)
        all_best_results.append(best_results)

    final_best_results_df = pd.concat(all_best_results, ignore_index=True)
    
    # Ensure results directory exists
    os.makedirs('results', exist_ok=True)
    
    # Save results with absolute path for clarity
    absolute_path = os.path.join(os.getcwd(), 'results', 'best_results_extra_trees.csv')
    final_best_results_df.to_csv(absolute_path, index=False)
    print(f"Resultados ExtraTree salvos em: {absolute_path}")


def main():
    # Set the renadom seed 
    np.random.seed(7) 

    # read the dataset
    dataset = get_file()
    dataset.head(5)
    print(f"Total de colunas no dataset: {len(dataset.columns)}")

    # Extract feature and label to ddos, income to adults, HeartDisease to heart
    # method to cmc, severity to mgm, ocean_proximity to cahousing
    y = np.array(dataset['ocean_proximity'])
    del dataset['ocean_proximity']
    X = np.array(dataset)

    # Execute somente Chi2 e ExtraTree
    Chi2(X, y)
    ExtraTree(X, y)


if __name__ == "__main__":
    main()