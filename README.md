# Adaptive Anonymization with Mutual Information and Differential Privacy

This project implements an adaptive data anonymization framework that uses mutual information analysis combined with differential privacy mechanisms. The system dynamically allocates privacy budgets based on feature importance and correlation patterns, optimizing the utility-privacy trade-off for machine learning tasks.

## Key Features

- **MI-Adaptive Differential Privacy**: Dynamically allocates epsilon budget based on mutual information scores
- **Correlation-Aware Noise Addition**: Groups correlated features for coordinated noise injection
- **K-Means Clustering Integration**: Applies anonymization within data clusters for better utility preservation
- **Feature Selection Integration**: Supports Chi2 and ExtraTrees feature selection methods
- **Multi-Model Evaluation**: Tests 6 different ML models with hyperparameter optimization using Optuna
- **Comprehensive Anonymization Scenarios**: Evaluates all combinations of anonymized/non-anonymized training and test sets

## Core Algorithm

The framework implements a novel MI-Adaptive Differential Privacy approach:

1. **Mutual Information Analysis**: Calculates feature importance using mutual information with target variable
2. **Feature Redundancy Detection**: Groups highly correlated features (correlation > threshold)
3. **Adaptive Epsilon Allocation**: Allocates privacy budget inversely proportional to feature importance
4. **Correlation-Aware Noise**: Adds coordinated noise to correlated feature groups

## Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```
## Project Structure

```
src/
├── main.py                     # Main experiment runner with Optuna optimization
├── ml.py                       # Cross-validation and model evaluation
├── file_utils.py               # Dataset loading and preprocessing
└── anonymization/
    ├── anon_main.py           # MIAdaptiveDPAnonymizer main class
    ├── clustering.py          # K-means clustering integration
    ├── mu.py                  # Mutual information analysis
    ├── noise_alocation.py     # Adaptive epsilon budget allocation
    └── dp_mechanism.py        # Differential privacy noise mechanisms
```

## Supported Datasets

The framework includes 7 pre-configured datasets:

1. **adults**: Adult income classification (label: 'income')
2. **bank**: Bank marketing campaign (label: 'y')  
3. **ddos**: DDoS attack detection (label: 'Label')
4. **heart**: Heart disease prediction (label: 'HeartDisease')
5. **cmc**: Contraceptive method choice (label: 'method')
6. **mgm**: Medical dataset (label: 'severity')
7. **cahousing**: California housing classification (label: 'ocean_proximity')

## Usage

### Basic Usage

Run with default parameters on California housing dataset:

```bash
python src/main.py
```

### Specify Dataset and Parameters

```bash
python src/main.py adults --epsilon=1.0 --mi_weight=0.8 --correlation_threshold=0.7
```

### Available Parameters

- `--epsilon=VALUE`: Privacy budget (default: 1.0)
- `--mi_weight=VALUE`: Weight for MI-based allocation (default: 0.8) 
- `--correlation_threshold=VALUE`: Correlation threshold for grouping (default: 0.7)
- `--noise_type=TYPE`: 'laplace' or 'gaussian' (default: 'laplace')
- `--n_trials=VALUE`: Optuna optimization trials per scenario (default: 20)

### Examples

High privacy protection:
```bash
python src/main.py heart --epsilon=0.1 --mi_weight=0.9
```

Focus on correlation patterns:
```bash
python src/main.py adults --correlation_threshold=0.5 --mi_weight=0.5
```

Fast experimentation:
```bash
python src/main.py mgm --n_trials=10
```

### Get Help

Display available options and datasets:

```bash
python src/main.py --help
```

## Experimental Design

The framework evaluates **4 anonymization scenarios** for each model:

1. **No Anonymization**: Original training and test data
2. **Training Only**: Anonymized training data, original test data  
3. **Testing Only**: Original training data, anonymized test data
4. **Full Anonymization**: Both training and test data anonymized

### Machine Learning Models

The system tests 6 different models with automatic hyperparameter optimization:

- **K-Nearest Neighbors (KNN)**
- **Random Forest**
- **Gaussian Naive Bayes**
- **Multi-Layer Perceptron (MLP)**
- **AdaBoost**
- **Logistic Regression**

### Feature Selection Methods

Two feature selection approaches are automatically tested:

- **Chi2**: Statistical test-based feature selection

## Output Files

Results are saved in the `results/` directory:

- `mi_adaptive_chi2_<dataset>_eps_<epsilon>_miw_<mi_weight>_<noise_type>_optuna.csv`
- `mi_adaptive_extra_trees_<dataset>_eps_<epsilon>_miw_<mi_weight>_<noise_type>_optuna.csv`

### Result Columns

Each CSV contains:
- `model`: ML model name
- `anonymized_train/test`: Whether training/test data was anonymized  
- `accuracy/precision/recall/f1_score`: Performance metrics
- `anon_train_time/anon_test_time`: Anonymization processing times
- `model_train_time`: Model training time
- `selected_features`: Indices of selected features
- `feature_method`: Feature selection method used
- `num_features`: Number of features selected
- `best_params`: Optimal hyperparameters found by Optuna

## Algorithm Details

### MI-Adaptive Epsilon Allocation

The privacy budget allocation follows:

```
epsilon_i = base_epsilon + adaptive_epsilon * (1 - importance_i * mi_weight)
```

Where:
- `base_epsilon = total_epsilon * min_epsilon_ratio` (minimum privacy for each feature)
- `importance_i` is the normalized mutual information score
- `mi_weight` controls the strength of adaptive allocation

### Correlation-Aware Noise

For correlated feature groups:
1. Calculate group correlation matrix
2. Add coordinated noise based on correlation strength  
3. Apply correlation factor to adjust noise variance



## Privacy Guarantees

The framework provides **ε-differential privacy** with:
- Formal privacy accounting across clusters
- Adaptive budget allocation based on feature utility
- Support for both Laplace and Gaussian noise mechanisms

## Performance Optimization

- **Optuna Integration**: Automatic hyperparameter tuning for all models
- **Cross-Validation**: 3-fold stratified cross-validation for robust evaluation
- **Feature Selection**: Reduces dimensionality before anonymization
- **Clustering**: Improves utility by preserving local data structure

