import os
import sys
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

DATASETS = {
    "adults": {
        "filename": "adults.csv",
        "label_column": "income"
    },
    "bank": {
        "filename": "banking.csv",
        "label_column": "y"  
    },
    "ddos": {
        "filename": "ddos_csv_100000.csv",
        "label_column": "Label"
    },
    "heart": {
        "filename": "heart.csv",
        "label_column": "HeartDisease"
    },
    "cmc": {
        "filename": "cmc.csv",
        "label_column": "method"
    },
    "mgm": {
        "filename": "mgm.csv",
        "label_column": "severity"
    },
    "cahousing": {
        "filename": "cahousing.csv",
        "label_column": "ocean_proximity"
    }
}

def list_available_datasets():
    """Display all available datasets and their label columns"""
    print("\nAvailable datasets:")
    print("------------------")
    for i, (ds_name, ds_info) in enumerate(DATASETS.items()):
        print(f"{i+1}. {ds_name}: label = '{ds_info['label_column']}', file = '{ds_info['filename']}'")

def get_file(dataset_name=None):
    """
    Load and preprocess a dataset
    """
    if dataset_name is None:
        if len(sys.argv) > 1:
            dataset_name = sys.argv[1].lower()
        else:
            dataset_name = "cahousing"
            print("\nNo dataset specified. Using default dataset 'cahousing'.")
            print("To specify a dataset, run: python main.py <dataset_name>")
            list_available_datasets()
    
    if dataset_name not in DATASETS:
        print(f"\nError: Dataset '{dataset_name}' not found!")
        list_available_datasets()
        sys.exit(1)
        
    dataset_info = DATASETS[dataset_name]
    filename = dataset_info["filename"]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, '..', 'data', filename)
    print(f"File path: {file_path}")
    print(f"Selected dataset: {dataset_name}, label column: {dataset_info['label_column']}")
    
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        sys.exit(1)
        
    dataset = pd.read_csv(file_path)
    
    num_columns = len(dataset.columns)
    print(f"Number of columns in dataset: {num_columns}")
    
    print("Dataset columns:")
    for i, column in enumerate(dataset.columns):
        print(f"  {i+1}. {column}")
    
    print("\nApplying Label Encoder to non-numeric columns:")
    for column in dataset.columns:
        if dataset[column].dtype == 'object':
            print(f"  Applying Label Encoder to column: {column}")
            le = LabelEncoder()
            dataset[column] = le.fit_transform(dataset[column])
    
    return dataset, dataset_info["label_column"]