import pandas as pd
import os
from glob import glob
from src.data_loader import auto_detect_target

def get_all_datasets():
    """
    Scan data/raw/ and return dataset info.

    Returns:
        List of dictionaries with keys:
            - name: dataset name (filename without .csv)
            - path: full path to CSV
            - rows: number of rows
            - features: number of columns
            - target: detected target column
            - churn_rate: percentage of minority class (for churn)
    """
    csv_files = glob('data/raw/*.csv')
    datasets = []

    for filepath in csv_files:
        filename = os.path.basename(filepath)
        try:
            df = pd.read_csv(filepath)

            # Auto-detect target
            target_col = auto_detect_target(df)
            if target_col not in df.columns:
                print(f"Skipping {filename}: target column not found")
                continue  # skip if detection failed

            shape = df.shape
            # Safe churn rate calculation
            counts = df[target_col].value_counts(normalize=True)
            churn_rate = counts.min() * 100 if len(counts) > 1 else 0

            datasets.append({
                'name': filename.replace('.csv', ''),
                'path': filepath,
                'rows': shape[0],
                'features': shape[1],
                'target': target_col,
                'churn_rate': f'{churn_rate:.1f}% churn'
            })

        except Exception as e:
            print(f"Skipping {filename}: {e}")
            continue

    # Sort datasets by name for consistent UI
    return sorted(datasets, key=lambda d: d['name'])


def load_selected_dataset(dataset_name):
    """
    Load a dataset by name.

    Args:
        dataset_name (str): filename without .csv

    Returns:
        pd.DataFrame: loaded dataset

    Raises:
        FileNotFoundError: if CSV does not exist in data/raw
    """
    filepath = os.path.join("data/raw", f"{dataset_name}.csv")
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    raise FileNotFoundError(f"Dataset {dataset_name} not found in data/raw/")
