import pandas as pd
import os
from glob import glob
import logging

def find_dataset(data_dir="data/raw"):
    """Dynamically find ANY CSV dataset in data/raw"""
    csv_files = glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV found in {data_dir}. Download any churn dataset here.")
    
    filepath = csv_files[0]  # Use first CSV found
    logging.info(f"Using dataset: {filepath}")
    return filepath

def auto_detect_target(df, common_targets=["Churn", "churn", "target", "label"]):
    """Auto-detect churn column"""
    for col in common_targets:
        if col in df.columns:
            logging.info(f"Target detected: {col}")
            return col
    raise ValueError("No churn target column found. Expected: Churn, churn, target, etc.")
