#!/usr/bin/env python3
"""
Dynamic Customer Churn Pipeline - Works with ANY churn dataset
"""
import sys
import os
sys.path.append('src')

from data_loader import find_dataset, auto_detect_target
from preprocessor import DynamicPreprocessor
from trainer import train_all_models
import pandas as pd
import joblib
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)

def main():
    print("🚀 Dynamic Customer Churn Prediction Pipeline")
    
    # STEP 1: Auto-detect dataset
    filepath = find_dataset()
    df = pd.read_csv(filepath)
    target_col = auto_detect_target(df)
    print(f"📊 Dataset: {df.shape} | Target: {target_col}")
    
    # STEP 2: Dynamic preprocessing
    preprocessor = DynamicPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.fit(df, target_col)
    
    # STEP 3: Train models
    results = train_all_models(X_train, y_train, X_test, y_test)
    
    # STEP 4: Results
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_auc = results[best_model_name]['auc']
    
    print(f"\n🎉 SUCCESS!")
    print(f"Best Model: {best_model_name} (AUC: {best_auc:.3f})")
    print(f"Features used: {len(preprocessor.feature_names)}")
    print(f"Models saved: models/best_model.pkl")
    
    # Quick prediction test
    model = joblib.load('models/best_model.pkl')
    sample_pred = model.predict_proba(X_test[:1])[0, 1]
    print(f"Sample prediction: {sample_pred:.1%} churn probability")

if __name__ == "__main__":
    main()
