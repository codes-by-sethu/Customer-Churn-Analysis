from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_auc_score
import joblib
import numpy as np
from typing import Dict
import yaml

def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)

def train_all_models(X_train, y_train, X_test, y_test):
    config = load_config()
    
    models = {
        'Logistic': LogisticRegression(**config['models']['logistic'], random_state=42),
        'RandomForest': RandomForestClassifier(**config['models']['rf'], random_state=42),
        'XGBoost': XGBClassifier(**config['models']['xgb'], random_state=42)
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        auc = roc_auc_score(y_test, y_pred_proba)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='roc_auc')
        
        results[name] = {'auc': auc, 'cv_mean': cv_scores.mean(), 'model': model}
        
        # Save best model
        if auc == max([r['auc'] for r in results.values()]):
            joblib.dump(model, 'models/best_model.pkl')
    
    joblib.dump(results, 'models/model_results.pkl')
    return results
