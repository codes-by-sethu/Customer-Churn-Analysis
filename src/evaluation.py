import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import joblib

def plot_feature_importance(model, feature_names: list, top_n: int = 10):
    if hasattr(model, 'feature_importances_'):
        importance = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=importance, x='importance', y='feature')
        plt.title('Top 10 Feature Importance')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png')
        plt.show()

def generate_report(results: dict, X_test: np.ndarray, y_test: np.ndarray):
    best_model_name = max(results.keys(), key=lambda k: results[k]['auc'])
    best_model = results[best_model_name]['model']
    
    y_pred = best_model.predict(X_test)
    
    print(f"\n{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"AUC Score: {results[best_model_name]['auc']:.3f}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.savefig('models/confusion_matrix.png')
    plt.show()
