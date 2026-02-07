import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
from typing import Tuple, List
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DynamicPreprocessor:
    def __init__(self, test_size=0.2, random_state=42):
        self.test_size = test_size
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.le_target = LabelEncoder()
        self.feature_names: List[str] = None
        self.target_encoder = None
        
    def fit(self, df: pd.DataFrame, target_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        🔥 FULLY FIXED: AUTO-ENCODES ALL FEATURES (numeric + categorical)!
        No more dropping gender/Partner/Contract columns!
        """
        print(f"🔍 Starting preprocessing: {len(df)} rows, {len(df.columns)} columns")
        print(f"🔍 Columns: {df.columns.tolist()}")
        
        df = df.copy()
        
        # Step 1: Encode target
        print(f"🔍 Target '{target_col}' distribution:\n{df[target_col].value_counts()}")
        df[target_col] = self.le_target.fit_transform(df[target_col].astype(str))
        print(f"🔍 Encoded target: {self.le_target.classes_}")
        
        # Step 2: 🔥 FIX - Identify ALL potential feature columns
        feature_cols = [col for col in df.columns if col != target_col]
        print(f"🔍 Potential features: {len(feature_cols)}")
        
        # Step 3: 🔥 AUTO-ENCODE ALL CATEGORICAL COLUMNS
        categorical_encoders = {}
        for col in feature_cols:
            if df[col].dtype == 'object' or df[col].dtype.name == 'category':
                print(f"🔍 Encoding categorical: {col}")
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                categorical_encoders[col] = le
        
        # Step 4: Clean numeric columns
        for col in feature_cols:
            if df[col].dtype in [np.number, 'int64', 'float64']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(df[col].median())
        
        # Step 5: Final feature matrix (ALL columns except target)
        X = df[feature_cols].fillna(0)
        y = df[target_col]
        
        self.feature_names = feature_cols
        self.categorical_encoders = categorical_encoders
        
        print(f"✅ FINAL: {len(self.feature_names)} features: {self.feature_names[:10]}...")
        print(f"✅ X shape: {X.shape}, y shape: {y.shape}")
        
        # Step 6: Train/test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.test_size, 
            random_state=self.random_state, 
            stratify=y
        )
        
        # Step 7: Scale ALL features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Step 8: SMOTE if imbalanced
        if len(np.unique(y_train)) > 1 and abs(y_train.mean() - 0.5) > 0.1:
            print(f"🔍 Applying SMOTE: {y_train.mean():.3f} → balanced")
            smote = SMOTE(random_state=self.random_state)
            X_train_scaled, y_train = smote.fit_resample(X_train_scaled, y_train)
        else:
            print("✅ Balanced enough, skipping SMOTE")
        
        print(f"✅ Train shape: {X_train_scaled.shape}, Test: {X_test_scaled.shape}")
        print(f"✅ Churn rate: train={y_train.mean():.1%}, test={y_test.mean():.1%}")
        
        return X_train_scaled, y_train, X_test_scaled, y_test
    
    def transform_new_data(self, new_data: dict) -> np.ndarray:
        """
        Transform prediction inputs from Streamlit app
        Input: dict like {'SeniorCitizen': 'Yes', 'tenure': 12.0, ...}
        """
        # Convert dict to DataFrame
        input_df = pd.DataFrame([new_data])
        
        # Reindex to EXACT training features (fill missing with 0)
        X_new = input_df.reindex(columns=self.feature_names, fill_value=0)
        
        # Handle Yes/No → 1/0 for prediction
        for col in X_new.columns:
            if X_new[col].dtype == 'object':
                X_new[col] = X_new[col].map({'Yes': 1, 'No': 0}).fillna(0)
        
        # Ensure numeric
        X_new = X_new.astype(float)
        print(f"🔍 Prediction input: {X_new.shape} → {X_new.columns.tolist()[:5]}")
        
        # Scale with training scaler
        X_scaled = self.scaler.transform(X_new)
        return X_scaled
    
    def save_artifacts(self, dataset_name: str):
        """Save all preprocessing artifacts"""
        os.makedirs("models", exist_ok=True)
        joblib.dump(self.scaler, f"models/{dataset_name}_scaler.pkl")
        joblib.dump(self.feature_names, f"models/{dataset_name}_features.pkl")
        joblib.dump(self.le_target, f"models/{dataset_name}_target_encoder.pkl")
        joblib.dump(self.categorical_encoders, f"models/{dataset_name}_cat_encoders.pkl")
        print(f"✅ Artifacts saved for {dataset_name}")
    
    @classmethod
    def load_artifacts(cls, dataset_name: str):
        """Load preprocessor artifacts"""
        try:
            scaler = joblib.load(f"models/{dataset_name}_scaler.pkl")
            features = joblib.load(f"models/{dataset_name}_features.pkl")
            
            preprocessor = cls()
            preprocessor.scaler = scaler
            preprocessor.feature_names = features
            return preprocessor
        except FileNotFoundError:
            print(f"❌ Artifacts not found for {dataset_name}")
            return None

# Quick test function
def test_preprocessor(dataset_path: str, target_col: str = "Churn"):
    """Test your preprocessor works"""
    df = pd.read_csv(dataset_path)
    print("Dataset preview:")
    print(df.head())
    print(f"Shape: {df.shape}")
    
    preprocessor = DynamicPreprocessor()
    X_train, y_train, X_test, y_test = preprocessor.fit(df, target_col)
    
    # Test prediction transform
    sample_input = {feat: 50.0 for feat in preprocessor.feature_names[:5]}
    sample_input['tenure'] = 12.0 if 'tenure' in sample_input else 12.0
    X_pred = preprocessor.transform_new_data(sample_input)
    
    print(f"✅ Prediction transform works: {X_pred.shape}")
    return preprocessor

if __name__ == "__main__":
    # Test with your dataset
    # preprocessor = test_preprocessor("data/raw/customer_churn_hf.csv")
    print("✅ DynamicPreprocessor ready!")
