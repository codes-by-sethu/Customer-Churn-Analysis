import sys
import os

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# ✅ Import your modules
from src.dataset_selector import get_all_datasets, load_selected_dataset
from src.preprocessor import DynamicPreprocessor
from src.trainer import train_all_models

# Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Streamlit page config
st.set_page_config(page_title="Universal Churn Predictor", layout="wide", page_icon="🔮")
st.title("🔮 Universal Churn Prediction Engine")
st.markdown("**Select dataset → Auto-train → Live predictions**")

# Sidebar: Dataset browser
st.sidebar.header("📊 Available Datasets")
datasets = get_all_datasets()
if not datasets:
    st.error("❌ No datasets in `data/raw/`. Add CSV files!")
    st.stop()

dataset_options = {f"{d['name']} ({d['rows']} customers)": d['name'] for d in datasets}
selected_dataset = st.sidebar.selectbox("Choose Dataset", options=list(dataset_options.keys()))
dataset_name = dataset_options[selected_dataset]

# Tabs: Live prediction & model training
tab1, tab2 = st.tabs(["🎯 Live Predictor", "⚙️ Train New Model"])

# ✅ FIXED: Function moved to GLOBAL SCOPE (before tabs)
@st.cache_data
def get_feature_types(dataset_name, features):
    try:
        df_sample = load_selected_dataset(dataset_name).head(100)
        feature_types = {}
        for feature in features:
            if feature in df_sample.columns:
                col_data = df_sample[feature].dropna()
                unique_vals = col_data.unique()
                unique_count = len(unique_vals)
                
                # ✅ FIXED: SeniorCitizen & similar ALWAYS binary Yes/No - FIRST priority
                feature_lower = feature.lower()
                if any(x in feature_lower for x in ['seniorcitizen', 'senior', 'partner', 'dependents', 
                                                   'phoneservice', 'paperlessbilling']):
                    feature_types[feature] = {'type': 'binary', 'options': ['No', 'Yes']}
                
                # Categorical: object type OR few unique values (<10) BUT NOT binary
                elif col_data.dtype in ['object', 'string'] or (unique_count <= 10 and unique_count > 2):
                    options = sorted(unique_vals[:8])
                    feature_types[feature] = {'type': 'categorical', 'options': options}
                    
                # Numeric sliders for common churn features
                elif any(x in feature_lower for x in ['tenure', 'monthlycharges', 'totalcharges']):
                    feature_types[feature] = {'type': 'slider', 'min': 0.0, 'max': 100.0}
                else:
                    feature_types[feature] = {'type': 'numeric', 'min': 0.0, 'max': 100.0}
            else:
                # Fallback - PRIORITIZE binary fields
                feature_lower = feature.lower()
                if any(x in feature_lower for x in ['seniorcitizen', 'senior', 'partner', 'dependents', 
                                                  'gender', 'phoneservice', 'paperlessbilling']):
                    feature_types[feature] = {'type': 'binary', 'options': ['No', 'Yes']}
                elif any(x in feature_lower for x in ['tenure', 'monthlycharges', 'totalcharges']):
                    feature_types[feature] = {'type': 'slider', 'min': 0.0, 'max': 100.0}
                else:
                    feature_types[feature] = {'type': 'numeric', 'min': 0.0, 'max': 100.0}
        return feature_types
    except:
        # Ultimate fallback
        feature_types = {}
        for feature in features:
            feature_lower = feature.lower()
            if any(x in feature_lower for x in ['seniorcitizen', 'senior', 'partner', 'dependents', 
                                              'gender', 'phoneservice', 'paperlessbilling']):
                feature_types[feature] = {'type': 'binary', 'options': ['No', 'Yes']}
            elif any(x in feature_lower for x in ['tenure', 'monthlycharges', 'totalcharges']):
                feature_types[feature] = {'type': 'slider', 'min': 0.0, 'max': 100.0}
            else:
                feature_types[feature] = {'type': 'numeric', 'min': 0.0, 'max': 100.0}
        return feature_types

with tab1:
    model_path = f"models/{dataset_name}_model.pkl"
    features_path = f"models/{dataset_name}_features.pkl"
    scaler_path = f"models/{dataset_name}_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        features = joblib.load(features_path)
        
        # Load scaler if exists
        scaler = None
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
        
        st.success(f"✅ **{dataset_name} model loaded** (trained on {len(features)} features)")

        # ✅ Smart dynamic input form - matches screenshot exactly
        st.subheader("👤 Enter Customer Data")
        input_data = {}

        feature_types = get_feature_types(dataset_name, features)
        
        # Dynamic form layout (4 columns max, like screenshot)
        n_cols = min(4, len(features))
        cols = st.columns(n_cols)
        
        for i, feature in enumerate(features):
            col_idx = i % n_cols
            with cols[col_idx]:
                safe_name = feature.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')
                key = f"{dataset_name}_{safe_name}_{i}"
                
                ftype = feature_types.get(feature, {'type': 'numeric', 'min': 0.0, 'max': 100.0})
                
                if ftype['type'] == 'categorical':
                    # ✅ Dropdown with REAL dataset values (InternetService, etc.)
                    options = ftype['options']
                    input_data[feature] = st.selectbox(feature, options, index=0, key=key)
                    
                elif ftype['type'] == 'binary':
                    # ✅ Yes/No dropdowns (SeniorCitizen, Partner, etc.)
                    input_data[feature] = st.selectbox(feature, ['No', 'Yes'], index=0, key=key)
                    
                elif ftype['type'] == 'slider':
                    # ✅ Red sliders (tenure, charges - exactly like screenshot)
                    input_data[feature] = st.slider(feature, 
                                                  min_value=float(ftype['min']), 
                                                  max_value=float(ftype['max']), 
                                                  value=50.0, 
                                                  key=key)
                    
                else:
                    # ✅ Numeric inputs (customerID, etc.)
                    input_data[feature] = st.number_input(feature, 
                                                        min_value=float(ftype['min']), 
                                                        max_value=float(ftype['max']), 
                                                        value=50.0, 
                                                        key=key)

        if st.button("🚀 Predict Churn", type="primary"):
            # 🔥 Create properly formatted input array
            input_array = np.zeros((1, len(features)))
            for i, feature in enumerate(features):
                val = input_data.get(feature, 0)
                if isinstance(val, str):
                    input_array[0, i] = 1.0 if val == 'Yes' else 0.0
                else:
                    input_array[0, i] = float(val)

            # Scale if available (matches training)
            if scaler is not None:
                input_array = scaler.transform(input_array)

            prob = model.predict_proba(input_array)[0, 1]

            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Churn Probability", f"{prob:.1%}")
            with col2:
                if prob > 0.7: st.error("🚨 HIGH RISK")
                elif prob > 0.4: st.warning("⚠️ MEDIUM RISK")
                else: st.success("✅ LOW RISK")

            # ✅ Feature importance analysis
            st.markdown("---")
            st.subheader("💡 Why This Customer Will/Might Leave?")

            top_reasons = []
            try:
                if hasattr(model, 'coef_'):  # Logistic Regression
                    coef_abs = np.abs(model.coef_[0])
                    top_idx = np.argsort(coef_abs)[-min(3, len(features)):][::-1]
                    for i in top_idx:
                        feature_name = features[i]
                        feature_val = input_data.get(feature_name, 0)
                        
                        display_val = feature_val if isinstance(feature_val, str) else round(float(feature_val), 1)
                        direction = "high" if model.coef_[0][i] > 0 else "low"
                        top_reasons.append((feature_name, display_val, direction))
                        
                elif hasattr(model, 'feature_importances_'):  # Tree models
                    imp = model.feature_importances_
                    top_idx = np.argsort(imp)[-min(3, len(features)):][::-1]
                    for i in top_idx:
                        feature_name = features[i]
                        feature_val = input_data.get(feature_name, 0)
                        
                        display_val = feature_val if isinstance(feature_val, str) else round(float(feature_val), 1)
                        
                        num_val = 1.0 if feature_val == 'Yes' else 0.0 if feature_val == 'No' else float(feature_val or 0)
                        threshold = 0.5 if isinstance(feature_val, str) else 50.0
                        direction = "high" if num_val > threshold else "low"
                        top_reasons.append((feature_name, display_val, direction))
                        
                else:
                    for feature in features[:3]:
                        feature_val = input_data.get(feature, 0)
                        display_val = feature_val if isinstance(feature_val, str) else round(float(feature_val or 0), 1)
                        top_reasons.append((feature, display_val, "medium"))
                        
            except Exception as e:
                st.error(f"Feature analysis error: {e}")
                for i, feature in enumerate(features[:3]):
                    top_reasons.append((feature, "N/A", "medium"))

            # Business interpretation
            col1, col2, col3 = st.columns([1, 2, 1])

            with col1:
                st.markdown("**🎯 Risk Level**")
                if prob > 0.7:
                    st.error("**🚨 Will leave**")
                elif prob > 0.4:
                    st.warning("**⚠️ Might leave**")
                else:
                    st.success("**✅ Staying**")

            with col2:
                st.markdown("**🔍 Top Risk Factors:**")
                for feature, value, direction in top_reasons[:3]:
                    color = "🔴" if prob > 0.7 else "🟡" if prob > 0.4 else "🟢"
                    st.write(f"{color} **{feature}** is {direction}")

            with col3:
                st.markdown("**💰 Revenue Impact**")
                if prob > 0.7:
                    st.error("**High loss**")
                elif prob > 0.4:
                    st.warning("**Medium loss**")
                else:
                    st.success("**Safe**")

            # Actionable recommendations
            st.markdown("**🎯 Save This Customer:**")
            if prob > 0.7:
                st.error("• 💰 **Offer 20% discount NOW**")
                st.error("• 📞 **Call this week**")
                st.error("• 🎁 **Free 3 months streaming**")
            elif prob > 0.4:
                st.warning("• 📧 **Send retention email**")
                st.warning("• 💸 **Review their billing**")
                st.warning("• 📊 **Check service usage**")
            else:
                st.success("• ✅ **Monitor monthly**")
                st.success("• 😊 **Continue normal service**")

            # Technical details
            with st.expander("🔧 Technical Details"):
                st.markdown("**📊 Model Analysis**")
                
                try:
                    if hasattr(model, 'coef_'):
                        importance = np.abs(model.coef_[0])
                    elif hasattr(model, 'feature_importances_'):
                        importance = model.feature_importances_
                    else:
                        importance = np.ones(len(features))

                    imp_df = pd.DataFrame({
                        'Feature': features,
                        'Importance': importance
                    }).sort_values('Importance', ascending=True)
                    
                    st.markdown("**Feature Importance:**")
                    st.bar_chart(imp_df.set_index('Feature')['Importance'])
                    
                    st.markdown("**Detailed Breakdown:**")
                    st.dataframe(imp_df.round(3))
                    
                except Exception as e:
                    st.info(f"📈 Model analysis unavailable: {e}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Churn Risk", f"{prob:.1%}", "57.9%")
                with col2:
                    st.metric("Stay Risk", f"{1-prob:.1%}", "42.1%")
                    
    else:
        st.warning(f"⚠️ No model for **{dataset_name}**. Train it first below 👇")

with tab2:
    st.subheader("⚙️ Train Model")
    if st.button(f"🚀 Train {dataset_name} Model", type="primary"):
        with st.spinner(f"Training {dataset_name}..."):
            df = load_selected_dataset(dataset_name)
            preprocessor = DynamicPreprocessor()

            X_train, y_train, X_test, y_test = preprocessor.fit(df, target_col="Churn")
            results = train_all_models(X_train, y_train, X_test, y_test)

            model_path = f"models/{dataset_name}_model.pkl"
            features_path = f"models/{dataset_name}_features.pkl"
            scaler_path = f"models/{dataset_name}_scaler.pkl"
            results_path = f"models/{dataset_name}_results.pkl"
            
            best_name = max(results, key=lambda k: results[k]['auc'])
            joblib.dump(results[best_name]['model'], model_path)
            joblib.dump(preprocessor.feature_names, features_path)
            joblib.dump(preprocessor.scaler, scaler_path)
            joblib.dump(results, results_path)

            st.success(f"✅ **{dataset_name} trained!** Best: {best_name} (AUC: {results[best_name]['auc']:.3f})")
            st.rerun()

# Dataset comparison
st.markdown("---")
st.subheader("📈 Dataset Comparison")
comparison_data = []
for dataset in datasets:
    model_file = f"models/{dataset['name']}_model.pkl"
    results_file = f"models/{dataset['name']}_results.pkl"
    if os.path.exists(model_file) and os.path.exists(results_file):
        results = joblib.load(results_file)
        best_auc = max(r['auc'] for r in results.values())
        comparison_data.append([dataset['name'], dataset['rows'], dataset['target'], f"{best_auc:.3f}"])

if comparison_data:
    st.dataframe(pd.DataFrame(comparison_data, columns=["Dataset", "Customers", "Target", "Best AUC"]))
else:
    st.info("No models trained yet for comparison.")
