# 🔮 Universal Customer Churn Predictor

**Zero configuration ML pipeline** - works with **ANY churn dataset automatically!**

## ✨ Key Features
- ✅ **Dynamic data loading** - any CSV with "Churn" target column
- ✅ **Auto feature detection** - sliders, dropdowns, Yes/No detected automatically
- ✅ **Multi-model training** - Logistic Regression, Random Forest, Gradient Boosting
- ✅ **Streamlit demo** - live predictions with feature importance
- **Verified**: Telco (11 features), Bank (9+ features), Orange Telecom (19 features)

## 📁 Project Structure
```
D:\MyProjects\Customer-Churn-Analysis\
├── app\
│   └── app.py              # Main Streamlit predictor
├── src\
│   ├── dataset_selector.py
│   ├── preprocessor.py
│   └── trainer.py
├── data\
│   └── raw\                # Drop your CSV files here
├── models\                 # Auto-generated models (.gitignore)
├── venv\                   # Virtual environment (.gitignore)
├── .gitignore              # venv, models, pycache excluded
├── requirements.txt
└── README.md
```

## 🚀 Quick Start
```bash
# 1. Activate virtual environment
D:\MyProjects\Customer-Churn-Analysis\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add CSV files to data/raw/
# Your datasets: customer_churn_hf, orange-telecom-20, etc.

# 4. Launch predictor
streamlit run app/app.py
```

## 🎯 How It Works
```
1. Select dataset → Auto-trains best model (highest AUC)
2. Smart input forms → Sliders for tenure/age, dropdowns for categorical
3. Predict → Real-time probability + business actions
4. Insights → Top risk factors + feature importance chart
```

## ✅ Verified Results
| Dataset | Features | Sample Prediction | Status |
|---------|----------|-------------------|--------|
| customer_churn_hf | 11 | 49-52% Medium Risk | ✅ |
| orange-telecom-20 | **19** | **96.4% HIGH RISK** | ✅ |
| Bank dataset | 9+ | 50.2% Low Risk | ✅ |

## 💡 Sample High-Risk Tests

### Orange Telecom (96.4% achieved):
```
International plan: Yes
Customer service calls: 5
```

### Expected High Risk Patterns:
```
Telco: tenure=1 + Fiber optic + SeniorCitizen=Yes
Bank: IsActiveMember=No + Age=25 + Spain
Orange: International plan=Yes + Customer service calls=5+
```

## 🔧 Git Clean Configuration
```
✅ venv/ excluded
✅ models/*.pkl excluded  
✅ data/processed/ excluded
✅ __pycache__/ excluded
✅ .ipynb_checkpoints/ excluded
✅ .DS_Store excluded
```

## 🎉 Production Ready Features
- **Universal**: Works with 9, 11, 19+ features automatically
- **Smart UI**: Auto-detects field types (sliders/dropdowns/numeric)
- **Business Actions**: Emergency retention workflows
- **Model Insights**: Feature importance + risk explanations
- **Zero Config**: Drop CSV → instant predictor

**Drop any churn CSV in `data/raw/` → production-ready predictor in 30 seconds!** 🚀
