```markdown
# Credit Default Prediction

End-to-end machine learning system for predicting credit default risk, deployed as a REST API.

**Status**: ✅ Deployed (Production-ready API)

## 🚀 Live Demo

- **API**: https://credit-default-prediction-api.onrender.com
- **Interactive Docs (Swagger UI)**: https://credit-default-prediction-api.onrender.com/docs

> **Note**: API is hosted on Render's free tier. First request after 15 minutes of inactivity may take 30–50 seconds (cold start). Subsequent requests respond in milliseconds.

### Example usage

`POST /predict` with borrower data:

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.95,
  "age": 25,
  "NumberOfTime30_59DaysPastDueNotWorse": 3,
  "DebtRatio": 0.8,
  "MonthlyIncome": 2000,
  "NumberOfOpenCreditLinesAndLoans": 5,
  "NumberOfTimes90DaysLate": 2,
  "NumberRealEstateLoansOrLines": 0,
  "NumberOfTime60_89DaysPastDueNotWorse": 1,
  "NumberOfDependents": 2,
  "income_missing_flag": 0,
  "total_delinquency": 6,
  "has_dependents": 1,
  "income_per_dependent": 666.67,
  "zero_income_flag": 0
}
```

Response:

```json
{
  "default_probability": 0.9626,
  "default_probability_percent": "96.26%",
  "prediction": "DEFAULT",
  "risk_level": "HIGH"
}
```

## 📊 Results

| Model | ROC-AUC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| Logistic Regression (balanced) | 0.8541 | 73.58% | 21.13% | 32.83% |
| **LightGBM (balanced)** ⭐ | **0.8630** | **74.90%** | **21.85%** | **33.82%** |
| XGBoost (balanced) | 0.8436 | 68.22% | 22.62% | 33.97% |

**Selected model**: LightGBM (balanced)
**Cross-validation**: 5-fold ROC-AUC = 0.8625 ± 0.0017 (highly stable)

### Key findings

- Dataset is imbalanced: 6.6% default rate. Accuracy alone is misleading (a naive "no default" model achieves 93% accuracy but 0% Recall).
- `class_weight='balanced'` improved Recall from 16% → 75% at the cost of Precision — appropriate when False Negatives are costly (missed defaulters).
- Feature engineering: 5 new features created. `income_per_dependent` ranked 4th in feature importance. `total_delinquency` showed 24× signal in default rates (0 delinquencies → 2.7%, 9 delinquencies → 67%).
- Counterintuitive insight: borrowers with `MonthlyIncome = 0` had **lower** default rate (3.9%) than those with reported income (6.6%) — likely a bank-side selection effect (students, dependents on family credit).

## 🛠 Tech Stack

- **Python 3.12**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — visualization
- **scikit-learn, LightGBM, XGBoost** — ML models
- **FastAPI, Pydantic, uvicorn** — REST API and validation
- **joblib** — model serialization
- **Render** — cloud deployment

## 📁 Project Structure

```
credit-default-prediction/
├── data/
│   ├── raw/                          # Original Kaggle data (not in repo)
│   └── processed/                    # Cleaned & featured datasets (not in repo)
├── notebooks/
│   ├── 01_eda.ipynb                  # Exploratory data analysis
│   ├── 02_data_cleaning.ipynb        # Data cleaning pipeline
│   ├── 03_feature_engineering.ipynb  # Feature engineering (5 new features)
│   └── 04_modeling.ipynb             # Model training, tuning, evaluation
├── src/
│   └── app.py                        # FastAPI web service (/predict endpoint)
├── models/                           # Trained artifacts
│   ├── lgbm_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── runtime.txt                       # Python version for deployment
├── requirements.txt
└── README.md
```

## 🚀 Setup (Local Development)

### 1. Clone repository

```bash
git clone https://github.com/Shakhriyor-git/credit-default-prediction.git
cd credit-default-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download `cs-training.csv` from [Kaggle: Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) and place it in `data/raw/`.

### 5. Reproduce results

Run notebooks in order:

```bash
jupyter notebook
```

1. `notebooks/01_eda.ipynb` — explore data
2. `notebooks/02_data_cleaning.ipynb` — clean and save processed data
3. `notebooks/03_feature_engineering.ipynb` — create features
4. `notebooks/04_modeling.ipynb` — train, evaluate, save models

### 6. Run API locally

```bash
uvicorn src.app:app --reload
```

Open `http://127.0.0.1:8000/docs` in your browser.

## ✅ Progress

- [x] Exploratory data analysis (EDA)
- [x] Data cleaning pipeline
- [x] Feature engineering (5 new features)
- [x] Model training (Logistic Regression, LightGBM, XGBoost)
- [x] Model evaluation with cross-validation
- [x] Threshold tuning and Precision/Recall analysis
- [x] Model serialization (joblib)
- [x] REST API (FastAPI + Pydantic validation)
- [x] **Cloud deployment (Render)** ← Live
- [ ] Hyperparameter tuning (Optuna)
- [ ] SHAP interpretability
- [ ] Docker containerization
- [ ] Monitoring (data/prediction drift)

## 📬 Contact

**Shakhriyor** — [GitHub](https://github.com/Shakhriyor-git) 
```