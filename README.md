# credit-default-prediction

Credit default risk prediction using LightGBM and XGBoost with SHAP interpretability.
End-to-end ML system for predicting credit default risk.

**Status**: In development

## Project structure
```bash
credit-default-prediction/
├── data/
│   ├── raw/                      # Original Kaggle data (not in repo)
│   └── processed/                # Cleaned & featured datasets (not in repo)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   ├── 02_data_cleaning.ipynb    # Data cleaning pipeline
│   ├── 03_feature_engineering.ipynb  # Feature engineering (5 new features)
│   └── 04_modeling.ipynb         # Model training, tuning, evaluation
├── src/
│   └── app.py                    # FastAPI web service (/predict endpoint)
├── models/                       # Trained models (.pkl, not in repo)
│   ├── lgbm_model.pkl
│   ├── scaler.pkl
│   └── feature_names.pkl
├── requirements.txt
└── README.md
```
## Setup

### 1. Clone repository

```bash
git clone https://github.com/Shakhriyor-git/credit-default-prediction.git
cd credit-default-prediction
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download dataset

Download `cs-training.csv` from [Kaggle: Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit/data) 
and place it in `data/raw/` folder.

### 5. Run notebooks

```bash
jupyter notebook
```

Open `notebooks/01_eda.ipynb` and run all cells.
Then `notebooks/02_data_cleaning.ipynb`.

## Tech stack

- **Python 3.12**
- **pandas, numpy** — data manipulation
- **matplotlib, seaborn** — visualization
- **scikit-learn, LightGBM, XGBoost** — ML models (coming soon)
- **SHAP** — model interpretability (coming soon)


## Results

| Model | ROC-AUC | Recall | Precision | F1 |
|-------|---------|--------|-----------|-----|
| Logistic Regression (balanced) | 0.8541 | 73.58% | 21.13% | 32.83% |
| LightGBM (balanced) | 0.8630 | 74.90% | 21.85% | 33.82% |
| XGBoost (balanced) | 0.8436 | 68.22% | 22.62% | 33.97% |

**Best model**: LightGBM (balanced)  
**Cross-validation**: 5-fold ROC-AUC = 0.8625 ± 0.0017 (stable)

### Key findings
- Accuracy misleading on imbalanced data (6.6% default rate)
- class_weight='balanced' improved Recall 16% → 75%
- Feature engineering: income_per_dependent ranked top-4 in importance

## Progress

- [x] Exploratory data analysis (EDA)
- [x] Data cleaning pipeline
- [X] Feature engineering
- [X] Model training
- [ ] Model evaluation
- [ ] API deployment (FastAPI)
- [ ] Docker containerization
- [ ] Cloud deployment