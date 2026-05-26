# credit-default-prediction

Credit default risk prediction using LightGBM and XGBoost with SHAP interpretability.
End-to-end ML system for predicting credit default risk.

**Status**: In development

## Project structure
```bash
credit-default-prediction/
├── data/
│   ├── raw/              # Original Kaggle data (not in repo)
│   └── processed/        # Cleaned datasets (not in repo)
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory data analysis
│   └── 02_data_cleaning.ipynb    # Data cleaning pipeline
├── src/                  # Production code (coming soon)
├── models/               # Trained models (coming soon)
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

## Progress

- [x] Exploratory data analysis (EDA)
- [x] Data cleaning pipeline
- [ ] Feature engineering
- [ ] Model training
- [ ] Model evaluation
- [ ] API deployment (FastAPI)
- [ ] Docker containerization
- [ ] Cloud deployment