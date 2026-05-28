"""
Credit Default Prediction API
FastAPI bilan model'ni web service qilish
"""

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
import os

# ============================================
# 1. Model va artifactlarni yuklash
# ============================================

# Yo'l - src/ dan models/ ga
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

model = joblib.load(os.path.join(MODELS_DIR, 'lgbm_model.pkl'))
scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
feature_names = joblib.load(os.path.join(MODELS_DIR, 'feature_names.pkl'))

# ============================================
# 2. FastAPI ilovasi
# ============================================

app = FastAPI(
    title="Credit Default Prediction API",
    description="Predicts credit default risk using LightGBM",
    version="1.0"
)

# ============================================
# 3. Input schema (Pydantic)
# ============================================

class CreditApplication(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float
    age: int
    NumberOfTime30_59DaysPastDueNotWorse: int
    DebtRatio: float
    MonthlyIncome: float
    NumberOfOpenCreditLinesAndLoans: int
    NumberOfTimes90DaysLate: int
    NumberRealEstateLoansOrLines: int
    NumberOfTime60_89DaysPastDueNotWorse: int
    NumberOfDependents: int
    income_missing_flag: int
    total_delinquency: int
    has_dependents: int
    income_per_dependent: float
    zero_income_flag: int

# ============================================
# 4. Endpoints
# ============================================

@app.get("/")
def home():
    return {"message": "Credit Default Prediction API", "status": "running"}


@app.post("/predict")
def predict(application: CreditApplication):
    # Input'ni dict'ga, keyin DataFrame'ga
    data = application.dict()
    
    # Feature nomlarini to'g'rilash (Pydantic _ ishlatadi, original - bilan)
    input_dict = {
        'RevolvingUtilizationOfUnsecuredLines': data['RevolvingUtilizationOfUnsecuredLines'],
        'age': data['age'],
        'NumberOfTime30-59DaysPastDueNotWorse': data['NumberOfTime30_59DaysPastDueNotWorse'],
        'DebtRatio': data['DebtRatio'],
        'MonthlyIncome': data['MonthlyIncome'],
        'NumberOfOpenCreditLinesAndLoans': data['NumberOfOpenCreditLinesAndLoans'],
        'NumberOfTimes90DaysLate': data['NumberOfTimes90DaysLate'],
        'NumberRealEstateLoansOrLines': data['NumberRealEstateLoansOrLines'],
        'NumberOfTime60-89DaysPastDueNotWorse': data['NumberOfTime60_89DaysPastDueNotWorse'],
        'NumberOfDependents': data['NumberOfDependents'],
        'income_missing_flag': data['income_missing_flag'],
        'total_delinquency': data['total_delinquency'],
        'has_dependents': data['has_dependents'],
        'income_per_dependent': data['income_per_dependent'],
        'zero_income_flag': data['zero_income_flag'],
    }
    
    # DataFrame yaratish (feature tartibida)
    df = pd.DataFrame([input_dict])[feature_names]
    
    # Scale qilish
    df_scaled = scaler.transform(df)
    
    # Bashorat
    probability = float(model.predict_proba(df_scaled)[0, 1])
    prediction = int(model.predict(df_scaled)[0])
    
    return {
        "default_probability": round(probability, 4),
        "default_probability_percent": f"{probability*100:.2f}%",
        "prediction": "DEFAULT" if prediction == 1 else "GOOD",
        "risk_level": "HIGH" if probability > 0.5 else "MEDIUM" if probability > 0.2 else "LOW"
    }