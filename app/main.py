from pathlib import Path
import sys

import joblib
import pandas as pd
from fastapi import FastAPI

from app.model_features import AdultFeatureEngineer
from app.schemas import PredictionRequest, PredictionResponse


# The original Task 4 model was serialized with AdultFeatureEngineer
# recorded under __main__. Make that class available during loading.
setattr(sys.modules["__main__"], "AdultFeatureEngineer", AdultFeatureEngineer)


BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "model" / "champion_model.joblib"

model = joblib.load(MODEL_PATH)

app = FastAPI(
    title="RabTech Task 6 - Real-Time ML Inference API",
    description="FastAPI service for the Task 4 champion Adult Income classification model.",
    version="1.0.0",
)


@app.get("/")
def root():
    return {
        "message": "RabTech Task 6 ML Inference API is running",
        "endpoint": "/predict",
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    data = request.model_dump()

    # Convert API field names back to the exact raw Adult dataset columns
    row = {
        "age": data["age"],
        "workclass": data["workclass"],
        "fnlwgt": data["fnlwgt"],
        "education": data["education"],
        "education-num": data["education_num"],
        "marital-status": data["marital_status"],
        "occupation": data["occupation"],
        "relationship": data["relationship"],
        "race": data["race"],
        "sex": data["sex"],
        "capital-gain": data["capital_gain"],
        "capital-loss": data["capital_loss"],
        "hours-per-week": data["hours_per_week"],
        "native-country": data["native_country"],
    }

    input_df = pd.DataFrame([row])

    probabilities = model.predict_proba(input_df)[0]
    classes = list(model.classes_)

    probability_negative = float(probabilities[classes.index(0)])
    probability_positive = float(probabilities[classes.index(1)])

    prediction = int(model.predict(input_df)[0])

    return PredictionResponse(
        prediction=prediction,
        probability_negative=probability_negative,
        probability_positive=probability_positive,
    )