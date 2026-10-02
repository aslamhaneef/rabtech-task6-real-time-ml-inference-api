from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: float = Field(..., ge=0)
    workclass: str
    fnlwgt: float = Field(..., ge=0)
    education: str
    education_num: float = Field(..., ge=0)
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(..., ge=0)
    capital_loss: float = Field(..., ge=0)
    hours_per_week: float = Field(..., ge=0)
    native_country: str


class PredictionResponse(BaseModel):
    prediction: int
    probability_negative: float
    probability_positive: float