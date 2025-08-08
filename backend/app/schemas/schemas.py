from pydantic import BaseModel
from typing import List, Optional

class PredictRequest(BaseModel):
    user_id: Optional[str]
    features: List[float]

class PredictResponse(BaseModel):
    risk_level: str
    scores: List[float]

class ProfileResponse(BaseModel):
    user_id: str
    age: int
    gender: str
    language: str

class PredictionHistoryItem(BaseModel):
    timestamp: str
    risk_level: str
    scores: List[float]
