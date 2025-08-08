from typing import List
from fastapi import APIRouter
from app.services.db import db
from app.schemas.schemas import PredictionHistoryItem

router = APIRouter()

@router.get("/predictions/{user_id}", response_model=List[PredictionHistoryItem])
async def get_predictions(user_id: str):
    cursor = db.predictions.find({"user_id": user_id}, {'_id':0}).sort("timestamp", -1)
    return await cursor.to_list(length=100)
