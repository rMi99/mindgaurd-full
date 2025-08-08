from fastapi import APIRouter, HTTPException
from datetime import datetime
import torch
import os
from app.schemas.schemas import PredictRequest, PredictResponse
from app.services.db import db
from app.models.model import RiskModel
from dotenv import load_dotenv

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "./app/models/checkpoint.pt")
# Convert to absolute path
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), MODEL_PATH.lstrip('./'))

device = torch.device("cpu")
model = RiskModel(input_dim=10, hidden_dim=64, output_dim=3)  # Changed from 3 to 10 to match checkpoint
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

router = APIRouter()

@router.post("/predict", response_model=PredictResponse)
async def predict(req: PredictRequest):
    x = torch.tensor([req.features], dtype=torch.float32)
    with torch.no_grad():
        scores = model(x).squeeze().tolist()
    idx = int(torch.argmax(torch.tensor(scores)))
    labels = ["low", "moderate", "high"]
    risk = labels[idx]
    await db.predictions.insert_one({
        "user_id": req.user_id or "anonymous",
        "timestamp": datetime.utcnow().isoformat(),
        "risk_level": risk,
        "scores": scores,
    })
    return {"risk_level": risk, "scores": scores}
