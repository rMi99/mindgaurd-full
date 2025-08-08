import pickle
import torch
from typing import Tuple, List
from app.models.model import RiskModel

_model: RiskModel = None
_scaler = None
_risk_map = {0: "low", 1: "moderate", 2: "high"}

def load_model_and_scaler(
    model_path: str = "./app/models/checkpoint.pt", 
    scaler_path: str = "./app/models/checkpoint_scaler.pkl"
):
    global _model, _scaler
    with open(scaler_path, "rb") as f:
        _scaler = pickle.load(f)

    input_dim = _scaler.mean_.shape[0]
    output_dim = len(_risk_map)
    _model = RiskModel(input_dim, hidden_dim=64, output_dim=output_dim)
    _model.load_state_dict(torch.load(model_path))
    _model.eval()

def predict_risk(features: List[float]) -> Tuple[str, float]:
    import numpy as np
    x = _scaler.transform([features])
    x_tensor = torch.tensor(x, dtype=torch.float32)
    with torch.no_grad():
        logits = _model(x_tensor)
        probs = torch.softmax(logits, dim=1).numpy()[0]
    idx = int(probs.argmax())
    return _risk_map[idx], float(probs[idx])
