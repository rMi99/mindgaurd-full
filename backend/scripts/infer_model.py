import os
import torch
import pandas as pd
from dotenv import load_dotenv
from app.models.model import RiskModel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# ─── Load environment and constants ────────────────────────────────────────────
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "./app/models/checkpoint.pt")
torch.serialization.add_safe_globals([StandardScaler])

# ─── Load scaler ───────────────────────────────────────────────────────────────
scaler = torch.load(MODEL_PATH.replace(".pt", "_scaler.pkl"), weights_only=False)

# ─── Load and preprocess data ──────────────────────────────────────────────────
data = pd.read_csv("data/dataset.csv")
X = data.drop(columns=["label"]).values
y_true = data["label"].values

X_scaled = scaler.transform(X)
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

# ─── Load model ────────────────────────────────────────────────────────────────
input_dim = X_tensor.shape[1]
output_dim = len(set(y_true))
model = RiskModel(input_dim, hidden_dim=64, output_dim=output_dim)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# ─── Predict ───────────────────────────────────────────────────────────────────
with torch.no_grad():
    preds = model(X_tensor).argmax(dim=1)

y_pred = preds.numpy()

# ─── Show predictions and metrics ──────────────────────────────────────────────
print("Predictions:", y_pred.tolist())
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ Accuracy: {accuracy * 100:.2f}%")
print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, digits=4))
