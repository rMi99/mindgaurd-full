# scripts/train_model.py

import os
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from dotenv import load_dotenv

from app.models.model import RiskModel

load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "./app/models/checkpoint.pt")

# ─── 1) LOAD & PREPROCESS ──────────────────────────────────────────────────────
df = pd.read_csv("data/dataset.csv")
X = df.drop(columns=["label"]).values
y = df["label"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# save scaler for inference
torch.save(scaler, MODEL_PATH.replace(".pt", "_scaler.pkl"))

X_train, X_val, y_train, y_val = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

train_x = torch.tensor(X_train, dtype=torch.float32)
train_y = torch.tensor(y_train, dtype=torch.long)
val_x   = torch.tensor(X_val,   dtype=torch.float32)
val_y   = torch.tensor(y_val,   dtype=torch.long)

# ─── 2) MODEL SETUP ────────────────────────────────────────────────────────────
input_dim  = train_x.shape[1]
num_classes = len(torch.unique(train_y))
model      = RiskModel(input_dim, hidden_dim=64, output_dim=num_classes)
criterion  = nn.CrossEntropyLoss()
optimizer  = torch.optim.Adam(model.parameters(), lr=1e-3)

# ─── 3) TRAIN LOOP ─────────────────────────────────────────────────────────────
epochs = 7
for epoch in range(1, epochs + 1):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_x)
    loss    = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch}/{epochs}   Loss: {loss.item():.4f}")

# ─── 4) EVALUATION ──────────────────────────────────────────────────────────────
model.eval()
with torch.no_grad():
    preds = model(val_x).argmax(dim=1)

acc = accuracy_score(val_y, preds)
print(f"\nValidation Accuracy: {acc * 100:.2f}%\n")
print(classification_report(val_y, preds, digits=4))

# ─── 5) SAVE CHECKPOINT ─────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)
print(f"✅ Model saved to {MODEL_PATH}")
