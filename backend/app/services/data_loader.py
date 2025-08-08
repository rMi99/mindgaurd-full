import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path: str) -> torch.Tensor:
    df = pd.read_csv(path)
    X = df.drop(columns=["label"]).values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return torch.tensor(X_scaled, dtype=torch.float32)
