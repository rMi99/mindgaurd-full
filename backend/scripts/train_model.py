import os
import pandas as pd
import numpy as np
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import pickle

# === 1. Feature Engineering ===
def engineer_features(df):
    df_eng = df.copy()
    df_eng['age_sleep_ratio'] = df['age'] / (df['sleep_hours'] + 1e-6)
    df_eng['age_stress_interaction'] = df['age'] * df['stress_level']
    df_eng['age_group'] = pd.cut(df['age'], bins=[0, 30, 45, 100], labels=['Young', 'Middle', 'Senior'])
    df_eng['work_life_balance'] = df['work_hours'] / (df['social_support'] + 1e-6)
    df_eng['stress_support_ratio'] = df['stress_level'] / (df['social_support'] + 1e-6)
    df_eng['exercise_sleep_score'] = df['exercise_freq'] * df['sleep_hours']
    df_eng['log_income'] = np.log(df['income_level'] + 1)
    df_eng['lifestyle_score'] = (df['sleep_hours'] * df['exercise_freq']) / (df['stress_level'] + 1)
    df_eng['mental_health_risk'] = (df['therapy_history'] + df['family_history'] + (df['stress_level'] > 7).astype(int))
    age_group_encoded = pd.get_dummies(df_eng['age_group'], prefix='age_group')
    df_eng = pd.concat([df_eng.drop('age_group', axis=1), age_group_encoded], axis=1)
    return df_eng

# === 2. Data Augmentation ===
def augment_data(df, target_col='label', augmentation_factor=15):
    augmented_data = []
    for class_label in df[target_col].unique():
        class_data = df[df[target_col] == class_label].drop(columns=[target_col])
        means = class_data.mean()
        stds = class_data.std().fillna(0.1)
        mins = class_data.min()
        maxs = class_data.max()
        n_synthetic = len(class_data) * augmentation_factor
        for _ in range(n_synthetic):
            base_sample = class_data.sample(1).iloc[0]
            synthetic_sample = {}
            for col in class_data.columns:
                if col in ['therapy_history', 'family_history']:
                    synthetic_sample[col] = np.random.choice([0, 1], p=[0.6, 0.4])
                elif col in ['stress_level', 'social_support', 'exercise_freq', 'education_level']:
                    noise = np.random.normal(0, stds[col] * 0.3)
                    new_val = base_sample[col] + noise
                    synthetic_sample[col] = np.clip(new_val, mins[col], maxs[col])
                elif 'age_group' in col:
                    synthetic_sample[col] = base_sample[col]
                else:
                    noise = np.random.normal(0, stds[col] * 0.5)
                    new_val = base_sample[col] + noise
                    synthetic_sample[col] = np.clip(new_val, mins[col], maxs[col])
            synthetic_sample[target_col] = class_label
            augmented_data.append(synthetic_sample)
    original_data = df.to_dict('records')
    return pd.DataFrame(original_data + augmented_data)

# === 3. Optimized Model Definition ===
class OptimizedRiskModel(nn.Module):
    def __init__(self, input_dim, num_classes, dropout_rate=0.3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(32, num_classes)
        )
        self.apply(self._init_weights)
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)
    def forward(self, x):
        return self.network(x)

# === 4. Training Pipeline ===
def train_optimized_model(df, model_path="optimized_model.pt", scaler_path="optimized_scaler.pkl"):
    # Feature engineering & augmentation
    df_engineered = engineer_features(df)
    df_augmented = augment_data(df_engineered)
    X = df_augmented.drop(columns=['label'])
    y = df_augmented['label'].values

    # Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Train/val split
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    train_x = torch.tensor(X_train, dtype=torch.float32)
    train_y = torch.tensor(y_train, dtype=torch.long)
    val_x = torch.tensor(X_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.long)

    # Model setup
    input_dim = train_x.shape[1]
    num_classes = len(np.unique(y))
    model = OptimizedRiskModel(input_dim, num_classes, dropout_rate=0.3)

    # Class weights for imbalance
    class_weights = compute_class_weight('balanced', classes=np.unique(y), y=y)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)


    # Training loop with early stopping
    epochs = 1000
    best_val_acc = 0
    patience_counter = 0
    patience_limit = 50
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        outputs = model(train_x)
        loss = criterion(outputs, train_y)
        loss.backward()
        optimizer.step()
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(val_x)
            val_loss = criterion(val_outputs, val_y)
            preds = val_outputs.argmax(dim=1)
        acc = accuracy_score(val_y.numpy(), preds.numpy())
        scheduler.step(val_loss)
        if acc > best_val_acc:
            best_val_acc = acc
            patience_counter = 0
            torch.save(model.state_dict(), model_path)
        else:
            patience_counter += 1
        if patience_counter >= patience_limit:
            print(f"Early stopping at epoch {epoch}")
            break
        if epoch % 50 == 0:
            print(f"Epoch {epoch:4d} | Loss: {loss.item():.4f} | Val Acc: {acc:.4f}")
    print(f"Training completed. Best validation accuracy: {best_val_acc:.4f}")
    return model, scaler, best_val_acc

# === 5. Usage Example ===
if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    model, scaler, accuracy = train_optimized_model(df)
    print(f"Optimized model accuracy: {accuracy:.4f}")
