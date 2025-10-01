#!/usr/bin/env python3
"""
Audio Analysis Model Training Script
Uses Whisper + Librosa for audio emotion detection
"""

import os
import json
import numpy as np
import librosa
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioEmotionModel(nn.Module):
    def __init__(self, input_dim=13, hidden_dim=64, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def extract_audio_features(audio_path, sr=16000):
    """Extract MFCC features from audio file"""
    try:
        y, sr = librosa.load(audio_path, sr=sr)
        
        # Extract MFCC features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Extract additional features
        rms = librosa.feature.rms(y=y).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
        
        # Combine features
        features = np.concatenate([
            mfcc.mean(axis=1),  # 13 MFCC coefficients
            [rms, zcr, spectral_centroid]  # Additional features
        ])
        
        return features
    except Exception as e:
        logger.error(f"Error processing {audio_path}: {e}")
        return np.zeros(16)  # Return zeros if processing fails

def load_audio_dataset(data_dir):
    """Load audio dataset with emotion labels"""
    features = []
    labels = []
    
    emotion_map = {
        'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3,
        'surprise': 4, 'neutral': 5, 'disgust': 6
    }
    
    if not os.path.exists(data_dir):
        logger.warning(f"Data directory {data_dir} not found. Creating sample data...")
        # Create sample data for demonstration
        for emotion, label in emotion_map.items():
            sample_features = np.random.randn(16) + label * 0.5
            features.append(sample_features)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
            
        label = emotion_map.get(emotion, 5)  # Default to neutral
        
        for audio_file in os.listdir(emotion_path):
            if audio_file.endswith(('.wav', '.mp3', '.m4a')):
                audio_path = os.path.join(emotion_path, audio_file)
                feature = extract_audio_features(audio_path)
                features.append(feature)
                labels.append(label)
    
    return np.array(features), np.array(labels)

def train_audio_model():
    """Train audio emotion detection model"""
    logger.info("🎵 Starting audio emotion model training...")
    
    # Load dataset
    data_dir = "data/audio"
    X, y = load_audio_dataset(data_dir)
    
    if len(X) == 0:
        logger.error("No audio data found!")
        return
    
    logger.info(f"Loaded {len(X)} audio samples with {len(np.unique(y))} emotion classes")
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Initialize model
    model = AudioEmotionModel(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    epochs = 50
    best_val_loss = float('inf')
    patience = 10
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'scaler': scaler,
                'input_dim': X_train.shape[1],
                'emotion_classes': ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'disgust']
            }, "models/audio_emotion_model.pt")
        else:
            patience_counter += 1
            if patience_counter > patience:
                logger.info("Early stopping triggered")
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_val_tensor).argmax(dim=1)
        accuracy = (predictions == y_val_tensor).float().mean()
        logger.info(f"Final validation accuracy: {accuracy:.4f}")
        
        # Classification report
        emotion_classes = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'disgust']
        report = classification_report(y_val, predictions.numpy(), target_names=emotion_classes)
        logger.info(f"Classification Report:\n{report}")
    
    logger.info("✅ Audio emotion model training completed!")

if __name__ == "__main__":
    train_audio_model()

