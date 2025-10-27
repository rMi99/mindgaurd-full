etecte#!/usr/bin/env python3
"""
Multimodal Model Training Script
Combines audio and facial features for enhanced emotion detection
"""

import os
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MultimodalEmotionModel(nn.Module):
    def __init__(self, audio_dim=16, facial_dim=468*3, hidden_dim=512, output_dim=7):
        super().__init__()
        
        # Audio branch
        self.audio_branch = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Facial branch
        self.facial_branch = nn.Sequential(
            nn.Linear(facial_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64)
        )
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(128, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=64, num_heads=8, batch_first=True)
    
    def forward(self, audio_features, facial_features):
        # Process audio features
        audio_out = self.audio_branch(audio_features)
        
        # Process facial features
        facial_out = self.facial_branch(facial_features)
        
        # Apply attention
        combined = torch.stack([audio_out, facial_out], dim=1)  # [batch, 2, 64]
        attended, _ = self.attention(combined, combined, combined)
        
        # Flatten and fuse
        fused = attended.view(attended.size(0), -1)  # [batch, 128]
        
        # Final classification
        output = self.fusion(fused)
        
        return output

def load_multimodal_dataset(audio_dir, facial_dir):
    """Load multimodal dataset with both audio and facial features"""
    features_audio = []
    features_facial = []
    labels = []
    
    emotion_map = {
        'happy': 0, 'sad': 1, 'angry': 2, 'fear': 3,
        'surprise': 4, 'neutral': 5, 'disgust': 6
    }
    
    # Check if directories exist
    if not os.path.exists(audio_dir) or not os.path.exists(facial_dir):
        logger.warning("Data directories not found. Creating sample multimodal data...")
        # Create sample data for demonstration
        for emotion, label in emotion_map.items():
            audio_features = np.random.randn(16) + label * 0.5
            facial_features = np.random.randn(468 * 3) + label * 0.1
            
            features_audio.append(audio_features)
            features_facial.append(facial_features)
            labels.append(label)
        
        return np.array(features_audio), np.array(features_facial), np.array(labels)
    
    # Load actual data (placeholder implementation)
    # In practice, you'd match audio and facial samples by timestamp or session ID
    for emotion in emotion_map.keys():
        audio_emotion_dir = os.path.join(audio_dir, emotion)
        facial_emotion_dir = os.path.join(facial_dir, emotion)
        
        if os.path.exists(audio_emotion_dir) and os.path.exists(facial_emotion_dir):
            label = emotion_map[emotion]
            
            # Get matching files (simplified - in practice you'd match by filename/timestamp)
            audio_files = [f for f in os.listdir(audio_emotion_dir) if f.endswith(('.wav', '.mp3'))]
            facial_files = [f for f in os.listdir(facial_emotion_dir) if f.endswith(('.jpg', '.png'))]
            
            # Match files (simplified matching)
            min_files = min(len(audio_files), len(facial_files))
            
            for i in range(min_files):
                # Generate sample features (replace with actual feature extraction)
                audio_features = np.random.randn(16) + label * 0.5
                facial_features = np.random.randn(468 * 3) + label * 0.1
                
                features_audio.append(audio_features)
                features_facial.append(facial_features)
                labels.append(label)
    
    return np.array(features_audio), np.array(features_facial), np.array(labels)

def train_multimodal_model():
    """Train multimodal emotion detection model"""
    logger.info("🔮 Starting multimodal emotion model training...")
    
    # Load dataset
    audio_dir = "data/multimodal/audio"
    facial_dir = "data/multimodal/facial"
    X_audio, X_facial, y = load_multimodal_dataset(audio_dir, facial_dir)
    
    if len(X_audio) == 0:
        logger.error("No multimodal data found!")
        return
    
    logger.info(f"Loaded {len(X_audio)} multimodal samples with {len(np.unique(y))} emotion classes")
    
    # Split data
    indices = np.arange(len(X_audio))
    train_idx, val_idx = train_test_split(indices, test_size=0.2, random_state=42, stratify=y)
    
    X_audio_train, X_audio_val = X_audio[train_idx], X_audio[val_idx]
    X_facial_train, X_facial_val = X_facial[train_idx], X_facial[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    # Scale features
    audio_scaler = StandardScaler()
    facial_scaler = StandardScaler()
    
    X_audio_train_scaled = audio_scaler.fit_transform(X_audio_train)
    X_audio_val_scaled = audio_scaler.transform(X_audio_val)
    
    X_facial_train_scaled = facial_scaler.fit_transform(X_facial_train)
    X_facial_val_scaled = facial_scaler.transform(X_facial_val)
    
    # Convert to tensors
    X_audio_train_tensor = torch.tensor(X_audio_train_scaled, dtype=torch.float32)
    X_facial_train_tensor = torch.tensor(X_facial_train_scaled, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
    
    X_audio_val_tensor = torch.tensor(X_audio_val_scaled, dtype=torch.float32)
    X_facial_val_tensor = torch.tensor(X_facial_val_scaled, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.long)
    
    # Initialize model
    model = MultimodalEmotionModel()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    epochs = 150
    best_val_loss = float('inf')
    patience = 20
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        optimizer.zero_grad()
        outputs = model(X_audio_train_tensor, X_facial_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_audio_val_tensor, X_facial_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).item()
        
        scheduler.step(val_loss)
        
        if epoch % 25 == 0:
            logger.info(f"Epoch {epoch}/{epochs}, Loss: {loss.item():.4f}, Val Loss: {val_loss:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            os.makedirs("models", exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'audio_scaler': audio_scaler,
                'facial_scaler': facial_scaler,
                'audio_dim': X_audio.shape[1],
                'facial_dim': X_facial.shape[1],
                'emotion_classes': ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'disgust']
            }, "models/multimodal_emotion_model.pt")
        else:
            patience_counter += 1
            if patience_counter > patience:
                logger.info("Early stopping triggered")
                break
    
    # Final evaluation
    model.eval()
    with torch.no_grad():
        predictions = model(X_audio_val_tensor, X_facial_val_tensor).argmax(dim=1)
        accuracy = (predictions == y_val_tensor).float().mean()
        logger.info(f"Final validation accuracy: {accuracy:.4f}")
        
        # Classification report
        emotion_classes = ['happy', 'sad', 'angry', 'fear', 'surprise', 'neutral', 'disgust']
        report = classification_report(y_val, predictions.numpy(), target_names=emotion_classes)
        logger.info(f"Classification Report:\n{report}")
    
    logger.info("✅ Multimodal emotion model training completed!")

if __name__ == "__main__":
    train_multimodal_model()

