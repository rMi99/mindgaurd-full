#!/usr/bin/env python3
"""
Facial Expression Model Training Script
Uses MediaPipe + OpenCV for facial landmark detection and emotion classification
"""

import os
import json
import numpy as np
import cv2
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import mediapipe as mp
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialEmotionModel(nn.Module):
    def __init__(self, input_dim=468*3, hidden_dim=256, output_dim=7):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 4, output_dim)
        )
    
    def forward(self, x):
        return self.net(x)

def extract_facial_landmarks(image_path):
    """Extract facial landmarks using MediaPipe"""
    try:
        # Initialize MediaPipe Face Mesh
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return np.zeros(468 * 3)  # Return zeros if image can't be loaded
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = face_mesh.process(image_rgb)
        
        if results.multi_face_landmarks:
            # Get first face landmarks
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract landmark coordinates (x, y, z)
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            return np.array(landmarks)
        else:
            return np.zeros(468 * 3)  # Return zeros if no face detected
            
    except Exception as e:
        logger.error(f"Error processing {image_path}: {e}")
        return np.zeros(468 * 3)

def load_facial_dataset(data_dir):
    """Load facial expression dataset with emotion labels"""
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
            sample_features = np.random.randn(468 * 3) + label * 0.1
            features.append(sample_features)
            labels.append(label)
        
        return np.array(features), np.array(labels)
    
    for emotion in os.listdir(data_dir):
        emotion_path = os.path.join(data_dir, emotion)
        if not os.path.isdir(emotion_path):
            continue
            
        label = emotion_map.get(emotion, 5)  # Default to neutral
        
        for image_file in os.listdir(emotion_path):
            if image_file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                image_path = os.path.join(emotion_path, image_file)
                landmarks = extract_facial_landmarks(image_path)
                features.append(landmarks)
                labels.append(label)
    
    return np.array(features), np.array(labels)

def train_facial_model():
    """Train facial emotion detection model"""
    logger.info("👁️ Starting facial emotion model training...")
    
    # Load dataset
    data_dir = "data/facial"
    X, y = load_facial_dataset(data_dir)
    
    if len(X) == 0:
        logger.error("No facial data found!")
        return
    
    logger.info(f"Loaded {len(X)} facial samples with {len(np.unique(y))} emotion classes")
    
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
    model = FacialEmotionModel(input_dim=X_train.shape[1])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training loop
    epochs = 100
    best_val_loss = float('inf')
    patience = 15
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
        
        if epoch % 20 == 0:
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
            }, "models/facial_emotion_model.pt")
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
    
    logger.info("✅ Facial emotion model training completed!")

if __name__ == "__main__":
    train_facial_model()

