#!/usr/bin/env python3
"""
Enhanced Facial Analysis Model Training Script

This script trains improved models for facial expression recognition,
sleepiness detection, and PHQ-9 estimation using various datasets.
"""

import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

# Add the app directory to the path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision import models
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from PIL import Image
import joblib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FacialExpressionDataset(Dataset):
    """Dataset for facial expression recognition."""
    
    def __init__(self, image_paths: List[str], labels: List[int], transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

class EnhancedFacialModel(nn.Module):
    """Enhanced facial analysis model with multiple outputs."""
    
    def __init__(self, num_emotions: int = 7, num_sleepiness_levels: int = 3):
        super(EnhancedFacialModel, self).__init__()
        
        # Base feature extractor (ResNet-18)
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Identity()  # Remove final classification layer
        
        # Feature dimension
        feature_dim = 512
        
        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_emotions)
        )
        
        # Sleepiness detection head
        self.sleepiness_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_sleepiness_levels)
        )
        
        # PHQ-9 regression head
        self.phq9_head = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output between 0 and 1, scale to 0-27
        )
        
        # Stress level head
        self.stress_head = nn.Sequential(
            nn.Linear(feature_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 3)  # Low, Medium, High
        )
    
    def forward(self, x):
        # Extract features
        features = self.backbone(x)
        
        # Multiple outputs
        emotion_output = self.emotion_head(features)
        sleepiness_output = self.sleepiness_head(features)
        phq9_output = self.phq9_head(features) * 27  # Scale to 0-27
        stress_output = self.stress_head(features)
        
        return {
            'emotion': emotion_output,
            'sleepiness': sleepiness_output,
            'phq9': phq9_output,
            'stress': stress_output
        }

class FacialModelTrainer:
    """Trainer for enhanced facial analysis models."""
    
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.model = None
        self.optimizer = None
        self.criterion = None
        
        # Emotion labels
        self.emotion_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
        self.sleepiness_labels = ['alert', 'slightly_tired', 'very_tired']
        self.stress_labels = ['low', 'medium', 'high']
        
        # Data transforms
        self.train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def setup_model(self):
        """Initialize the model and optimizer."""
        self.model = EnhancedFacialModel(
            num_emotions=len(self.emotion_labels),
            num_sleepiness_levels=len(self.sleepiness_labels)
        ).to(self.device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        # Multi-task loss weights
        self.loss_weights = {
            'emotion': 1.0,
            'sleepiness': 0.8,
            'phq9': 0.6,
            'stress': 0.7
        }
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.loss_weights.keys()}
        
        for batch_idx, (images, labels) in enumerate(dataloader):
            images = images.to(self.device)
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate losses (simplified - in practice you'd have proper labels)
            losses = {}
            total_batch_loss = 0.0
            
            for task, weight in self.loss_weights.items():
                if task == 'phq9':
                    # PHQ-9 regression loss
                    loss = nn.MSELoss()(outputs[task].squeeze(), torch.zeros(images.size(0)).to(self.device))
                else:
                    # Classification losses
                    loss = nn.CrossEntropyLoss()(outputs[task], torch.zeros(images.size(0), dtype=torch.long).to(self.device))
                
                losses[task] = loss
                total_batch_loss += weight * loss
                task_losses[task] += loss.item()
            
            # Backward pass
            self.optimizer.zero_grad()
            total_batch_loss.backward()
            self.optimizer.step()
            
            total_loss += total_batch_loss.item()
            
            if batch_idx % 10 == 0:
                logger.info(f'Batch {batch_idx}/{len(dataloader)}, Loss: {total_batch_loss.item():.4f}')
        
        # Average losses
        avg_losses = {task: loss / len(dataloader) for task, loss in task_losses.items()}
        avg_losses['total'] = total_loss / len(dataloader)
        
        return avg_losses
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        task_losses = {task: 0.0 for task in self.loss_weights.keys()}
        
        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                outputs = self.model(images)
                
                # Calculate losses (simplified)
                for task, weight in self.loss_weights.items():
                    if task == 'phq9':
                        loss = nn.MSELoss()(outputs[task].squeeze(), torch.zeros(images.size(0)).to(self.device))
                    else:
                        loss = nn.CrossEntropyLoss()(outputs[task], torch.zeros(images.size(0), dtype=torch.long).to(self.device))
                    
                    task_losses[task] += loss.item()
                    total_loss += weight * loss.item()
        
        avg_losses = {task: loss / len(dataloader) for task, loss in task_losses.items()}
        avg_losses['total'] = total_loss / len(dataloader)
        
        return avg_losses
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, 
              epochs: int = 50, save_path: str = None):
        """Train the model."""
        logger.info(f"Starting training for {epochs} epochs on {self.device}")
        
        best_val_loss = float('inf')
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            logger.info(f"Epoch {epoch+1}/{epochs}")
            
            # Train
            train_loss = self.train_epoch(train_loader)
            train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate(val_loader)
            val_losses.append(val_loss)
            
            logger.info(f"Train Loss: {train_loss['total']:.4f}, Val Loss: {val_loss['total']:.4f}")
            
            # Save best model
            if val_loss['total'] < best_val_loss:
                best_val_loss = val_loss['total']
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model with val loss: {best_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def save_model(self, path: str):
        """Save the trained model."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'emotion_labels': self.emotion_labels,
            'sleepiness_labels': self.sleepiness_labels,
            'stress_labels': self.stress_labels
        }, path)
        
        logger.info(f"Model saved to {path}")
    
    def load_model(self, path: str):
        """Load a trained model."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        logger.info(f"Model loaded from {path}")

def create_synthetic_dataset(num_samples: int = 1000) -> Tuple[List[str], List[int]]:
    """Create a synthetic dataset for training (in practice, use real data)."""
    logger.info(f"Creating synthetic dataset with {num_samples} samples")
    
    # Create synthetic image paths and labels
    image_paths = []
    labels = []
    
    for i in range(num_samples):
        # Generate synthetic image path
        image_path = f"synthetic_image_{i:04d}.jpg"
        image_paths.append(image_path)
        
        # Random label for emotion (0-6)
        labels.append(i % 7)
    
    return image_paths, labels

def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train enhanced facial analysis model')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--save_path', type=str, default='models/enhanced_facial_model.pt', 
                       help='Path to save the trained model')
    parser.add_argument('--data_path', type=str, default='data/facial_expressions', 
                       help='Path to training data')
    parser.add_argument('--num_samples', type=int, default=1000, 
                       help='Number of synthetic samples to generate')
    
    args = parser.parse_args()
    
    # Initialize trainer
    trainer = FacialModelTrainer()
    trainer.setup_model()
    
    # Create synthetic dataset (in practice, load real data)
    logger.info("Creating synthetic dataset...")
    image_paths, labels = create_synthetic_dataset(args.num_samples)
    
    # Split data
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        image_paths, labels, test_size=0.2, random_state=42
    )
    
    # Create datasets
    train_dataset = FacialExpressionDataset(train_paths, train_labels, trainer.train_transform)
    val_dataset = FacialExpressionDataset(val_paths, val_labels, trainer.val_transform)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    
    logger.info(f"Training samples: {len(train_dataset)}")
    logger.info(f"Validation samples: {len(val_dataset)}")
    
    # Train the model
    train_losses, val_losses = trainer.train(
        train_loader, val_loader, 
        epochs=args.epochs, 
        save_path=args.save_path
    )
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'epochs': args.epochs,
        'batch_size': args.batch_size
    }
    
    history_path = args.save_path.replace('.pt', '_history.joblib')
    joblib.dump(history, history_path)
    logger.info(f"Training history saved to {history_path}")
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()