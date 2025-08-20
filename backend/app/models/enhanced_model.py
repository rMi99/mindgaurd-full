import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import os
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class EnhancedHealthModel:
    """
    Enhanced health risk prediction model using GradientBoostingClassifier.
    Provides comprehensive health risk assessment with personalized insights.
    """
    
    def __init__(self, model_path: str = "models/health_model.pkl"):
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.feature_names = [
            'age', 'gender', 'sleep_hours', 'exercise_frequency', 'stress_level',
            'diet_quality', 'social_connections', 'work_life_balance',
            'mental_health_history', 'physical_health_history', 'substance_use',
            'family_history', 'financial_stress', 'relationship_status'
        ]
        
    def prepare_features(self, data: Dict) -> np.ndarray:
        """
        Prepare and normalize features for prediction.
        
        Args:
            data: Dictionary containing user health data
            
        Returns:
            Normalized feature array
        """
        try:
            # Extract features in the correct order
            features = []
            for feature in self.feature_names:
                if feature in data:
                    features.append(float(data[feature]))
                else:
                    # Default value for missing features
                    features.append(0.0)
            
            features_array = np.array(features).reshape(1, -1)
            return self.scaler.transform(features_array)
            
        except Exception as e:
            logger.error(f"Error preparing features: {e}")
            raise
    
    def train(self, training_data: List[Dict], labels: List[int]) -> Dict:
        """
        Train the GradientBoostingClassifier model.
        
        Args:
            training_data: List of dictionaries containing health data
            labels: List of risk labels (0: Low, 1: Normal, 2: High)
            
        Returns:
            Dictionary containing training metrics
        """
        try:
            # Prepare training data
            X = []
            for data_point in training_data:
                features = []
                for feature in self.feature_names:
                    if feature in data_point:
                        features.append(float(data_point[feature]))
                    else:
                        features.append(0.0)
                X.append(features)
            
            X = np.array(X)
            y = np.array(labels)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize and train model
            self.model = GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            report = classification_report(y_test, y_pred, output_dict=True)
            
            metrics = {
                'accuracy': accuracy,
                'precision': report['weighted avg']['precision'],
                'recall': report['weighted avg']['recall'],
                'f1_score': report['weighted avg']['f1-score']
            }
            
            logger.info(f"Model training completed. Accuracy: {accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise
    
    def predict(self, health_data: Dict) -> Dict:
        """
        Predict health risk level and provide confidence scores.
        
        Args:
            health_data: Dictionary containing user health data
            
        Returns:
            Dictionary with prediction results and confidence scores
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            # Prepare features
            features = self.prepare_features(health_data)
            
            # Make prediction
            prediction = self.model.predict(features)[0]
            probabilities = self.model.predict_proba(features)[0]
            
            # Map prediction to risk level
            risk_levels = {0: 'Low', 1: 'Normal', 2: 'High'}
            risk_level = risk_levels[prediction]
            
            # Get confidence score
            confidence = probabilities[prediction]
            
            # Risk interpretation
            risk_interpretation = {
                'Low': 'Minimal health risks detected. Continue maintaining healthy habits.',
                'Normal': 'Moderate health risks. Some areas could use improvement.',
                'High': 'Elevated health risks detected. Consider professional consultation.'
            }
            
            result = {
                'risk_level': risk_level,
                'confidence': float(confidence),
                'probabilities': {
                    'low': float(probabilities[0]),
                    'normal': float(probabilities[1]),
                    'high': float(probabilities[2])
                },
                'interpretation': risk_interpretation[risk_level],
                'timestamp': pd.Timestamp.now().isoformat()
            }
            
            logger.info(f"Prediction made: {risk_level} risk with {confidence:.2f} confidence")
            return result
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def save_model(self, filepath: Optional[str] = None) -> str:
        """
        Save the trained model and scaler to disk.
        
        Args:
            filepath: Optional custom path for saving
            
        Returns:
            Path where model was saved
        """
        try:
            if self.model is None:
                raise ValueError("No trained model to save.")
            
            save_path = filepath or self.model_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save model and scaler
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names
            }
            
            joblib.dump(model_data, save_path)
            logger.info(f"Model saved successfully to {save_path}")
            return save_path
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise
    
    def load_model(self, filepath: Optional[str] = None) -> bool:
        """
        Load a trained model and scaler from disk.
        
        Args:
            filepath: Optional custom path for loading
            
        Returns:
            True if model loaded successfully
        """
        try:
            load_path = filepath or self.model_path
            
            if not os.path.exists(load_path):
                logger.warning(f"Model file not found at {load_path}")
                return False
            
            # Load model data
            model_data = joblib.load(load_path)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            
            logger.info(f"Model loaded successfully from {load_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        try:
            if self.model is None:
                raise ValueError("Model not trained. Please train the model first.")
            
            importance_scores = self.model.feature_importances_
            feature_importance = dict(zip(self.feature_names, importance_scores))
            
            # Sort by importance
            sorted_features = sorted(
                feature_importance.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            
            return dict(sorted_features)
            
        except Exception as e:
            logger.error(f"Error getting feature importance: {e}")
            raise
    
    def update_model(self, new_data: List[Dict], new_labels: List[int]) -> Dict:
        """
        Update the existing model with new data (incremental learning).
        
        Args:
            new_data: List of new health data points
            new_labels: List of corresponding labels
            
        Returns:
            Updated training metrics
        """
        try:
            if self.model is None:
                return self.train(new_data, new_labels)
            
            # Prepare new data
            X_new = []
            for data_point in new_data:
                features = []
                for feature in self.feature_names:
                    if feature in data_point:
                        features.append(float(data_point[feature]))
                    else:
                        features.append(0.0)
                X_new.append(features)
            
            X_new = np.array(X_new)
            y_new = np.array(new_labels)
            
            # Scale new data
            X_new_scaled = self.scaler.transform(X_new)
            
            # Update model with new data
            self.model.fit(X_new_scaled, y_new)
            
            # Evaluate on new data
            y_pred = self.model.predict(X_new_scaled)
            accuracy = accuracy_score(y_new, y_pred)
            
            logger.info(f"Model updated with new data. New accuracy: {accuracy:.4f}")
            
            return {
                'accuracy': accuracy,
                'samples_added': len(new_data),
                'status': 'updated'
            }
            
        except Exception as e:
            logger.error(f"Error updating model: {e}")
            raise 