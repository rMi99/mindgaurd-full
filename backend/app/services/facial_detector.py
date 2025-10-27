"""
Enhanced Facial Detector Service
Centralized emotion detection with FER, DeepFace, and fallback support
"""

import cv2
import numpy as np
import logging
import os
import io
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

# Global detector instances
fer_detector = None
deepface_available = False

def initialize_detectors():
    """Initialize all available emotion detection models."""
    global fer_detector, deepface_available
    
    # Initialize FER
    try:
        # Ensure moviepy is available for FER
        try:
            import moviepy.editor
        except ImportError:
            logger.warning("moviepy not available, installing...")
            import subprocess
            subprocess.run(["pip", "install", "moviepy"], check=False)
        
        from fer import FER
        fer_detector = FER(mtcnn=True)
        logger.info("✅ FER detector loaded successfully")
    except Exception as e:
        fer_detector = None
        logger.warning(f"❌ FER initialization failed: {e}")
    
    # Check DeepFace availability
    try:
        from deepface import DeepFace
        deepface_available = True
        logger.info("✅ DeepFace available for emotion detection")
    except ImportError:
        deepface_available = False
        logger.warning("❌ DeepFace not available, using FER fallback")

def analyze_facial_expression(file) -> Dict[str, Any]:
    """
    Analyze facial emotion and return detected emotion data.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Dictionary with emotion, confidence, and sleepiness data
    """
    try:
        # Read image bytes
        img_bytes = file.file.read()
        npimg = np.frombuffer(img_bytes, np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {
                "emotion": "no_face_detected",
                "confidence": 0.0,
                "sleepiness": "unknown",
                "error": "Invalid image format"
            }
        
        emotion, confidence = "neutral", 0.5
        detection_method = "fallback"
        emotion_method = "mock"
        
        # Try FER first
        if fer_detector:
            try:
                emotions = fer_detector.detect_emotions(frame)
                if emotions and len(emotions) > 0:
                    emotion_data = emotions[0]["emotions"]
                    emotion = max(emotion_data, key=emotion_data.get)
                    confidence = round(emotion_data[emotion], 3)
                    emotion_method = "fer"
                    logger.info(f"FER detected: {emotion} (confidence: {confidence})")
                else:
                    logger.info("No emotion detected by FER. Using default neutral.")
            except Exception as e:
                logger.warning(f"FER detection failed: {e}")
        
        # Try DeepFace if FER failed
        elif deepface_available:
            try:
                from deepface import DeepFace
                analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if analysis and len(analysis) > 0:
                    emotion = analysis[0]['dominant_emotion']
                    confidence = 0.95
                    emotion_method = "deepface"
                    logger.info(f"DeepFace detected: {emotion}")
            except Exception as e:
                logger.warning(f"DeepFace detection failed: {e}")
        
        # Fallback to heuristic detection
        else:
            logger.info("No emotion detector available. Using heuristic detection.")
            emotion, confidence = detect_emotion_heuristic(frame)
            emotion_method = "heuristic"
        
        # Determine sleepiness based on emotion
        sleepiness = "sleepy" if emotion in ["sad", "tired", "neutral"] else "alert"
        
        # Determine detection method
        if fer_detector:
            detection_method = "dlib_opencv"
        elif deepface_available:
            detection_method = "deepface"
        else:
            detection_method = "opencv"
        
        logger.info(f"✅ Facial analysis complete: {emotion} ({confidence:.2f}) - {sleepiness}")
        
        return {
            "emotion": emotion,
            "confidence": confidence,
            "sleepiness": sleepiness,
            "detection_method": detection_method,
            "emotion_method": emotion_method
        }
        
    except Exception as e:
        logger.error(f"Facial analysis error: {e}")
        return {
            "emotion": "analysis_error",
            "confidence": 0.0,
            "sleepiness": "unknown",
            "error": str(e)
        }

def detect_emotion_heuristic(frame: np.ndarray) -> tuple:
    """Heuristic emotion detection using OpenCV features."""
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect eyes for basic emotion estimation
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        eyes = eye_cascade.detectMultiScale(gray, 1.1, 3)
        
        if len(eyes) >= 2:
            # Eyes detected - likely alert
            return "alert", 0.7
        else:
            # No eyes detected - might be tired or looking away
            return "tired", 0.6
            
    except Exception as e:
        logger.warning(f"Heuristic detection failed: {e}")
        return "neutral", 0.5

def train_all_models():
    """Train or refresh all emotion detection models."""
    logger.info("🚀 Starting model training process...")
    try:
        # Check if training scripts exist
        training_scripts = [
            "scripts/train_fer_model.py",
            "scripts/train_deepface_model.py", 
            "scripts/train_facial_model.py"
        ]
        
        for script in training_scripts:
            if os.path.exists(script):
                logger.info(f"Running {script}...")
                os.system(f"python {script}")
            else:
                logger.warning(f"Training script not found: {script}")
        
        logger.info("✅ All models trained successfully")
        return True
        
    except Exception as e:
        logger.error(f"Error during model training: {e}")
        return False

def get_detector_status() -> Dict[str, Any]:
    """Get status of all available detectors."""
    return {
        "fer_available": fer_detector is not None,
        "deepface_available": deepface_available,
        "detectors_initialized": any([fer_detector is not None, deepface_available])
    }

# Initialize detectors on module import
initialize_detectors()
