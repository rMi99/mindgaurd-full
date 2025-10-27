from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any, List
import io
from PIL import Image
from collections import deque
from datetime import datetime
from app.services.enhanced_facial_detector import analyze_facial_expression, get_detector_status
from app.services.real_facial_analyzer import analyze_facial_expression as real_analyze_facial_expression, get_analyzer_status

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/facial-analysis", tags=["facial-analysis"])

# Global emotion history tracking (in production, use Redis or database)
emotion_history = deque(maxlen=50)  # Keep last 50 emotion readings

class ImageData(BaseModel):
    image: str

class EmotionResult(BaseModel):
    emotion: str
    confidence: float
    faceDetected: bool
    emotions: Optional[Dict[str, float]] = None
    sleepiness: Optional[str] = "unknown"
    detection_method: Optional[str] = None
    emotion_method: Optional[str] = None

class MockEmotionDetector:
    """Mock emotion detector for development/testing when FER is not available."""
    
    def detect_emotions(self, image):
        """Mock emotion detection that returns random but realistic emotions."""
        import random
        
        # Simulate face detection
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            return []
        
        # Mock emotions with realistic distributions
        emotions = {
            'neutral': random.uniform(0.2, 0.8),
            'happy': random.uniform(0.1, 0.6),
            'sad': random.uniform(0.05, 0.3),
            'angry': random.uniform(0.02, 0.2),
            'surprise': random.uniform(0.02, 0.2),
            'disgust': random.uniform(0.01, 0.1),
            'fear': random.uniform(0.01, 0.15)
        }
        
        # Normalize to sum to 1
        total = sum(emotions.values())
        emotions = {k: v/total for k, v in emotions.items()}
        
        return [{
            'emotions': emotions,
            'box': [50, 50, 200, 200]  # Mock bounding box
        }]

def decode_base64_image(image_data: str) -> np.ndarray:
    """Decode base64 image string to numpy array."""
    try:
        # Remove data URL prefix if present
        if ',' in image_data:
            image_data = image_data.split(',')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(image_data)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(img_bytes))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array
        img_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        return img_bgr
    
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail="Invalid image data")

def analyze_facial_emotion(image: np.ndarray) -> EmotionResult:
    """Analyze facial emotion from image using enhanced detector."""
    try:
        # Use real facial analyzer for emotion analysis
        import io
        
        # Create a mock file object for the real analyzer
        class MockFile:
            def __init__(self, image_data):
                self.file = io.BytesIO(image_data)
        
        # Convert image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        mock_file = MockFile(buffer.tobytes())
        
        # Get result from real analyzer
        result = real_analyze_facial_expression(mock_file)
        
        if result and 'emotion' in result:
            emotion = result.get('emotion', 'neutral')
            confidence = result.get('confidence', 0.5)
            sleepiness = result.get('sleepiness', 'unknown')
            
            # Ensure sleepiness is a string
            if isinstance(sleepiness, dict):
                sleepiness = sleepiness.get('level', 'unknown')
            elif sleepiness is None:
                sleepiness = 'unknown'
            else:
                sleepiness = str(sleepiness)
            
            # Create emotion distribution
            emotions = {
                emotion: confidence,
                'neutral': 0.1,
                'happy': 0.1,
                'sad': 0.1,
                'angry': 0.1,
                'fear': 0.1,
                'surprise': 0.1,
                'disgust': 0.1
            }
            emotions[emotion] = confidence
            
            return EmotionResult(
                emotion=emotion,
                confidence=confidence,
                faceDetected=True,
                emotions=emotions,
                sleepiness=sleepiness,
                detection_method=result.get('detection_method', 'unknown'),
                emotion_method=result.get('emotion_method', 'unknown')
            )
        else:
            # Provide more detailed feedback for no face detection
            return EmotionResult(
                emotion="no_face_detected",
                confidence=0.0,
                faceDetected=False,
                emotions={},
                sleepiness="unknown",
                detection_method="none",
                emotion_method="none"
            )
        
    except Exception as e:
        logger.error(f"Error analyzing emotion: {e}")
        # Provide more specific error information
        error_detail = "Failed to analyze facial emotion"
        if "no module named" in str(e).lower():
            error_detail = "Required dependencies are missing. Please check system configuration."
        elif "invalid image" in str(e).lower():
            error_detail = "Invalid image format. Please ensure the image is properly encoded."
        elif "face" in str(e).lower() and "detect" in str(e).lower():
            error_detail = "Face detection failed. Please ensure your face is clearly visible and well-lit."
        
        return EmotionResult(
            emotion="analysis_error",
            confidence=0.0,
            faceDetected=False,
            emotions={},
            sleepiness="unknown",
            detection_method="error",
            emotion_method="error"
        )

@router.post("/", response_model=EmotionResult)
async def analyze_expression(data: ImageData) -> EmotionResult:
    """
    Analyze facial expression from a base64 encoded image.
    
    Args:
        data: ImageData containing base64 encoded image
        
    Returns:
        EmotionResult with detected emotion and confidence
    """
    try:
        logger.info("Received facial expression analysis request")
        
        # Decode the image
        image = decode_base64_image(data.image)
        logger.debug(f"Decoded image shape: {image.shape}")
        
        # Analyze emotion using enhanced detector
        result = analyze_facial_emotion(image)
        logger.info(f"Emotion analysis result: {result.emotion} (confidence: {result.confidence:.2f})")
        
        # Track emotion in history
        emotion_record = {
            "timestamp": datetime.now().isoformat(),
            "emotion": result.emotion,
            "confidence": result.confidence,
            "sleepiness": result.sleepiness,
            "faceDetected": result.faceDetected
        }
        emotion_history.append(emotion_record)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in facial expression analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during emotion analysis")

@router.post("/analyze", response_model=EmotionResult)
async def analyze_face_file(file: UploadFile = File(...)) -> EmotionResult:
    """
    Analyze facial expression from uploaded image file.
    
    Args:
        file: Uploaded image file
        
    Returns:
        EmotionResult with detected emotion and confidence
    """
    try:
        logger.info("Received facial expression analysis request from file upload")
        
        # Use enhanced facial detector
        result = analyze_facial_expression(file)
        
        # Convert to EmotionResult format
        sleepiness = result.get("sleepiness", "unknown")
        # Ensure sleepiness is a string
        if isinstance(sleepiness, dict):
            sleepiness = sleepiness.get('level', 'unknown')
        elif sleepiness is None:
            sleepiness = 'unknown'
        else:
            sleepiness = str(sleepiness)
            
        emotion_result = EmotionResult(
            emotion=result.get("emotion", "unknown"),
            confidence=result.get("confidence", 0.0),
            faceDetected=result.get("face_count", 0) > 0,
            sleepiness=sleepiness,
            detection_method=result.get("detection_method", "unknown"),
            emotion_method=result.get("emotion_method", "unknown")
        )
        
        logger.info(f"Enhanced analysis result: {emotion_result.emotion} (confidence: {emotion_result.confidence:.2f})")
        
        return emotion_result
        
    except Exception as e:
        logger.error(f"Error in enhanced facial analysis: {e}")
        # Provide more specific error information
        error_detail = "Failed to analyze facial expression"
        if "no module named" in str(e).lower():
            error_detail = "Required dependencies are missing. Please check system configuration."
        elif "invalid image" in str(e).lower():
            error_detail = "Invalid image format. Please ensure the image is properly encoded."
        elif "face" in str(e).lower() and "detect" in str(e).lower():
            error_detail = "Face detection failed. Please ensure your face is clearly visible and well-lit."
        
        return EmotionResult(
            emotion="analysis_error",
            confidence=0.0,
            faceDetected=False,
            emotions={},
            sleepiness="unknown",
            detection_method="error",
            emotion_method="error"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint for facial analysis service."""
    try:
        # Get real analyzer status
        analyzer_status = get_analyzer_status()
        
        # Get enhanced detector status as fallback
        detector_status = get_detector_status()
        
        # Determine primary analyzer type
        if analyzer_status["model_loaded"]:
            analyzer_type = "Real Trained Model"
        elif detector_status["fer_available"]:
            analyzer_type = "FER"
        elif detector_status["deepface_available"]:
            analyzer_type = "DeepFace"
        elif detector_status["dlib_available"]:
            analyzer_type = "Dlib"
        elif detector_status["mediapipe_available"]:
            analyzer_type = "MediaPipe"
        else:
            analyzer_type = "OpenCV Fallback"
        
        return {
            "status": "healthy",
            "analyzer_type": analyzer_type,
            "message": "Real facial expression analysis service is running",
            "real_analyzer": analyzer_status,
            "fallback_detectors": detector_status
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.get("/emotions")
async def get_supported_emotions():
    """Get list of supported emotions."""
    return {
        "emotions": [
            "happy",
            "sad", 
            "angry",
            "surprise",
            "fear",
            "disgust",
            "neutral"
        ],
        "description": "Supported facial emotions for analysis"
    }

@router.get("/trends")
async def get_emotion_trends():
    """Get recent emotion trends and statistics."""
    try:
        if not emotion_history:
            return {
                "trends": [],
                "statistics": {},
                "message": "No emotion data available yet"
            }
        
        # Calculate emotion statistics
        emotions = [record["emotion"] for record in emotion_history if record["faceDetected"]]
        confidences = [record["confidence"] for record in emotion_history if record["faceDetected"]]
        sleepiness_levels = [record["sleepiness"] for record in emotion_history if record["faceDetected"]]
        
        # Count emotions
        emotion_counts = {}
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        # Calculate averages
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Count sleepiness levels
        sleepiness_counts = {}
        for level in sleepiness_levels:
            sleepiness_counts[level] = sleepiness_counts.get(level, 0) + 1
        
        # Determine dominant emotion
        dominant_emotion = max(emotion_counts, key=emotion_counts.get) if emotion_counts else "unknown"
        
        # Determine trend direction
        recent_emotions = emotions[-10:] if len(emotions) >= 10 else emotions
        earlier_emotions = emotions[:-10] if len(emotions) >= 20 else []
        
        trend_direction = "stable"
        if len(earlier_emotions) > 0 and len(recent_emotions) > 0:
            recent_positive = sum(1 for e in recent_emotions if e in ["happy", "surprise"])
            earlier_positive = sum(1 for e in earlier_emotions if e in ["happy", "surprise"])
            
            if recent_positive > earlier_positive:
                trend_direction = "improving"
            elif recent_positive < earlier_positive:
                trend_direction = "declining"
        
        return {
            "trends": list(emotion_history),
            "statistics": {
                "total_readings": len(emotion_history),
                "successful_detections": len([r for r in emotion_history if r["faceDetected"]]),
                "emotion_counts": emotion_counts,
                "dominant_emotion": dominant_emotion,
                "average_confidence": round(avg_confidence, 3),
                "sleepiness_distribution": sleepiness_counts,
                "trend_direction": trend_direction
            },
            "message": f"Analyzed {len(emotion_history)} emotion readings"
        }
        
    except Exception as e:
        logger.error(f"Error getting emotion trends: {e}")
        return {
            "trends": [],
            "statistics": {},
            "error": str(e)
        }
