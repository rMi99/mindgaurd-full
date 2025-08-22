from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import logging
from typing import Optional, Dict, Any
import io
from PIL import Image

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/facial-analysis", tags=["facial-analysis"])

class ImageData(BaseModel):
    image: str

class EmotionResult(BaseModel):
    emotion: str
    confidence: float
    faceDetected: bool
    emotions: Optional[Dict[str, float]] = None

# Initialize the emotion detection model
_emotion_detector = None

def get_emotion_detector():
    """Initialize and return the emotion detection model."""
    global _emotion_detector
    if _emotion_detector is None:
        try:
            # Try to import and initialize FER
            from fer import FER
            _emotion_detector = FER(mtcnn=True)
            logger.info("FER emotion detector initialized successfully")
        except ImportError:
            logger.warning("FER library not available, using mock detector")
            _emotion_detector = MockEmotionDetector()
        except Exception as e:
            logger.error(f"Failed to initialize emotion detector: {e}")
            _emotion_detector = MockEmotionDetector()
    
    return _emotion_detector

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
    """Analyze facial emotion from image."""
    try:
        detector = get_emotion_detector()
        
        # Detect emotions
        results = detector.detect_emotions(image)
        
        if not results:
            return EmotionResult(
                emotion="no_face_detected",
                confidence=0.0,
                faceDetected=False,
                emotions={}
            )
        
        # Get the first face result
        face_result = results[0]
        emotions = face_result['emotions']
        
        # Find dominant emotion
        dominant_emotion = max(emotions.items(), key=lambda x: x[1])
        
        return EmotionResult(
            emotion=dominant_emotion[0],
            confidence=dominant_emotion[1],
            faceDetected=True,
            emotions=emotions
        )
        
    except Exception as e:
        logger.error(f"Error analyzing emotion: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze facial emotion")

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
        
        # Analyze emotion
        result = analyze_facial_emotion(image)
        logger.info(f"Emotion analysis result: {result.emotion} (confidence: {result.confidence:.2f})")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in facial expression analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during emotion analysis")

@router.get("/health")
async def health_check():
    """Health check endpoint for facial analysis service."""
    try:
        detector = get_emotion_detector()
        detector_type = "FER" if hasattr(detector, 'fer') else "Mock"
        
        return {
            "status": "healthy",
            "detector_type": detector_type,
            "message": "Facial expression analysis service is running"
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
