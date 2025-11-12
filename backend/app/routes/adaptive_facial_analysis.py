"""
Enhanced Facial Analysis API Routes
Integrates all design patterns with WebSocket support
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, WebSocket, WebSocketDisconnect, Depends
from fastapi.responses import JSONResponse
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
import uuid

from app.services.adaptive_facial_service import get_adaptive_service
from app.services.websocket_manager import get_websocket_manager
from app.core.patterns import ModelType

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/facial-analysis", tags=["facial-analysis"])

# Request/Response Models
class ImageData(BaseModel):
    image: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_raw_metrics: bool = False
    include_phq9_estimation: bool = False

class ModelSwitchRequest(BaseModel):
    model_type: str
    user_id: Optional[str] = None

class SessionStartRequest(BaseModel):
    user_id: str

class AnalysisResult(BaseModel):
    emotion: str
    confidence: float
    faceDetected: bool
    emotions: Optional[Dict[str, float]] = None
    sleepiness: Optional[str] = "unknown"
    detection_method: Optional[str] = None
    emotion_method: Optional[str] = None
    model_type: Optional[str] = None
    adaptive_analysis: Optional[bool] = False
    session_id: Optional[str] = None
    timestamp: Optional[str] = None

class WebSocketMessage(BaseModel):
    type: str
    data: Optional[Dict[str, Any]] = None
    session_id: Optional[str] = None

# Global service instances
adaptive_service = get_adaptive_service()
websocket_manager = get_websocket_manager()

# Subscribe WebSocket manager to adaptive service
adaptive_service.attach(websocket_manager.get_observer())

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

@router.post("/analyze", response_model=AnalysisResult)
async def analyze_expression(data: ImageData) -> AnalysisResult:
    """
    Analyze facial expression with adaptive accuracy tuning.
    
    Args:
        data: ImageData containing base64 encoded image and analysis options
        
    Returns:
        AnalysisResult with detected emotion and comprehensive analysis
    """
    try:
        logger.info("Received adaptive facial expression analysis request")
        
        # Decode the image
        image = decode_base64_image(data.image)
        logger.debug(f"Decoded image shape: {image.shape}")
        
        # Create mock file object for the service
        class MockFile:
            def __init__(self, image_data):
                self.file = io.BytesIO(image_data)
        
        # Convert image to bytes
        _, buffer = cv2.imencode('.jpg', image)
        mock_file = MockFile(buffer.tobytes())
        
        # Perform adaptive analysis
        result = adaptive_service.analyze_facial_expression(mock_file)
        
        # Add session information if provided
        if data.session_id:
            result['session_id'] = data.session_id
        
        if data.user_id:
            result['user_id'] = data.user_id
        
        logger.info(f"Adaptive analysis result: {result.get('emotion', 'unknown')} (confidence: {result.get('confidence', 0.0):.2f})")
        
        # Send real-time update via WebSocket
        if data.session_id:
            await websocket_manager.send_analysis_result(result, session_id=data.session_id)
        
        return AnalysisResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in adaptive facial expression analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during emotion analysis")

@router.post("/analyze-file", response_model=AnalysisResult)
async def analyze_face_file(file: UploadFile = File(...)) -> AnalysisResult:
    """
    Analyze facial expression from uploaded image file with adaptive tuning.
    
    Args:
        file: Uploaded image file
        
    Returns:
        AnalysisResult with detected emotion and comprehensive analysis
    """
    try:
        logger.info("Received adaptive facial expression analysis request from file upload")
        
        # Perform adaptive analysis
        result = adaptive_service.analyze_facial_expression(file)
        
        logger.info(f"Adaptive analysis result: {result.get('emotion', 'unknown')} (confidence: {result.get('confidence', 0.0):.2f})")
        
        return AnalysisResult(**result)
        
    except Exception as e:
        logger.error(f"Error in adaptive facial analysis: {e}")
        raise HTTPException(status_code=500, detail="Failed to analyze facial expression")

@router.post("/model/switch")
async def switch_model(request: ModelSwitchRequest) -> JSONResponse:
    """
    Switch to a different AI model type.
    
    Args:
        request: ModelSwitchRequest with model type and user ID
        
    Returns:
        JSON response with switch status
    """
    try:
        logger.info(f"Model switch request: {request.model_type}")
        
        result = adaptive_service.switch_model(request.model_type)
        
        # Send model update notification
        await websocket_manager.send_model_update({
            'action': 'model_switched',
            'model_type': request.model_type,
            'user_id': request.user_id,
            'result': result
        })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error switching model: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/status")
async def get_model_status() -> JSONResponse:
    """
    Get current model status and statistics.
    
    Returns:
        JSON response with model status and accuracy statistics
    """
    try:
        status = adaptive_service.get_model_status()
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting model status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/supported")
async def get_supported_models() -> JSONResponse:
    """
    Get list of supported model types.
    
    Returns:
        JSON response with supported model types
    """
    try:
        supported_models = adaptive_service.model_manager.get_supported_models()
        return JSONResponse(content={
            'supported_models': supported_models,
            'current_model': adaptive_service.current_model.get_model_info()['name'] if adaptive_service.current_model else None
        })
        
    except Exception as e:
        logger.error(f"Error getting supported models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/trends")
async def get_analysis_trends(limit: int = 50) -> JSONResponse:
    """
    Get analysis trends and statistics.
    
    Args:
        limit: Maximum number of recent analyses to include
        
    Returns:
        JSON response with trends and statistics
    """
    try:
        trends = adaptive_service.get_analysis_trends(limit)
        return JSONResponse(content=trends)
        
    except Exception as e:
        logger.error(f"Error getting analysis trends: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/start")
async def start_session(request: SessionStartRequest) -> JSONResponse:
    """
    Start a new analysis session.
    
    Args:
        request: SessionStartRequest with user ID
        
    Returns:
        JSON response with session information
    """
    try:
        result = adaptive_service.start_session(request.user_id)
        
        # Send session start notification
        await websocket_manager.send_analysis_result({
            'type': 'session_started',
            'session_id': result.get('session_id'),
            'user_id': request.user_id
        })
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/session/{session_id}/stop")
async def stop_session(session_id: str) -> JSONResponse:
    """
    Stop an analysis session.
    
    Args:
        session_id: Session ID to stop
        
    Returns:
        JSON response with session summary
    """
    try:
        result = adaptive_service.stop_session(session_id)
        
        # Send session stop notification
        await websocket_manager.send_analysis_result({
            'type': 'session_stopped',
            'session_id': session_id
        }, session_id=session_id)
        
        return JSONResponse(content=result)
        
    except Exception as e:
        logger.error(f"Error stopping session: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str) -> JSONResponse:
    """
    Get session status.
    
    Args:
        session_id: Session ID
        
    Returns:
        JSON response with session status
    """
    try:
        status = adaptive_service.get_session_status(session_id)
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check() -> JSONResponse:
    """
    Health check endpoint for adaptive facial analysis service.
    
    Returns:
        JSON response with service health status
    """
    try:
        model_status = adaptive_service.get_model_status()
        websocket_stats = websocket_manager.get_stats()
        
        return JSONResponse(content={
            "status": "healthy",
            "service": "Adaptive Facial Analysis API",
            "version": "2.0.0",
            "model_status": model_status.get('service_status', 'unknown'),
            "websocket_stats": websocket_stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

@router.get("/emotions")
async def get_supported_emotions() -> JSONResponse:
    """
    Get list of supported emotions.
    
    Returns:
        JSON response with supported emotions
    """
    return JSONResponse(content={
        "emotions": [
            "happy",
            "sad", 
            "angry",
            "surprise",
            "fear",
            "disgust",
            "neutral"
        ],
        "description": "Supported facial emotions for adaptive analysis"
    })

# WebSocket endpoint for real-time communication
@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """
    WebSocket endpoint for real-time facial analysis updates.
    
    Args:
        websocket: WebSocket connection
        client_id: Unique client identifier
    """
    await websocket_manager.connect_client(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message = WebSocketMessage.parse_raw(data)
                
                if message.type == "join_session" and message.session_id:
                    await websocket_manager.join_session(client_id, message.session_id)
                elif message.type == "leave_session" and message.session_id:
                    await websocket_manager.leave_session(client_id, message.session_id)
                elif message.type == "ping":
                    await websocket_manager.connection_manager.send_personal_message({
                        'type': 'pong',
                        'timestamp': datetime.now().isoformat()
                    }, client_id)
                
            except Exception as e:
                logger.error(f"Error processing WebSocket message: {e}")
                await websocket_manager.connection_manager.send_personal_message({
                    'type': 'error',
                    'message': 'Invalid message format',
                    'timestamp': datetime.now().isoformat()
                }, client_id)
                
    except WebSocketDisconnect:
        websocket_manager.disconnect_client(client_id)
        logger.info(f"WebSocket client {client_id} disconnected")
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        websocket_manager.disconnect_client(client_id)

@router.get("/websocket/stats")
async def get_websocket_stats() -> JSONResponse:
    """
    Get WebSocket connection statistics.
    
    Returns:
        JSON response with WebSocket statistics
    """
    try:
        stats = websocket_manager.get_stats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting WebSocket stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))
