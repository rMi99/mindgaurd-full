"""
Simplified facial dashboard routes for testing without complex dependencies.
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import logging
import uuid
from datetime import datetime
import random

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/facial-dashboard", tags=["facial-dashboard"])

class FacialAnalysisRequest(BaseModel):
    image: str
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_raw_metrics: bool = True
    include_phq9_estimation: bool = True

# Mock session storage
active_sessions: Dict[str, Dict] = {}

@router.post("/analyze")
async def analyze_comprehensive_facial(request: FacialAnalysisRequest):
    """Simplified facial analysis endpoint for testing."""
    try:
        logger.info("Received comprehensive facial analysis request")
        
        # Mock analysis results
        result = {
            "face_detected": True,
            "primary_emotion": random.choice(["happy", "neutral", "sad", "angry", "surprised"]),
            "emotion_confidence": random.uniform(0.6, 0.9),
            "emotion_distribution": {
                "happy": random.uniform(0.1, 0.4),
                "sad": random.uniform(0.1, 0.3),
                "angry": random.uniform(0.05, 0.2),
                "neutral": random.uniform(0.2, 0.5),
                "surprise": random.uniform(0.05, 0.2),
                "fear": random.uniform(0.05, 0.15),
                "disgust": random.uniform(0.05, 0.1)
            },
            "mood_assessment": random.choice(["Happy", "Neutral", "Sad", "Angry", "Surprised"]),
            "sleepiness": {
                "level": random.choice(["Alert", "Slightly tired", "Very tired"]),
                "confidence": random.uniform(0.6, 0.9),
                "contributing_factors": ["Eye aspect ratio analysis"]
            },
            "fatigue": {
                "yawning_detected": random.choice([True, False]),
                "head_droop_detected": random.choice([True, False]),
                "overall_fatigue": random.choice([True, False]),
                "confidence": random.uniform(0.5, 0.8)
            },
            "stress": {
                "level": random.choice(["Low", "Medium", "High"]),
                "confidence": random.uniform(0.6, 0.8),
                "indicators": ["Micro-expression analysis"]
            },
            "phq9_estimation": {
                "estimated_score": random.randint(0, 15),
                "confidence": random.uniform(0.5, 0.8),
                "severity_level": random.choice(["Minimal", "Mild", "Moderate"]),
                "contributing_expressions": ["Facial expression analysis"]
            },
            "eye_metrics": {
                "left_ear": random.uniform(0.25, 0.4),
                "right_ear": random.uniform(0.25, 0.4),
                "avg_ear": random.uniform(0.25, 0.4),
                "blink_rate": random.uniform(10, 25),
                "blink_duration": random.uniform(100, 200)
            },
            "head_pose": {
                "pitch": random.uniform(-15, 15),
                "yaw": random.uniform(-20, 20),
                "roll": random.uniform(-10, 10),
                "stability": random.uniform(0.6, 0.9)
            },
            "micro_expressions": {
                "muscle_tension": random.uniform(0.2, 0.7),
                "asymmetry": random.uniform(0.1, 0.4),
                "micro_movement_frequency": random.uniform(0.1, 0.6),
                "expression_variability": random.uniform(0.3, 0.8)
            },
            "analysis_quality": random.uniform(0.6, 0.9),
            "frame_quality": random.uniform(0.5, 0.9),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Update session if provided
        if request.session_id and request.session_id in active_sessions:
            session = active_sessions[request.session_id]
            session["total_frames_analyzed"] += 1
            session["average_quality"] = (
                (session["average_quality"] * (session["total_frames_analyzed"] - 1) + result["analysis_quality"]) 
                / session["total_frames_analyzed"]
            )
        
        logger.info(f"Mock analysis complete: {result['primary_emotion']}")
        return result
        
    except Exception as e:
        logger.error(f"Error in facial analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

@router.post("/session/start")
async def start_analysis_session(user_id: str):
    """Start a new facial analysis session."""
    try:
        session_id = str(uuid.uuid4())
        session = {
            "session_id": session_id,
            "user_id": user_id,
            "start_time": datetime.utcnow(),
            "total_frames_analyzed": 0,
            "average_quality": 0.0
        }
        
        active_sessions[session_id] = session
        
        logger.info(f"Started new facial analysis session: {session_id} for user: {user_id}")
        
        return {
            "session_id": session_id,
            "status": "started",
            "start_time": session["start_time"].isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start analysis session")

@router.post("/session/{session_id}/stop")
async def stop_analysis_session(session_id: str):
    """Stop an active facial analysis session."""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        session["end_time"] = datetime.utcnow()
        
        # Generate session summary
        session_duration = (session["end_time"] - session["start_time"]).total_seconds()
        session["session_summary"] = {
            "duration_seconds": session_duration,
            "total_frames": session["total_frames_analyzed"],
            "average_quality": session["average_quality"],
            "frames_per_minute": (session["total_frames_analyzed"] / (session_duration / 60)) if session_duration > 0 else 0
        }
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Stopped facial analysis session: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "stopped",
            "summary": session["session_summary"]
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping analysis session: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop analysis session")

@router.get("/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get the current status of an analysis session."""
    try:
        if session_id not in active_sessions:
            return {"status": "not_found"}
        
        session = active_sessions[session_id]
        current_time = datetime.utcnow()
        duration = (current_time - session["start_time"]).total_seconds()
        
        return {
            "session_id": session_id,
            "status": "active",
            "user_id": session["user_id"],
            "start_time": session["start_time"].isoformat(),
            "duration_seconds": duration,
            "total_frames_analyzed": session["total_frames_analyzed"],
            "average_quality": session["average_quality"],
            "frames_per_minute": (session["total_frames_analyzed"] / (duration / 60)) if duration > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session status")

@router.get("/insights/{user_id}")
async def get_facial_insights(user_id: str, days: int = 7):
    """Get mock insights for testing."""
    try:
        # Mock insights data
        insights = [
            "Your mood analysis shows improvement over the past week",
            "Stress levels appear to be within normal ranges",
            "Consider maintaining good lighting for better analysis accuracy"
        ]
        
        recommendations = [
            "Try to maintain good posture during analysis sessions",
            "Ensure adequate lighting for optimal facial detection",
            "Consider regular breaks if stress levels appear elevated"
        ]
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "analysis_period": f"{days} days",
            "total_analyses": random.randint(10, 50)
        }
        
    except Exception as e:
        logger.error(f"Error generating facial insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate insights")

@router.get("/health")
async def health_check():
    """Health check for facial dashboard."""
    return {
        "status": "healthy",
        "service": "Facial Dashboard",
        "message": "Mock facial analysis service is running"
    }
