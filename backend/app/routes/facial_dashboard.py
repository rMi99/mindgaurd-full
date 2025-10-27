"""
Enhanced facial analysis routes providing comprehensive real-time assessment
dashboard functionality.
"""

from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, List, Dict, Any
import logging
import uuid
from datetime import datetime, timedelta
import base64
import cv2
import numpy as np
import io
from PIL import Image

from app.models.facial_metrics import (
    FacialAnalysisRequest, ComprehensiveFacialAnalysis, 
    FacialAnalysisSession, FacialAnalysisHistory
)
from app.models.health_recommendations import (
    EnhancedFacialAnalysisResult, HealthRecommendations, 
    RealtimeMonitoring, SessionProgress
)
from app.services.enhanced_facial_analyzer import EnhancedFacialAnalyzer
from app.services.health_recommendation_service import HealthRecommendationService
from app.services.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/facial-dashboard", tags=["facial-dashboard"])
security = HTTPBearer(auto_error=False)

# Initialize the enhanced analyzer and health recommendation service
analyzer = EnhancedFacialAnalyzer()
health_service = HealthRecommendationService()

# Active sessions tracking
active_sessions: Dict[str, FacialAnalysisSession] = {}

@router.post("/analyze", response_model=ComprehensiveFacialAnalysis)
async def analyze_comprehensive_facial(
    request: FacialAnalysisRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Perform comprehensive facial analysis including mood, sleepiness, 
    fatigue, stress indicators, and PHQ-9 estimation.
    """
    try:
        logger.info("Received comprehensive facial analysis request")
        
        # Decode the image
        image = decode_base64_image(request.image)
        logger.debug(f"Decoded image shape: {image.shape}")
        
        # Perform comprehensive analysis
        result = analyzer.analyze_comprehensive(
            image=image,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Store result in database if user_id provided
        if request.user_id:
            await store_analysis_result(result, request.user_id, request.session_id)
        
        # Update session tracking
        if request.session_id and request.session_id in active_sessions:
            session = active_sessions[request.session_id]
            session.total_frames_analyzed += 1
            session.average_quality = (
                (session.average_quality * (session.total_frames_analyzed - 1) + result.analysis_quality) 
                / session.total_frames_analyzed
            )
        
        logger.info(f"Analysis complete: {result.primary_emotion} (quality: {result.analysis_quality:.2f})")
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in comprehensive facial analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during analysis")

@router.post("/analyze-enhanced", response_model=EnhancedFacialAnalysisResult)
async def analyze_enhanced_facial(
    request: FacialAnalysisRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Perform enhanced facial analysis with health recommendations, exercises, 
    games, and real-time monitoring capabilities.
    """
    try:
        logger.info("Received enhanced facial analysis request")
        
        # Decode the image
        image = decode_base64_image(request.image)
        logger.debug(f"Decoded image shape: {image.shape}")
        
        # Perform comprehensive facial analysis
        facial_analysis = analyzer.analyze_comprehensive(
            image=image,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Generate health recommendations
        health_recommendations = health_service.analyze_health_status(facial_analysis)
        
        # Create real-time monitoring data
        realtime_monitoring = health_service.create_realtime_monitoring(
            session_id=request.session_id or str(uuid.uuid4()),
            facial_analysis=facial_analysis,
            health_recommendations=health_recommendations
        )
        
        # Create enhanced result with all required fields
        enhanced_result = EnhancedFacialAnalysisResult(
            # Original analysis data
            emotions=facial_analysis.emotion_distribution.model_dump() if hasattr(facial_analysis.emotion_distribution, 'model_dump') else facial_analysis.emotion_distribution.dict(),
            dominant_emotion=facial_analysis.primary_emotion,
            confidence=facial_analysis.emotion_confidence,
            
            # Enhanced health analysis
            health_recommendations=health_recommendations,
            realtime_monitoring=realtime_monitoring,
            
            # Analysis metadata
            analysis_timestamp=datetime.utcnow(),
            user_id=request.user_id,
            session_id=request.session_id or realtime_monitoring.session_id,
            image_quality=facial_analysis.analysis_quality
        )
        
        # Store result in database if user_id provided
        if request.user_id:
            await store_enhanced_analysis_result(enhanced_result, request.user_id)
        
        # Update session tracking
        if request.session_id and request.session_id in active_sessions:
            session = active_sessions[request.session_id]
            session.total_frames_analyzed += 1
            session.average_quality = (
                (session.average_quality * (session.total_frames_analyzed - 1) + facial_analysis.analysis_quality) 
                / session.total_frames_analyzed
            )
        
        logger.info(f"Enhanced analysis complete: {facial_analysis.primary_emotion} -> {health_recommendations.health_status}")
        
        return enhanced_result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in enhanced facial analysis: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during enhanced analysis")

@router.post("/session/start")
async def start_analysis_session(
    user_id: str = Query(..., description="User ID for the session"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Start a new facial analysis session."""
    try:
        session_id = str(uuid.uuid4())
        session = FacialAnalysisSession(
            session_id=session_id,
            user_id=user_id,
            start_time=datetime.utcnow(),
            total_frames_analyzed=0,
            average_quality=0.0
        )
        
        active_sessions[session_id] = session
        
        logger.info(f"Started new facial analysis session: {session_id} for user: {user_id}")
        
        return {
            "session_id": session_id,
            "status": "started",
            "start_time": session.start_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error starting analysis session: {e}")
        raise HTTPException(status_code=500, detail="Failed to start analysis session")

@router.post("/session/{session_id}/stop")
async def stop_analysis_session(
    session_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Stop an active facial analysis session."""
    try:
        if session_id not in active_sessions:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = active_sessions[session_id]
        session.end_time = datetime.utcnow()
        
        # Generate session summary
        session_duration = (session.end_time - session.start_time).total_seconds()
        session.session_summary = {
            "duration_seconds": session_duration,
            "total_frames": session.total_frames_analyzed,
            "average_quality": session.average_quality,
            "frames_per_minute": (session.total_frames_analyzed / (session_duration / 60)) if session_duration > 0 else 0
        }
        
        # Store session in database
        if session.user_id:
            await store_session_data(session)
        
        # Remove from active sessions
        del active_sessions[session_id]
        
        logger.info(f"Stopped facial analysis session: {session_id}")
        
        return {
            "session_id": session_id,
            "status": "stopped",
            "summary": session.session_summary
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping analysis session: {e}")
        raise HTTPException(status_code=500, detail="Failed to stop analysis session")

@router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get the current status of an analysis session."""
    try:
        if session_id not in active_sessions:
            return {"status": "not_found"}
        
        session = active_sessions[session_id]
        current_time = datetime.utcnow()
        duration = (current_time - session.start_time).total_seconds()
        
        return {
            "session_id": session_id,
            "status": "active",
            "user_id": session.user_id,
            "start_time": session.start_time.isoformat(),
            "duration_seconds": duration,
            "total_frames_analyzed": session.total_frames_analyzed,
            "average_quality": session.average_quality,
            "frames_per_minute": (session.total_frames_analyzed / (duration / 60)) if duration > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"Error getting session status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session status")

@router.get("/session/{session_id}/progress", response_model=SessionProgress)
async def get_session_progress(
    session_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get detailed progress tracking for a session."""
    try:
        progress = health_service.track_progress(session_id)
        
        if not progress:
            raise HTTPException(status_code=404, detail="Session progress not found")
        
        return progress
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session progress: {e}")
        raise HTTPException(status_code=500, detail="Failed to get session progress")

@router.get("/session/{session_id}/monitoring", response_model=RealtimeMonitoring)
async def get_realtime_monitoring(
    session_id: str,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get real-time monitoring data for a session."""
    try:
        if session_id not in health_service.session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Get the latest monitoring data (simplified for this endpoint)
        # In a real implementation, this would fetch the latest monitoring data
        session_data = health_service.session_data[session_id]
        
        # Create a minimal monitoring response
        from app.models.health_recommendations import BiometricIndicators
        
        return RealtimeMonitoring(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            current_health_status="moderate",  # This should come from latest analysis
            mood_trend=session_data.get('mood_history', [0.5])[-10:],
            stress_trend=session_data.get('stress_history', [0.5])[-10:],
            brain_activity=BiometricIndicators(
                stress_level=0.5, focus_level=0.5, energy_level=0.5,
                emotional_stability=0.5, cognitive_load=0.5, alertness=0.5
            ),
            active_alerts=[],
            recommendations_queue=["Stay hydrated", "Take regular breaks"],
            session_start_time=session_data.get('start_time', datetime.utcnow()),
            total_analysis_frames=session_data.get('total_frames', 0),
            average_mood_score=sum(session_data.get('mood_scores', [0.5])) / len(session_data.get('mood_scores', [1]))
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting realtime monitoring: {e}")
        raise HTTPException(status_code=500, detail="Failed to get monitoring data")

@router.post("/session/{session_id}/exercise-completed")
async def mark_exercise_completed(
    session_id: str,
    exercise_data: Dict[str, Any],
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Mark an exercise as completed and update progress."""
    try:
        if session_id not in health_service.session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = health_service.session_data[session_id]
        
        # Add exercise completion data
        if 'exercises_completed' not in session:
            session['exercises_completed'] = []
        
        exercise_completion = {
            'exercise_id': exercise_data.get('exercise_id'),
            'exercise_name': exercise_data.get('exercise_name'),
            'completed_at': datetime.utcnow().isoformat(),
            'duration_seconds': exercise_data.get('duration_seconds', 0),
            'user_rating': exercise_data.get('user_rating')  # 1-5 scale
        }
        
        session['exercises_completed'].append(exercise_completion)
        
        # Update post-exercise mood if provided
        if 'post_exercise_mood' in exercise_data:
            session['post_exercise_mood'] = exercise_data['post_exercise_mood']
        
        logger.info(f"Exercise completed in session {session_id}: {exercise_data.get('exercise_name')}")
        
        return {
            "status": "success",
            "message": "Exercise completion recorded",
            "total_exercises_completed": len(session['exercises_completed'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording exercise completion: {e}")
        raise HTTPException(status_code=500, detail="Failed to record exercise completion")

@router.post("/session/{session_id}/game-completed")
async def mark_game_completed(
    session_id: str,
    game_data: Dict[str, Any],
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Mark a game as completed and update progress."""
    try:
        if session_id not in health_service.session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        session = health_service.session_data[session_id]
        
        # Add game completion data
        if 'games_played' not in session:
            session['games_played'] = []
        
        game_completion = {
            'game_id': game_data.get('game_id'),
            'game_name': game_data.get('game_name'),
            'completed_at': datetime.utcnow().isoformat(),
            'duration_seconds': game_data.get('duration_seconds', 0),
            'score': game_data.get('score'),
            'difficulty': game_data.get('difficulty'),
            'user_rating': game_data.get('user_rating')  # 1-5 scale
        }
        
        session['games_played'].append(game_completion)
        
        logger.info(f"Game completed in session {session_id}: {game_data.get('game_name')}")
        
        return {
            "status": "success",
            "message": "Game completion recorded",
            "total_games_played": len(session['games_played'])
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error recording game completion: {e}")
        raise HTTPException(status_code=500, detail="Failed to record game completion")

@router.get("/recommendations/{user_id}")
async def get_personalized_recommendations(
    user_id: str,
    limit: int = Query(10, ge=1, le=50),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get personalized recommendations based on user's history."""
    try:
        db = await get_db()
        collection = db["enhanced_facial_analyses"]
        
        # Get recent analyses
        cursor = collection.find({
            "user_id": user_id
        }).sort("analysis_timestamp", -1).limit(limit)
        
        analyses = await cursor.to_list(length=None)
        
        if not analyses:
            return {
                "recommendations": [],
                "message": "No analysis history found"
            }
        
        # Analyze patterns and generate recommendations
        recommendations = analyze_user_patterns_and_recommend(analyses)
        
        return {
            "user_id": user_id,
            "recommendations": recommendations,
            "based_on_analyses": len(analyses),
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting personalized recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to get recommendations")
        raise HTTPException(status_code=500, detail="Failed to get session status")

@router.get("/dashboard/{user_id}")
async def get_facial_dashboard(
    user_id: str,
    days: int = Query(7, ge=1, le=30, description="Number of days to include"),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get comprehensive facial analysis dashboard data for a user."""
    try:
        db = await get_db()
        collection = db["facial_analyses"]
        
        # Calculate date range
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Fetch analysis data
        cursor = collection.find({
            "user_id": user_id,
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }).sort("timestamp", 1)
        
        analyses = await cursor.to_list(length=None)
        
        if not analyses:
            return {
                "user_id": user_id,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "summary": generate_empty_summary(),
                "trends": {},
                "insights": [],
                "recommendations": []
            }
        
        # Generate dashboard data
        dashboard_data = generate_dashboard_data(analyses, days)
        dashboard_data["user_id"] = user_id
        dashboard_data["date_range"] = {
            "start": start_date.isoformat(),
            "end": end_date.isoformat()
        }
        
        return dashboard_data
        
    except Exception as e:
        logger.error(f"Error generating facial dashboard: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard")

@router.get("/real-time/{user_id}")
async def get_real_time_metrics(
    user_id: str,
    session_id: Optional[str] = Query(None),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get real-time facial analysis metrics for live dashboard."""
    try:
        db = await get_db()
        collection = db["facial_analyses"]
        
        # Get latest analysis
        query = {"user_id": user_id}
        if session_id:
            query["session_id"] = session_id
        
        latest_analysis = await collection.find_one(
            query,
            sort=[("timestamp", -1)]
        )
        
        if not latest_analysis:
            return {"status": "no_data", "message": "No recent analysis data available"}
        
        # Get session status if session_id provided
        session_status = None
        if session_id and session_id in active_sessions:
            session = active_sessions[session_id]
            current_time = datetime.utcnow()
            duration = (current_time - session.start_time).total_seconds()
            
            session_status = {
                "active": True,
                "duration_seconds": duration,
                "total_frames": session.total_frames_analyzed,
                "average_quality": session.average_quality
            }
        
        # Format real-time data
        real_time_data = {
            "status": "active",
            "timestamp": latest_analysis["timestamp"],
            "current_metrics": {
                "mood": latest_analysis["mood_assessment"],
                "primary_emotion": latest_analysis["primary_emotion"],
                "emotion_confidence": latest_analysis["emotion_confidence"],
                "sleepiness_level": latest_analysis["sleepiness"]["level"],
                "fatigue_detected": latest_analysis["fatigue"]["overall_fatigue"],
                "stress_level": latest_analysis["stress"]["level"],
                "phq9_score": latest_analysis["phq9_estimation"]["estimated_score"],
                "analysis_quality": latest_analysis["analysis_quality"]
            },
            "raw_metrics": {
                "eye_aspect_ratio": latest_analysis.get("eye_metrics", {}).get("avg_ear", 0),
                "head_angles": {
                    "pitch": latest_analysis.get("head_pose", {}).get("pitch", 0),
                    "yaw": latest_analysis.get("head_pose", {}).get("yaw", 0),
                    "roll": latest_analysis.get("head_pose", {}).get("roll", 0)
                },
                "micro_expressions": latest_analysis.get("micro_expressions", {})
            },
            "session": session_status
        }
        
        return real_time_data
        
    except Exception as e:
        logger.error(f"Error getting real-time metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get real-time metrics")

@router.get("/insights/{user_id}")
async def get_facial_insights(
    user_id: str,
    days: int = Query(7, ge=1, le=30),
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """Get personalized insights based on facial analysis history."""
    try:
        db = await get_db()
        collection = db["facial_analyses"]
        
        # Get analysis data
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        cursor = collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }).sort("timestamp", 1)
        
        analyses = await cursor.to_list(length=None)
        
        if not analyses:
            return {"insights": [], "recommendations": []}
        
        insights = generate_personalized_insights(analyses)
        recommendations = generate_facial_recommendations(analyses)
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "analysis_period": f"{days} days",
            "total_analyses": len(analyses)
        }
        
    except Exception as e:
        logger.error(f"Error generating facial insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate insights")

# Helper functions

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

async def store_analysis_result(result: ComprehensiveFacialAnalysis, user_id: str, session_id: Optional[str]):
    """Store facial analysis result in database."""
    try:
        db = await get_db()
        collection = db["facial_analyses"]
        
        # Convert to dictionary for storage
        data = result.dict()
        data["user_id"] = user_id
        if session_id:
            data["session_id"] = session_id
        
        await collection.insert_one(data)
        
    except Exception as e:
        logger.error(f"Error storing analysis result: {e}")

async def store_enhanced_analysis_result(result: EnhancedFacialAnalysisResult, user_id: str):
    """Store enhanced facial analysis result with health recommendations in database."""
    try:
        db = await get_db()
        collection = db["enhanced_facial_analyses"]
        
        # Convert to dictionary for storage
        data = result.dict()
        data["user_id"] = user_id
        
        await collection.insert_one(data)
        
    except Exception as e:
        logger.error(f"Error storing enhanced analysis result: {e}")

async def store_session_data(session: FacialAnalysisSession):
    """Store session data in database."""
    try:
        db = await get_db()
        collection = db["facial_sessions"]
        
        await collection.insert_one(session.dict())
        
    except Exception as e:
        logger.error(f"Error storing session data: {e}")

def generate_dashboard_data(analyses: List[Dict], days: int) -> Dict[str, Any]:
    """Generate comprehensive dashboard data from analyses."""
    if not analyses:
        return generate_empty_summary()
    
    # Calculate trends
    mood_trend = calculate_mood_trend(analyses)
    stress_trend = calculate_stress_trend(analyses)
    sleep_trend = calculate_sleep_trend(analyses)
    phq9_trend = calculate_phq9_trend(analyses)
    
    # Calculate averages
    avg_quality = sum(a.get("analysis_quality", 0) for a in analyses) / len(analyses)
    avg_phq9 = sum(a.get("phq9_estimation", {}).get("estimated_score", 0) for a in analyses) / len(analyses)
    
    # Count fatigue incidents
    fatigue_incidents = sum(1 for a in analyses if a.get("fatigue", {}).get("overall_fatigue", False))
    fatigue_rate = fatigue_incidents / len(analyses) * 100
    
    # Most common mood
    moods = [a.get("mood_assessment", "Unknown") for a in analyses]
    most_common_mood = max(set(moods), key=moods.count) if moods else "Unknown"
    
    return {
        "summary": {
            "total_analyses": len(analyses),
            "average_quality": round(avg_quality, 2),
            "average_phq9_score": round(avg_phq9, 1),
            "fatigue_incident_rate": round(fatigue_rate, 1),
            "most_common_mood": most_common_mood,
            "analysis_period_days": days
        },
        "trends": {
            "mood": mood_trend,
            "stress": stress_trend,
            "sleepiness": sleep_trend,
            "phq9_scores": phq9_trend
        },
        "insights": generate_trend_insights(analyses),
        "recommendations": generate_facial_recommendations(analyses)
    }

def calculate_mood_trend(analyses: List[Dict]) -> List[Dict]:
    """Calculate mood trend over time."""
    mood_mapping = {"Happy": 5, "Neutral": 3, "Sad": 1, "Angry": 2, "Surprised": 4}
    
    daily_moods = {}
    for analysis in analyses:
        date = analysis["timestamp"].strftime("%Y-%m-%d")
        mood = analysis.get("mood_assessment", "Neutral")
        mood_score = mood_mapping.get(mood, 3)
        
        if date not in daily_moods:
            daily_moods[date] = []
        daily_moods[date].append(mood_score)
    
    return [
        {"date": date, "mood_score": sum(scores) / len(scores)}
        for date, scores in sorted(daily_moods.items())
    ]

def calculate_stress_trend(analyses: List[Dict]) -> List[Dict]:
    """Calculate stress level trend over time."""
    stress_mapping = {"Low": 1, "Medium": 2, "High": 3}
    
    daily_stress = {}
    for analysis in analyses:
        date = analysis["timestamp"].strftime("%Y-%m-%d")
        stress = analysis.get("stress", {}).get("level", "Low")
        stress_score = stress_mapping.get(stress, 1)
        
        if date not in daily_stress:
            daily_stress[date] = []
        daily_stress[date].append(stress_score)
    
    return [
        {"date": date, "stress_level": sum(scores) / len(scores)}
        for date, scores in sorted(daily_stress.items())
    ]

def calculate_sleep_trend(analyses: List[Dict]) -> List[Dict]:
    """Calculate sleepiness trend over time."""
    sleep_mapping = {"Alert": 3, "Slightly tired": 2, "Very tired": 1}
    
    daily_sleep = {}
    for analysis in analyses:
        date = analysis["timestamp"].strftime("%Y-%m-%d")
        sleepiness = analysis.get("sleepiness", {}).get("level", "Alert")
        sleep_score = sleep_mapping.get(sleepiness, 3)
        
        if date not in daily_sleep:
            daily_sleep[date] = []
        daily_sleep[date].append(sleep_score)
    
    return [
        {"date": date, "alertness_level": sum(scores) / len(scores)}
        for date, scores in sorted(daily_sleep.items())
    ]

def calculate_phq9_trend(analyses: List[Dict]) -> List[Dict]:
    """Calculate PHQ-9 score trend over time."""
    daily_phq9 = {}
    for analysis in analyses:
        date = analysis["timestamp"].strftime("%Y-%m-%d")
        phq9_score = analysis.get("phq9_estimation", {}).get("estimated_score", 0)
        
        if date not in daily_phq9:
            daily_phq9[date] = []
        daily_phq9[date].append(phq9_score)
    
    return [
        {"date": date, "phq9_score": sum(scores) / len(scores)}
        for date, scores in sorted(daily_phq9.items())
    ]

def generate_trend_insights(analyses: List[Dict]) -> List[str]:
    """Generate insights based on trend analysis."""
    insights = []
    
    if len(analyses) < 2:
        return ["Not enough data for trend analysis"]
    
    # Analyze PHQ-9 trend
    recent_phq9 = [a.get("phq9_estimation", {}).get("estimated_score", 0) for a in analyses[-7:]]
    older_phq9 = [a.get("phq9_estimation", {}).get("estimated_score", 0) for a in analyses[-14:-7]] if len(analyses) >= 14 else []
    
    if older_phq9:
        recent_avg = sum(recent_phq9) / len(recent_phq9)
        older_avg = sum(older_phq9) / len(older_phq9)
        
        if recent_avg < older_avg - 2:
            insights.append("Your mental health indicators have improved significantly over the past week")
        elif recent_avg > older_avg + 2:
            insights.append("Your mental health indicators suggest increased stress - consider reaching out for support")
    
    # Analyze fatigue patterns
    fatigue_count = sum(1 for a in analyses if a.get("fatigue", {}).get("overall_fatigue", False))
    if fatigue_count > len(analyses) * 0.6:
        insights.append("High fatigue levels detected - consider improving sleep hygiene")
    
    # Analyze stress patterns
    high_stress_count = sum(1 for a in analyses if a.get("stress", {}).get("level") == "High")
    if high_stress_count > len(analyses) * 0.4:
        insights.append("Frequent high stress levels detected - stress management techniques may be helpful")
    
    return insights

def generate_facial_recommendations(analyses: List[Dict]) -> List[str]:
    """Generate personalized recommendations based on facial analysis."""
    recommendations = []
    
    if not analyses:
        return recommendations
    
    # Analyze recent patterns
    recent_analyses = analyses[-10:] if len(analyses) >= 10 else analyses
    
    # Sleep recommendations
    tired_count = sum(1 for a in recent_analyses 
                     if a.get("sleepiness", {}).get("level") in ["Slightly tired", "Very tired"])
    if tired_count > len(recent_analyses) * 0.5:
        recommendations.append("Consider establishing a consistent sleep schedule and limiting screen time before bed")
    
    # Stress recommendations
    stressed_count = sum(1 for a in recent_analyses 
                        if a.get("stress", {}).get("level") in ["Medium", "High"])
    if stressed_count > len(recent_analyses) * 0.4:
        recommendations.append("Try relaxation techniques like deep breathing or meditation")
    
    # Mood recommendations
    negative_moods = sum(1 for a in recent_analyses 
                        if a.get("mood_assessment") in ["Sad", "Angry"])
    if negative_moods > len(recent_analyses) * 0.6:
        recommendations.append("Consider engaging in activities that bring you joy or talking to a mental health professional")
    
    # PHQ-9 recommendations
    avg_phq9 = sum(a.get("phq9_estimation", {}).get("estimated_score", 0) for a in recent_analyses) / len(recent_analyses)
    if avg_phq9 > 15:
        recommendations.append("Your analysis suggests significant mental health concerns - please consider speaking with a healthcare provider")
    elif avg_phq9 > 10:
        recommendations.append("Consider self-care practices and monitoring your mental health regularly")
    
    return recommendations

def generate_personalized_insights(analyses: List[Dict]) -> List[str]:
    """Generate personalized insights from facial analysis history."""
    insights = []
    
    if not analyses:
        return insights
    
    # Time-based patterns
    morning_analyses = [a for a in analyses if 6 <= a["timestamp"].hour <= 12]
    evening_analyses = [a for a in analyses if 18 <= a["timestamp"].hour <= 23]
    
    if morning_analyses and evening_analyses:
        morning_avg_phq9 = sum(a.get("phq9_estimation", {}).get("estimated_score", 0) for a in morning_analyses) / len(morning_analyses)
        evening_avg_phq9 = sum(a.get("phq9_estimation", {}).get("estimated_score", 0) for a in evening_analyses) / len(evening_analyses)
        
        if morning_avg_phq9 < evening_avg_phq9 - 3:
            insights.append("You tend to feel better in the mornings - consider scheduling important activities earlier in the day")
        elif evening_avg_phq9 < morning_avg_phq9 - 3:
            insights.append("Your mood tends to improve throughout the day")
    
    # Quality patterns
    high_quality_analyses = [a for a in analyses if a.get("analysis_quality", 0) > 0.7]
    if len(high_quality_analyses) < len(analyses) * 0.5:
        insights.append("Consider improving lighting conditions for more accurate analysis")
    
    return insights

def generate_empty_summary() -> Dict[str, Any]:
    """Generate empty summary when no data is available."""
    return {
        "summary": {
            "total_analyses": 0,
            "average_quality": 0,
            "average_phq9_score": 0,
            "fatigue_incident_rate": 0,
            "most_common_mood": "Unknown",
            "analysis_period_days": 0
        },
        "trends": {
            "mood": [],
            "stress": [],
            "sleepiness": [],
            "phq9_scores": []
        },
        "insights": ["No analysis data available"],
        "recommendations": ["Start using facial analysis to get personalized insights"]
    }

def analyze_user_patterns_and_recommend(analyses: List[Dict]) -> List[Dict[str, Any]]:
    """Analyze user patterns from historical data and generate personalized recommendations."""
    try:
        if not analyses:
            return []
        
        recommendations = []
        
        # Analyze mood patterns
        recent_moods = [a.get("health_recommendations", {}).get("mood_category", "neutral") for a in analyses[:5]]
        stress_levels = [a.get("health_recommendations", {}).get("biometric_indicators", {}).get("stress_level", 0.5) for a in analyses[:5]]
        
        # High stress pattern
        if sum(stress_levels) / len(stress_levels) > 0.7:
            recommendations.append({
                "type": "stress_management",
                "priority": "high",
                "title": "Stress Management Recommended",
                "description": "Your recent analyses show elevated stress levels. Consider regular breathing exercises and mindfulness practices.",
                "suggested_exercises": ["4-7-8 Breathing Technique", "Progressive Muscle Relaxation"],
                "suggested_games": ["Rhythmic Breathing Game", "Guided Visualization"]
            })
        
        # Fatigue pattern
        fatigue_count = sum(1 for a in analyses[:10] if a.get("health_recommendations", {}).get("health_status") in ["concerning", "critical"])
        if fatigue_count > 3:
            recommendations.append({
                "type": "fatigue_management",
                "priority": "medium",
                "title": "Energy Management Tips",
                "description": "You've shown signs of fatigue in recent sessions. Focus on rest and gentle exercises.",
                "suggested_exercises": ["20-20-20 Eye Exercise", "Gentle Stretching"],
                "lifestyle_tips": ["Ensure 7-9 hours of sleep", "Take regular breaks", "Stay hydrated"]
            })
        
        # Positive trend
        if len(analyses) > 1:
            first_mood = analyses[-1].get("health_recommendations", {}).get("biometric_indicators", {}).get("stress_level", 0.5)
            recent_mood = analyses[0].get("health_recommendations", {}).get("biometric_indicators", {}).get("stress_level", 0.5)
            
            if first_mood > recent_mood + 0.2:  # Improvement
                recommendations.append({
                    "type": "positive_reinforcement",
                    "priority": "low",
                    "title": "Great Progress!",
                    "description": "Your stress levels have been improving. Keep up the good work!",
                    "encouragement": "Continue with your current routine and consider gradually adding more challenging exercises."
                })
        
        return recommendations
        
    except Exception as e:
        logger.error(f"Error analyzing user patterns: {e}")
        return []
