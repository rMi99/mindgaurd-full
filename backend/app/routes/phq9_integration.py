"""
PHQ-9 Integration routes for auto-filling depression screening questionnaires
based on facial analysis results.
"""

from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional, Dict, Any
import logging
from datetime import datetime

from app.models.facial_metrics import FacialAnalysisRequest
from app.services.phq9_integration import PHQ9IntegrationService
from app.services.enhanced_facial_analyzer import EnhancedFacialAnalyzer
from app.services.db import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/phq9-integration", tags=["phq9-integration"])
security = HTTPBearer(auto_error=False)

# Initialize services
phq9_service = PHQ9IntegrationService()
facial_analyzer = EnhancedFacialAnalyzer()

@router.post("/auto-fill")
async def auto_fill_phq9(
    request: FacialAnalysisRequest,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Auto-fill PHQ-9 questionnaire based on facial analysis.
    
    This endpoint analyzes facial expressions and automatically generates
    PHQ-9 responses with confidence scores and reasoning.
    """
    try:
        logger.info("Received PHQ-9 auto-fill request")
        
        # Decode and analyze the image
        from app.routes.facial_dashboard import decode_base64_image
        image = decode_base64_image(request.image)
        
        # Perform comprehensive facial analysis
        facial_analysis = facial_analyzer.analyze_comprehensive(
            image=image,
            session_id=request.session_id,
            user_id=request.user_id
        )
        
        # Generate auto-fill data
        auto_fill_data = phq9_service.generate_phq9_auto_fill(facial_analysis)
        
        # Store analysis result if user_id provided
        if request.user_id:
            await store_phq9_analysis(request.user_id, auto_fill_data, facial_analysis)
        
        logger.info(f"PHQ-9 auto-fill generated: score {auto_fill_data['estimated_total_score']}")
        
        return {
            "status": "success",
            "data": auto_fill_data,
            "facial_analysis_summary": {
                "primary_emotion": facial_analysis.primary_emotion,
                "mood_assessment": facial_analysis.mood_assessment,
                "sleepiness_level": facial_analysis.sleepiness.level,
                "stress_level": facial_analysis.stress.level,
                "fatigue_detected": facial_analysis.fatigue.overall_fatigue,
                "analysis_quality": facial_analysis.analysis_quality
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error in PHQ-9 auto-fill: {e}")
        raise HTTPException(status_code=500, detail="Internal server error during PHQ-9 auto-fill")

@router.post("/validate")
async def validate_phq9_responses(
    responses: Dict[str, int],
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Validate and analyze PHQ-9 responses.
    
    Args:
        responses: Dictionary with question IDs and scores (0-3)
        
    Returns:
        Analysis of PHQ-9 responses with recommendations
    """
    try:
        logger.info("Received PHQ-9 validation request")
        
        # Validate responses
        validation_result = phq9_service.validate_phq9_responses(responses)
        
        # Add timestamp and metadata
        validation_result.update({
            "timestamp": datetime.utcnow().isoformat(),
            "total_questions": len(responses),
            "response_completeness": len(responses) / 9.0  # 9 PHQ-9 questions
        })
        
        logger.info(f"PHQ-9 validation complete: score {validation_result['total_score']}, severity {validation_result['severity_level']}")
        
        return {
            "status": "success",
            "data": validation_result
        }
        
    except Exception as e:
        logger.error(f"Error validating PHQ-9 responses: {e}")
        raise HTTPException(status_code=500, detail="Failed to validate PHQ-9 responses")

@router.get("/history/{user_id}")
async def get_phq9_history(
    user_id: str,
    days: int = 30,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Get PHQ-9 analysis history for a user.
    
    Args:
        user_id: User identifier
        days: Number of days to include in history
        
    Returns:
        Historical PHQ-9 analysis data
    """
    try:
        db = await get_db()
        collection = db["phq9_analyses"]
        
        # Calculate date range
        from datetime import timedelta
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        # Fetch analysis data
        cursor = collection.find({
            "user_id": user_id,
            "timestamp": {
                "$gte": start_date,
                "$lte": end_date
            }
        }).sort("timestamp", -1)
        
        analyses = await cursor.to_list(length=None)
        
        if not analyses:
            return {
                "user_id": user_id,
                "date_range": {
                    "start": start_date.isoformat(),
                    "end": end_date.isoformat()
                },
                "analyses": [],
                "summary": {
                    "total_analyses": 0,
                    "average_score": 0,
                    "trend": "no_data"
                }
            }
        
        # Calculate summary statistics
        scores = [a.get("estimated_total_score", 0) for a in analyses]
        avg_score = sum(scores) / len(scores) if scores else 0
        
        # Determine trend
        if len(scores) >= 2:
            recent_avg = sum(scores[:len(scores)//2]) / (len(scores)//2)
            older_avg = sum(scores[len(scores)//2:]) / (len(scores) - len(scores)//2)
            
            if recent_avg < older_avg - 1:
                trend = "improving"
            elif recent_avg > older_avg + 1:
                trend = "worsening"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        return {
            "user_id": user_id,
            "date_range": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "analyses": analyses,
            "summary": {
                "total_analyses": len(analyses),
                "average_score": round(avg_score, 1),
                "trend": trend,
                "latest_score": scores[0] if scores else 0,
                "highest_score": max(scores) if scores else 0,
                "lowest_score": min(scores) if scores else 0
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting PHQ-9 history: {e}")
        raise HTTPException(status_code=500, detail="Failed to get PHQ-9 history")

@router.get("/insights/{user_id}")
async def get_phq9_insights(
    user_id: str,
    days: int = 30,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)
):
    """
    Get personalized insights based on PHQ-9 analysis history.
    
    Args:
        user_id: User identifier
        days: Number of days to analyze
        
    Returns:
        Personalized insights and recommendations
    """
    try:
        db = await get_db()
        collection = db["phq9_analyses"]
        
        # Get recent analyses
        from datetime import timedelta
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        cursor = collection.find({
            "user_id": user_id,
            "timestamp": {"$gte": start_date, "$lte": end_date}
        }).sort("timestamp", -1)
        
        analyses = await cursor.to_list(length=None)
        
        if not analyses:
            return {
                "insights": ["No PHQ-9 analysis data available for the specified period"],
                "recommendations": ["Start using facial analysis to get personalized insights"],
                "trend_analysis": "insufficient_data"
            }
        
        # Generate insights
        insights = []
        recommendations = []
        
        # Score trend analysis
        scores = [a.get("estimated_total_score", 0) for a in analyses]
        if len(scores) >= 3:
            recent_scores = scores[:3]
            older_scores = scores[3:6] if len(scores) >= 6 else scores[3:]
            
            if older_scores:
                recent_avg = sum(recent_scores) / len(recent_scores)
                older_avg = sum(older_scores) / len(older_scores)
                
                if recent_avg < older_avg - 2:
                    insights.append("Your mental health indicators have shown improvement over recent analyses")
                    recommendations.append("Continue with current self-care practices")
                elif recent_avg > older_avg + 2:
                    insights.append("Your mental health indicators suggest increased stress or depression risk")
                    recommendations.append("Consider reaching out to a mental health professional")
        
        # Severity level analysis
        severity_levels = [a.get("severity_level", "Unknown") for a in analyses]
        most_common_severity = max(set(severity_levels), key=severity_levels.count)
        
        if most_common_severity in ["Moderately severe", "Severe"]:
            insights.append("Consistent high depression risk detected in recent analyses")
            recommendations.append("Professional mental health evaluation strongly recommended")
        elif most_common_severity == "Moderate":
            insights.append("Moderate depression risk patterns observed")
            recommendations.append("Consider therapy or counseling for support")
        
        # Quality analysis
        quality_scores = [a.get("confidence", 0) for a in analyses]
        avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
        
        if avg_quality < 0.6:
            insights.append("Analysis quality has been inconsistent - ensure good lighting and camera positioning")
            recommendations.append("Improve analysis conditions for more accurate results")
        
        return {
            "insights": insights if insights else ["Continue monitoring your mental health regularly"],
            "recommendations": recommendations if recommendations else ["Maintain current self-care practices"],
            "trend_analysis": {
                "score_trend": "improving" if len(scores) >= 2 and scores[0] < scores[-1] else "stable",
                "average_quality": round(avg_quality, 2),
                "most_common_severity": most_common_severity
            }
        }
        
    except Exception as e:
        logger.error(f"Error generating PHQ-9 insights: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate insights")

async def store_phq9_analysis(user_id: str, auto_fill_data: Dict, facial_analysis: Any):
    """Store PHQ-9 analysis result in database."""
    try:
        db = await get_db()
        collection = db["phq9_analyses"]
        
        # Prepare data for storage
        data = {
            "user_id": user_id,
            "timestamp": datetime.utcnow(),
            "estimated_total_score": auto_fill_data["estimated_total_score"],
            "severity_level": auto_fill_data["severity_level"],
            "confidence": auto_fill_data["confidence"],
            "responses": auto_fill_data["responses"],
            "reasoning": auto_fill_data["reasoning"],
            "recommendations": auto_fill_data["recommendations"],
            "facial_analysis_summary": {
                "primary_emotion": facial_analysis.primary_emotion,
                "mood_assessment": facial_analysis.mood_assessment,
                "sleepiness_level": facial_analysis.sleepiness.level,
                "stress_level": facial_analysis.stress.level,
                "fatigue_detected": facial_analysis.fatigue.overall_fatigue
            }
        }
        
        await collection.insert_one(data)
        
    except Exception as e:
        logger.error(f"Error storing PHQ-9 analysis: {e}")
