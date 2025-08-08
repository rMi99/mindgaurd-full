from fastapi import APIRouter, HTTPException, Query
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.db import db

router = APIRouter()

class DashboardAssessment(BaseModel):
    phq9Score: int
    riskLevel: str
    sleepData: Optional[Dict[str, Any]] = None

class DashboardSaveRequest(BaseModel):
    userId: str
    assessment: DashboardAssessment

class AssessmentHistory(BaseModel):
    date: str
    phq9Score: int
    riskLevel: str
    sleepHours: Optional[str] = None
    sleepQuality: Optional[str] = None
    sleepHoursNumeric: Optional[float] = None
    stressLevel: Optional[str] = None
    exerciseFrequency: Optional[str] = None
    socialSupport: Optional[str] = None

class HistoricalTrend(BaseModel):
    overallTrend: str
    phq9Trend: float
    sleepTrend: str
    insights: List[str]
    recommendations: List[str]
    correlations: Dict[str, float]

class DashboardResponse(BaseModel):
    history: List[AssessmentHistory]
    trends: Optional[HistoricalTrend] = None

def convert_sleep_hours_to_numeric(sleep_hours: str) -> float:
    """Convert sleep hours string to numeric value"""
    sleep_hours_map = {
        "<4": 3.0,
        "4-5": 4.5,
        "5-6": 5.5,
        "6-7": 6.5,
        "7-8": 7.5,
        "8-9": 8.5,
        ">9": 9.5
    }
    return sleep_hours_map.get(sleep_hours, 7.0)

def analyze_trends(history: List[AssessmentHistory]) -> Optional[HistoricalTrend]:
    """Analyze trends from assessment history"""
    if len(history) < 2:
        return None
    
    # Sort by date
    sorted_history = sorted(history, key=lambda x: x.date)
    
    # Calculate PHQ-9 trend
    phq9_scores = [h.phq9Score for h in sorted_history]
    phq9_trend = (phq9_scores[-1] - phq9_scores[0]) / len(phq9_scores)
    
    # Determine overall trend
    if phq9_trend < -1:
        overall_trend = "improving"
    elif phq9_trend > 1:
        overall_trend = "declining"
    else:
        overall_trend = "stable"
    
    # Analyze sleep trend
    sleep_scores = [convert_sleep_hours_to_numeric(h.sleepHours or "7-8") for h in sorted_history if h.sleepHours]
    if len(sleep_scores) >= 2:
        sleep_change = sleep_scores[-1] - sleep_scores[0]
        if sleep_change > 0.5:
            sleep_trend = "improving"
        elif sleep_change < -0.5:
            sleep_trend = "declining"
        else:
            sleep_trend = "stable"
    else:
        sleep_trend = "stable"
    
    # Generate insights
    insights = []
    if overall_trend == "improving":
        insights.append("Your mental health scores show improvement over time")
    elif overall_trend == "declining":
        insights.append("Your mental health scores indicate a concerning trend")
    
    if sleep_trend == "improving":
        insights.append("Your sleep patterns are improving")
    elif sleep_trend == "declining":
        insights.append("Your sleep quality may need attention")
    
    # Generate recommendations
    recommendations = []
    if overall_trend == "declining":
        recommendations.append("Consider speaking with a healthcare professional")
        recommendations.append("Focus on stress management techniques")
    
    if sleep_trend == "declining":
        recommendations.append("Prioritize consistent sleep schedule")
        recommendations.append("Limit screen time before bed")
    
    # Simple correlations (placeholder)
    correlations = {
        "sleep_mood": 0.7 if sleep_trend == overall_trend else 0.3,
        "stress_mood": -0.6,
        "exercise_mood": 0.5
    }
    
    return HistoricalTrend(
        overallTrend=overall_trend,
        phq9Trend=phq9_trend,
        sleepTrend=sleep_trend,
        insights=insights,
        recommendations=recommendations,
        correlations=correlations
    )

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(userId: str = Query(...)):
    """Get dashboard data for a user"""
    try:
        # Get user's assessment history from database
        cursor = db.user_assessments.find({"user_id": userId}).sort("timestamp", -1).limit(50)
        assessments = await cursor.to_list(length=50)
        
        # Convert to frontend format
        history = []
        for assessment in assessments:
            sleep_data = assessment.get("sleep_data", {})
            history_item = AssessmentHistory(
                date=assessment.get("timestamp", datetime.utcnow().isoformat()),
                phq9Score=assessment.get("phq9_score", 0),
                riskLevel=assessment.get("risk_level", "low"),
                sleepHours=sleep_data.get("sleepHours"),
                sleepQuality=sleep_data.get("sleepQuality"),
                sleepHoursNumeric=convert_sleep_hours_to_numeric(sleep_data.get("sleepHours", "7-8")),
                stressLevel=sleep_data.get("stressLevel"),
                exerciseFrequency=sleep_data.get("exerciseFrequency"),
                socialSupport=sleep_data.get("socialSupport")
            )
            history.append(history_item)
        
        # Analyze trends if we have enough data
        trends = analyze_trends(history) if len(history) >= 2 else None
        
        return DashboardResponse(history=history, trends=trends)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard data: {str(e)}")

@router.post("/dashboard")
async def save_assessment(request: DashboardSaveRequest):
    """Save assessment data to user history"""
    try:
        # Create assessment record
        assessment_record = {
            "user_id": request.userId,
            "timestamp": datetime.utcnow().isoformat(),
            "phq9_score": request.assessment.phq9Score,
            "risk_level": request.assessment.riskLevel,
            "sleep_data": request.assessment.sleepData or {}
        }
        
        # Save to database
        await db.user_assessments.insert_one(assessment_record)
        
        return {"success": True, "message": "Assessment saved successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save assessment: {str(e)}")

