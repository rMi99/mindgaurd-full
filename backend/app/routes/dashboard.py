from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.db import db
from app.routes.auth import get_current_user

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

class PersonalizedInsights(BaseModel):
    encouragingMessage: str
    psychologicalInsights: List[str]
    personalizedRecommendations: List[str]
    progressSummary: str
    nextSteps: List[str]

class DashboardResponse(BaseModel):
    history: List[AssessmentHistory]
    trends: Optional[HistoricalTrend] = None
    personalizedInsights: Optional[PersonalizedInsights] = None
    userInfo: Dict[str, Any]

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

def generate_personalized_insights(history: List[AssessmentHistory], user_info: Dict[str, Any]) -> PersonalizedInsights:
    """Generate personalized insights and encouraging messages"""
    username = user_info.get("username", "there")
    is_temporary = user_info.get("is_temporary", False)
    
    # Base encouraging messages
    encouraging_messages = [
        f"Hello {username}! Remember, taking care of your mental health is a sign of strength.",
        f"Hi {username}! Every step you take towards understanding your mental health matters.",
        f"Welcome back, {username}! Your commitment to mental wellness is inspiring.",
        f"Great to see you, {username}! You're doing important work by monitoring your wellbeing."
    ]
    
    if is_temporary:
        encouraging_messages = [
            "Hello! Thank you for taking the time to assess your mental health.",
            "Welcome! Your mental health journey is important, and we're here to support you.",
            "Hi there! Taking care of your mental health is a brave and important step."
        ]
    
    # Analyze history for insights
    psychological_insights = []
    recommendations = []
    progress_summary = "You're taking positive steps by monitoring your mental health."
    next_steps = []
    
    if len(history) == 0:
        psychological_insights = [
            "Starting your mental health journey is a positive step forward.",
            "Regular self-assessment can help you understand patterns in your wellbeing.",
            "Building awareness of your mental state is the foundation of good mental health."
        ]
        recommendations = [
            "Take your first assessment to establish a baseline",
            "Consider setting a regular schedule for mental health check-ins",
            "Explore mindfulness and relaxation techniques"
        ]
        next_steps = [
            "Complete your first mental health assessment",
            "Set up a routine for regular check-ins",
            "Learn about mental health resources available to you"
        ]
    elif len(history) == 1:
        psychological_insights = [
            "You've taken the first step in understanding your mental health patterns.",
            "Consistency in self-assessment will help reveal important trends over time.",
            "One assessment provides a snapshot, but regular monitoring shows the full picture."
        ]
        recommendations = [
            "Continue with regular assessments to track patterns",
            "Pay attention to factors that might influence your mood",
            "Consider keeping a simple mood journal"
        ]
        next_steps = [
            "Take another assessment in a few days",
            "Notice what activities or situations affect your mood",
            "Begin tracking sleep and exercise patterns"
        ]
    else:
        # Analyze trends for more specific insights
        recent_scores = [h.phq9Score for h in history[-3:]]
        avg_recent = sum(recent_scores) / len(recent_scores)
        
        if avg_recent <= 4:
            psychological_insights = [
                "Your recent assessments show minimal depression symptoms.",
                "You're maintaining good mental health awareness and self-care.",
                "Your consistent monitoring demonstrates strong mental health habits."
            ]
            recommendations = [
                "Continue your current self-care practices",
                "Maintain regular sleep and exercise routines",
                "Stay connected with supportive relationships"
            ]
        elif avg_recent <= 9:
            psychological_insights = [
                "Your assessments indicate mild depression symptoms that are worth monitoring.",
                "You're being proactive by tracking your mental health regularly.",
                "Small changes in routine can have significant positive impacts."
            ]
            recommendations = [
                "Focus on maintaining consistent sleep schedules",
                "Incorporate regular physical activity into your routine",
                "Practice stress management techniques like deep breathing"
            ]
        else:
            psychological_insights = [
                "Your recent assessments suggest you may be experiencing significant symptoms.",
                "Seeking professional support can be very beneficial at this time.",
                "You're taking important steps by monitoring your mental health."
            ]
            recommendations = [
                "Consider speaking with a mental health professional",
                "Reach out to trusted friends or family for support",
                "Prioritize basic self-care: sleep, nutrition, and gentle exercise"
            ]
        
        # Progress summary based on trend
        if len(history) >= 3:
            trend_direction = "stable"
            if recent_scores[-1] < recent_scores[0]:
                trend_direction = "improving"
                progress_summary = "Your recent assessments show positive trends in your mental health."
            elif recent_scores[-1] > recent_scores[0]:
                trend_direction = "needs attention"
                progress_summary = "Your recent assessments suggest your mental health may need extra attention."
            else:
                progress_summary = "Your mental health appears stable based on recent assessments."
        
        next_steps = [
            "Continue regular mental health monitoring",
            "Implement one new self-care strategy this week",
            "Review your progress patterns monthly"
        ]
    
    return PersonalizedInsights(
        encouragingMessage=encouraging_messages[0],
        psychologicalInsights=psychological_insights,
        personalizedRecommendations=recommendations,
        progressSummary=progress_summary,
        nextSteps=next_steps
    )

@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(userId: Optional[str] = Query(None), current_user: Optional[str] = Depends(get_current_user)):
    """Get personalized dashboard data for a user with strict privacy controls"""
    try:
        # Determine the actual user ID to use
        actual_user_id = None
        user_info = {}
        
        if current_user:
            # Authenticated user - use their ID regardless of query parameter
            actual_user_id = current_user
            user = await db.users.find_one({"user_id": current_user})
            if user:
                user_info = {
                    "user_id": user["user_id"],
                    "email": user.get("email"),
                    "username": user.get("username"),
                    "is_temporary": False
                }
        elif userId:
            # Check if it's a temporary user
            temp_user = await db.temp_users.find_one({"temp_user_id": userId})
            if temp_user and not temp_user.get("linked_to_registered"):
                actual_user_id = userId
                user_info = {
                    "user_id": userId,
                    "is_temporary": True,
                    "username": "Guest"
                }
            else:
                raise HTTPException(status_code=403, detail="Access denied: Invalid temporary user")
        else:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if not actual_user_id:
            raise HTTPException(status_code=401, detail="User identification failed")
        
        # Get user's assessment history from database (only their own data)
        cursor = db.user_assessments.find({"user_id": actual_user_id}).sort("timestamp", -1).limit(50)
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
        
        # Generate personalized insights
        personalized_insights = generate_personalized_insights(history, user_info)
        
        return DashboardResponse(
            history=history, 
            trends=trends,
            personalizedInsights=personalized_insights,
            userInfo=user_info
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch dashboard data: {str(e)}")

@router.post("/dashboard")
async def save_assessment(request: DashboardSaveRequest, current_user: Optional[str] = Depends(get_current_user)):
    """Save assessment data to user history with privacy controls"""
    try:
        # Determine the actual user ID and validate access
        actual_user_id = None
        
        if current_user:
            # Authenticated user - use their ID regardless of request userId
            actual_user_id = current_user
        elif request.userId:
            # Check if it's a valid temporary user
            temp_user = await db.temp_users.find_one({"temp_user_id": request.userId})
            if temp_user and not temp_user.get("linked_to_registered"):
                actual_user_id = request.userId
            else:
                raise HTTPException(status_code=403, detail="Access denied: Invalid temporary user")
        else:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        if not actual_user_id:
            raise HTTPException(status_code=401, detail="User identification failed")
        
        # Create assessment record with the validated user ID
        assessment_record = {
            "user_id": actual_user_id,
            "timestamp": datetime.utcnow().isoformat(),
            "phq9_score": request.assessment.phq9Score,
            "risk_level": request.assessment.riskLevel,
            "sleep_data": request.assessment.sleepData or {}
        }
        
        # Save to database
        await db.user_assessments.insert_one(assessment_record)
        
        return {"success": True, "message": "Assessment saved successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save assessment: {str(e)}")

@router.delete("/dashboard/history/{assessment_id}")
async def delete_assessment(assessment_id: str, current_user: str = Depends(get_current_user)):
    """Delete a specific assessment from user's history"""
    try:
        # Find and delete the assessment, ensuring it belongs to the current user
        result = await db.user_assessments.delete_one({
            "_id": assessment_id,
            "user_id": current_user
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Assessment not found or access denied")
        
        return {"success": True, "message": "Assessment deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete assessment: {str(e)}")

@router.delete("/dashboard/history")
async def delete_all_assessments(current_user: str = Depends(get_current_user)):
    """Delete all assessments from user's history"""
    try:
        # Delete all assessments for the current user
        result = await db.user_assessments.delete_many({"user_id": current_user})
        
        return {
            "success": True, 
            "message": f"Deleted {result.deleted_count} assessments successfully"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete assessments: {str(e)}")

