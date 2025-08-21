from fastapi import APIRouter, HTTPException, Depends
from datetime import datetime
from typing import Dict, Any, List
from pydantic import BaseModel
import numpy as np
from app.routes.auth import get_current_user

router = APIRouter()

class PredictionRequest(BaseModel):
    phq9_score: int
    sleep_hours: float
    stress_level: int
    exercise_frequency: int
    social_support: int
    screen_time: int

class PredictionResponse(BaseModel):
    risk_level: str
    confidence: float
    key_factors: Dict[str, Dict[str, str]]
    recommendations: List[str]
    next_assessment_date: str

def simple_risk_prediction(request: PredictionRequest) -> Dict[str, Any]:
    """Simple rule-based risk prediction without ML dependencies"""
    
    # Calculate risk score based on factors
    risk_score = 0
    
    # PHQ-9 contribution (0-27 scale)
    if request.phq9_score >= 20:
        risk_score += 40
    elif request.phq9_score >= 15:
        risk_score += 30
    elif request.phq9_score >= 10:
        risk_score += 20
    elif request.phq9_score >= 5:
        risk_score += 10
    
    # Sleep contribution
    if request.sleep_hours < 6:
        risk_score += 15
    elif request.sleep_hours < 7:
        risk_score += 10
    elif request.sleep_hours > 9:
        risk_score += 5
    
    # Stress contribution
    if request.stress_level >= 4:
        risk_score += 15
    elif request.stress_level >= 3:
        risk_score += 10
    
    # Exercise contribution (inverse)
    if request.exercise_frequency <= 1:
        risk_score += 10
    elif request.exercise_frequency <= 2:
        risk_score += 5
    
    # Social support contribution (inverse)
    if request.social_support <= 1:
        risk_score += 15
    elif request.social_support <= 2:
        risk_score += 10
    
    # Screen time contribution
    if request.screen_time >= 5:
        risk_score += 10
    elif request.screen_time >= 4:
        risk_score += 5
    
    # Determine risk level
    if risk_score >= 60:
        risk_level = "high"
        confidence = 0.85
    elif risk_score >= 40:
        risk_level = "moderate"
        confidence = 0.75
    else:
        risk_level = "low"
        confidence = 0.80
    
    # Generate key factors
    key_factors = {}
    
    if request.phq9_score >= 10:
        key_factors["depression_symptoms"] = {
            "value": f"PHQ-9 Score: {request.phq9_score}",
            "impact": "High" if request.phq9_score >= 15 else "Moderate"
        }
    
    if request.sleep_hours < 7:
        key_factors["sleep_patterns"] = {
            "value": f"{request.sleep_hours} hours per night",
            "impact": "High" if request.sleep_hours < 6 else "Moderate"
        }
    
    if request.stress_level >= 3:
        key_factors["stress_levels"] = {
            "value": f"Level {request.stress_level}/5",
            "impact": "High" if request.stress_level >= 4 else "Moderate"
        }
    
    if request.exercise_frequency <= 2:
        key_factors["physical_activity"] = {
            "value": f"{request.exercise_frequency} times per week",
            "impact": "High" if request.exercise_frequency <= 1 else "Moderate"
        }
    
    if request.social_support <= 2:
        key_factors["social_support"] = {
            "value": f"Level {request.social_support}/5",
            "impact": "High" if request.social_support <= 1 else "Moderate"
        }
    
    # Generate recommendations
    recommendations = []
    
    if risk_level == "high":
        recommendations.extend([
            "Consider speaking with a mental health professional immediately",
            "Reach out to trusted friends, family, or counselors",
            "Contact a crisis helpline if experiencing thoughts of self-harm"
        ])
    elif risk_level == "moderate":
        recommendations.extend([
            "Consider scheduling an appointment with a healthcare provider",
            "Practice stress-reduction techniques like meditation",
            "Maintain regular sleep schedule and exercise routine"
        ])
    else:
        recommendations.extend([
            "Continue maintaining healthy lifestyle habits",
            "Stay connected with friends and family",
            "Monitor your mental health regularly"
        ])
    
    # Add specific recommendations based on factors
    if request.sleep_hours < 7:
        recommendations.append("Aim for 7-9 hours of sleep per night")
    
    if request.stress_level >= 3:
        recommendations.append("Practice stress management techniques like deep breathing")
    
    if request.exercise_frequency <= 2:
        recommendations.append("Incorporate regular physical activity into your routine")
    
    if request.social_support <= 2:
        recommendations.append("Consider joining support groups or social activities")
    
    # Calculate next assessment date
    if risk_level == "high":
        days_until_next = 3
    elif risk_level == "moderate":
        days_until_next = 7
    else:
        days_until_next = 14
    
    next_date = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    next_date = next_date.replace(day=next_date.day + days_until_next)
    
    return {
        "risk_level": risk_level,
        "confidence": confidence,
        "key_factors": key_factors,
        "recommendations": recommendations,
        "next_assessment_date": next_date.isoformat()
    }

@router.post("/predictions", response_model=PredictionResponse)
async def get_prediction(
    request: PredictionRequest,
    current_user: Dict = Depends(get_current_user)
):
    """Get ML-based prediction for user's mental health risk"""
    try:
        result = simple_risk_prediction(request)
        
        # Log prediction for analytics
        prediction_log = {
            "user_id": current_user["id"],
            "timestamp": datetime.utcnow().isoformat(),
            "prediction_data": request.dict(),
            "prediction_result": result
        }
        
        # Here you would save to database if needed
        # await db.predictions.insert_one(prediction_log)
        
        return PredictionResponse(**result)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.get("/predictions/history")
async def get_prediction_history(current_user: Dict = Depends(get_current_user)):
    """Get user's prediction history"""
    try:
        # Mock history - in real implementation, fetch from database
        history = [
            {
                "id": "pred_001",
                "timestamp": datetime.utcnow().isoformat(),
                "risk_level": "moderate",
                "confidence": 0.75,
                "key_factors": ["sleep_patterns", "stress_levels"]
            }
        ]
        
        return {
            "status": "ok",
            "data": history
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get prediction history: {str(e)}")
