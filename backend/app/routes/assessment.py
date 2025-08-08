from fastapi import APIRouter, HTTPException
from datetime import datetime
import torch
import os
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.db import db
from app.models.model import RiskModel
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
MODEL_PATH = os.getenv("MODEL_PATH", "./app/models/checkpoint.pt")
# Convert to absolute path
if not os.path.isabs(MODEL_PATH):
    MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), MODEL_PATH.lstrip('./'))

# Initialize model
device = torch.device("cpu")
model = RiskModel(input_dim=10, hidden_dim=64, output_dim=3)  # Changed from 3 to 10 to match checkpoint
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

router = APIRouter()

# Frontend compatible schemas
class DemographicData(BaseModel):
    age: str
    gender: str
    region: str
    education: str
    employmentStatus: str

class PHQ9Data(BaseModel):
    scores: Dict[str, Optional[int]]  # Frontend sends as {"1": 0, "2": 1, etc.}

class SleepData(BaseModel):
    sleepHours: str
    sleepQuality: str
    exerciseFrequency: str
    stressLevel: str
    socialSupport: str
    screenTime: str

class AssessmentRequest(BaseModel):
    demographics: DemographicData
    phq9: PHQ9Data
    sleep: SleepData
    language: str

class AssessmentResponse(BaseModel):
    phq9Score: int
    riskLevel: str
    scores: List[float]
    riskFactors: List[str]
    recommendations: List[str]

def calculate_phq9_score(phq9_data: Dict[str, Optional[int]]) -> int:
    """Calculate PHQ-9 total score from individual responses"""
    score = 0
    for key, value in phq9_data.items():
        if value is not None:
            score += value
    return score

def convert_sleep_to_features(sleep_data: SleepData, phq9_score: int) -> List[float]:
    """Convert sleep and demographic data to model features"""
    # Convert sleep hours to numeric
    sleep_hours_map = {
        "<4": 3.0,
        "4-5": 4.5,
        "5-6": 5.5,
        "6-7": 6.5,
        "7-8": 7.5,
        "8-9": 8.5,
        ">9": 9.5
    }
    sleep_hours = sleep_hours_map.get(sleep_data.sleepHours, 7.0)
    
    # Convert sleep quality to numeric
    quality_map = {
        "very-poor": 1.0,
        "poor": 2.0,
        "fair": 3.0,
        "good": 4.0,
        "excellent": 5.0
    }
    sleep_quality = quality_map.get(sleep_data.sleepQuality, 3.0)
    
    # Convert stress level to numeric
    stress_map = {
        "very-low": 1.0,
        "low": 2.0,
        "moderate": 3.0,
        "high": 4.0,
        "very-high": 5.0
    }
    stress_level = stress_map.get(sleep_data.stressLevel, 3.0)
    
    # Convert exercise frequency to numeric
    exercise_map = {
        "never": 0.0,
        "rarely": 1.0,
        "sometimes": 2.0,
        "often": 3.0,
        "daily": 4.0
    }
    exercise_freq = exercise_map.get(sleep_data.exerciseFrequency, 2.0)
    
    # Convert social support to numeric
    social_map = {
        "none": 0.0,
        "minimal": 1.0,
        "moderate": 2.0,
        "good": 3.0,
        "excellent": 4.0
    }
    social_support = social_map.get(sleep_data.socialSupport, 2.0)
    
    # Convert screen time to numeric
    screen_map = {
        "<2": 1.0,
        "2-4": 2.0,
        "4-6": 3.0,
        "6-8": 4.0,
        "8-10": 5.0,
        ">10": 6.0
    }
    screen_time = screen_map.get(sleep_data.screenTime, 3.0)
    
    # Create 10 features to match the model
    features = [
        float(phq9_score),      # Feature 0: PHQ-9 score
        sleep_hours,            # Feature 1: Sleep hours
        sleep_quality,          # Feature 2: Sleep quality
        stress_level,           # Feature 3: Stress level
        exercise_freq,          # Feature 4: Exercise frequency
        social_support,         # Feature 5: Social support
        screen_time,            # Feature 6: Screen time
        float(phq9_score) / 27.0,  # Feature 7: Normalized PHQ-9 score
        sleep_hours / 10.0,     # Feature 8: Normalized sleep hours
        stress_level / 5.0      # Feature 9: Normalized stress level
    ]
    
    return features

def analyze_risk_factors(assessment_data: AssessmentRequest, phq9_score: int) -> List[str]:
    """Analyze and return risk factors based on assessment data"""
    risk_factors = []
    
    # PHQ-9 specific risk factors
    if phq9_score >= 15:
        risk_factors.append("Severe depression symptoms detected")
    elif phq9_score >= 10:
        risk_factors.append("Moderate depression symptoms detected")
    
    # Check for suicidal ideation (question 9)
    if assessment_data.phq9.scores.get("9", 0) and assessment_data.phq9.scores["9"] > 0:
        risk_factors.append("Thoughts of self-harm reported - immediate attention needed")
    
    # Sleep-related risk factors
    if assessment_data.sleep.sleepHours in ["<4", "4-5"]:
        risk_factors.append("Insufficient sleep (less than 6 hours)")
    
    if assessment_data.sleep.sleepQuality in ["very-poor", "poor"]:
        risk_factors.append("Poor sleep quality")
    
    # Lifestyle risk factors
    if assessment_data.sleep.exerciseFrequency == "never":
        risk_factors.append("Sedentary lifestyle (no exercise)")
    
    if assessment_data.sleep.stressLevel in ["high", "very-high"]:
        risk_factors.append("High stress levels")
    
    if assessment_data.sleep.socialSupport in ["none", "minimal"]:
        risk_factors.append("Limited social support")
    
    if assessment_data.sleep.screenTime in ["8-10", ">10"]:
        risk_factors.append("Excessive screen time")
    
    return risk_factors

def generate_recommendations(risk_level: str, risk_factors: List[str]) -> List[str]:
    """Generate personalized recommendations based on risk level and factors"""
    recommendations = []
    
    if risk_level == "high":
        recommendations.extend([
            "Consider speaking with a mental health professional immediately",
            "Reach out to a trusted friend, family member, or counselor",
            "Contact a crisis helpline if you're having thoughts of self-harm"
        ])
    elif risk_level == "moderate":
        recommendations.extend([
            "Consider scheduling an appointment with a healthcare provider",
            "Practice stress-reduction techniques like meditation or deep breathing",
            "Maintain regular sleep schedule and exercise routine"
        ])
    else:
        recommendations.extend([
            "Continue maintaining healthy lifestyle habits",
            "Stay connected with friends and family",
            "Monitor your mental health regularly"
        ])
    
    # Add specific recommendations based on risk factors
    if "Insufficient sleep" in " ".join(risk_factors):
        recommendations.append("Aim for 7-9 hours of sleep per night")
    
    if "High stress levels" in " ".join(risk_factors):
        recommendations.append("Practice stress management techniques")
    
    if "Limited social support" in " ".join(risk_factors):
        recommendations.append("Consider joining support groups or social activities")
    
    return recommendations

@router.post("/assessment", response_model=AssessmentResponse)
async def process_assessment(assessment_data: AssessmentRequest):
    """Process assessment data and return risk analysis"""
    try:
        # Calculate PHQ-9 score
        phq9_score = calculate_phq9_score(assessment_data.phq9.scores)
        
        # Convert to model features
        features = convert_sleep_to_features(assessment_data.sleep, phq9_score)
        
        # Get model prediction
        x = torch.tensor([features], dtype=torch.float32)
        with torch.no_grad():
            scores = model(x).squeeze().tolist()
        
        # Determine risk level
        idx = int(torch.argmax(torch.tensor(scores)))
        labels = ["low", "moderate", "high"]
        risk_level = labels[idx]
        
        # Analyze risk factors
        risk_factors = analyze_risk_factors(assessment_data, phq9_score)
        
        # Generate recommendations
        recommendations = generate_recommendations(risk_level, risk_factors)
        
        # Save to database
        await db.assessments.insert_one({
            "timestamp": datetime.utcnow().isoformat(),
            "phq9_score": phq9_score,
            "risk_level": risk_level,
            "scores": scores,
            "demographics": assessment_data.demographics.dict(),
            "sleep_data": assessment_data.sleep.dict(),
            "language": assessment_data.language
        })
        
        return AssessmentResponse(
            phq9Score=phq9_score,
            riskLevel=risk_level,
            scores=scores,
            riskFactors=risk_factors,
            recommendations=recommendations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assessment processing failed: {str(e)}")

