from fastapi import APIRouter, HTTPException
from datetime import datetime, timedelta
from typing import List, Dict, Any
from pydantic import BaseModel
from app.services.db import get_db

router = APIRouter()

class GlobalStatistics(BaseModel):
    totalUsers: int
    totalRegisteredUsers: int
    totalTemporaryUsers: int
    totalAssessments: int
    averageRiskLevel: str
    riskLevelDistribution: Dict[str, int]
    mentalHealthTrends: Dict[str, Any]
    encouragingMessages: List[str]
    psychologicalInsights: List[str]
    platformInsights: List[str]

class TrendData(BaseModel):
    date: str
    averageScore: float
    assessmentCount: int

@router.get("/global-stats", response_model=GlobalStatistics)
async def get_global_statistics():
    """Get aggregate platform statistics accessible to all users"""
    try:
        db = await get_db()
        
        # Count total registered users
        total_registered_users = await db["users"].count_documents({})
        
        # Count total temporary users (not linked)
        total_temp_users = await db["temp_users"].count_documents({"linked_to_registered": {"$ne": True}})
        
        # Total users
        total_users = total_registered_users + total_temp_users
        
        # Count total assessments
        total_assessments = await db["user_assessments"].count_documents({})
        
        # Calculate risk level distribution
        risk_distribution = {"low": 0, "moderate": 0, "high": 0}
        
        # Get recent assessments for trend analysis (last 30 days)
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        recent_assessments = await db["user_assessments"].find({
            "timestamp": {"$gte": thirty_days_ago}
        }).to_list(length=None)
        
        # Analyze risk levels
        total_recent_assessments = len(recent_assessments)
        if total_recent_assessments > 0:
            for assessment in recent_assessments:
                risk_level = assessment.get("risk_level", "low")
                if risk_level in risk_distribution:
                    risk_distribution[risk_level] += 1
        
        # Calculate average risk level
        if total_recent_assessments > 0:
            risk_scores = {"low": 1, "moderate": 2, "high": 3}
            total_risk_score = sum(risk_scores.get(assessment.get("risk_level", "low"), 1) for assessment in recent_assessments)
            avg_risk_score = total_risk_score / total_recent_assessments
            
            if avg_risk_score <= 1.3:
                average_risk_level = "low"
            elif avg_risk_score <= 2.3:
                average_risk_level = "moderate"
            else:
                average_risk_level = "high"
        else:
            average_risk_level = "low"
        
        # Generate mental health trends
        mental_health_trends = await generate_mental_health_trends(recent_assessments)
        
        # Generate encouraging messages
        encouraging_messages = [
            "You're part of a community that cares about mental health.",
            "Every assessment you take contributes to better understanding of mental wellness.",
            "Your mental health journey matters, and you're not alone.",
            "Taking care of your mental health is a sign of strength and wisdom.",
            "Small steps in mental health awareness can lead to big positive changes."
        ]
        
        # Generate psychological insights
        psychological_insights = [
            "Regular mental health check-ins help identify patterns and improve wellbeing.",
            "Community awareness of mental health reduces stigma and promotes healing.",
            "Understanding your mental health is the first step toward positive change.",
            "Mental health is just as important as physical health in overall wellness.",
            "Seeking support and monitoring mental health are signs of self-care and strength."
        ]
        
        # Generate platform insights based on data
        platform_insights = []
        
        if total_users > 100:
            platform_insights.append(f"Over {total_users} people have used our platform to monitor their mental health.")
        
        if total_assessments > 500:
            platform_insights.append(f"Our community has completed over {total_assessments} mental health assessments.")
        
        if risk_distribution["low"] > risk_distribution["high"]:
            platform_insights.append("Most users in our community report positive mental health indicators.")
        
        platform_insights.extend([
            "Mental health awareness is growing, and seeking help is becoming more normalized.",
            "Regular self-assessment helps people understand their mental health patterns.",
            "Community support and shared experiences contribute to better mental health outcomes."
        ])
        
        return GlobalStatistics(
            totalUsers=total_users,
            totalRegisteredUsers=total_registered_users,
            totalTemporaryUsers=total_temp_users,
            totalAssessments=total_assessments,
            averageRiskLevel=average_risk_level,
            riskLevelDistribution=risk_distribution,
            mentalHealthTrends=mental_health_trends,
            encouragingMessages=encouraging_messages,
            psychologicalInsights=psychological_insights,
            platformInsights=platform_insights
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch global statistics: {str(e)}")

async def generate_mental_health_trends(recent_assessments: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate mental health trends from recent assessments"""
    try:
        if not recent_assessments:
            return {
                "trend_direction": "stable",
                "weekly_data": [],
                "insights": ["Not enough data to determine trends yet."]
            }
        
        # Group assessments by week
        weekly_data = {}
        for assessment in recent_assessments:
            timestamp = assessment.get("timestamp", "")
            if timestamp:
                try:
                    date = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    week_start = date - timedelta(days=date.weekday())
                    week_key = week_start.strftime("%Y-%m-%d")
                    
                    if week_key not in weekly_data:
                        weekly_data[week_key] = {"scores": [], "count": 0}
                    
                    phq9_score = assessment.get("phq9_score", 0)
                    weekly_data[week_key]["scores"].append(phq9_score)
                    weekly_data[week_key]["count"] += 1
                except:
                    continue
        
        # Calculate weekly averages
        trend_data = []
        for week, data in sorted(weekly_data.items()):
            if data["scores"]:
                avg_score = sum(data["scores"]) / len(data["scores"])
                trend_data.append({
                    "date": week,
                    "averageScore": round(avg_score, 1),
                    "assessmentCount": data["count"]
                })
        
        # Determine trend direction
        trend_direction = "stable"
        if len(trend_data) >= 2:
            first_avg = trend_data[0]["averageScore"]
            last_avg = trend_data[-1]["averageScore"]
            
            if last_avg < first_avg - 0.5:
                trend_direction = "improving"
            elif last_avg > first_avg + 0.5:
                trend_direction = "concerning"
        
        # Generate insights
        insights = []
        if trend_direction == "improving":
            insights.append("Community mental health indicators show positive trends.")
        elif trend_direction == "concerning":
            insights.append("Community mental health indicators suggest increased awareness of challenges.")
        else:
            insights.append("Community mental health indicators remain stable.")
        
        if len(recent_assessments) > 50:
            insights.append("High engagement in mental health monitoring shows community commitment to wellness.")
        
        return {
            "trend_direction": trend_direction,
            "weekly_data": trend_data[-8:],  # Last 8 weeks
            "insights": insights
        }
        
    except Exception as e:
        return {
            "trend_direction": "stable",
            "weekly_data": [],
            "insights": ["Unable to calculate trends at this time."]
        }

@router.get("/global-stats/feel-good-content")
async def get_feel_good_content():
    """Get encouraging content and positive mental health messages"""
    try:
        feel_good_content = {
            "daily_affirmations": [
                "You are stronger than you think.",
                "Every day is a new opportunity for growth.",
                "Your mental health journey is valid and important.",
                "You deserve peace, happiness, and good mental health.",
                "Taking care of yourself is not selfish, it's necessary."
            ],
            "mental_health_tips": [
                "Practice deep breathing for 5 minutes daily to reduce stress.",
                "Connect with nature - even a short walk can improve your mood.",
                "Maintain a regular sleep schedule for better mental health.",
                "Practice gratitude by writing down three things you're thankful for.",
                "Stay connected with friends and family for emotional support."
            ],
            "success_stories": [
                "Many people have found that regular mental health check-ins help them stay aware of their wellbeing.",
                "Community support and shared experiences contribute to better mental health outcomes.",
                "Small daily practices like mindfulness can lead to significant improvements in mental health.",
                "Seeking help when needed is a sign of strength, not weakness.",
                "Mental health awareness is growing, making it easier for people to get the support they need."
            ],
            "community_impact": [
                "By monitoring your mental health, you're contributing to a better understanding of community wellness.",
                "Your participation helps reduce mental health stigma and promotes open conversations.",
                "Every assessment helps create a more supportive environment for mental health awareness.",
                "You're part of a community that values mental health and supports each other's wellbeing."
            ]
        }
        
        return feel_good_content
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch feel-good content: {str(e)}")

