import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import os

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Reuse auth dependency to protect routes, but make it optional for some endpoints
from app.routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])
security = HTTPBearer(auto_error=False)  # Make auth optional

# MongoDB connection - same as assessment route
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "mindguard_db")
_client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)
_db = _client[MONGO_DB_NAME]
_assessments = _db["assessments"]
_users = _db["users"]


async def get_current_user_optional(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[Dict]:
	"""Get current user but don't fail if not authenticated"""
	if not credentials:
		return None
	try:
		return await get_current_user(credentials)
	except HTTPException:
		return None


def _mock_trends(days: int = 30) -> List[Dict[str, Any]]:
	"""Generate mock trend data for the past N days."""
	base = datetime.utcnow()
	series = []
	for i in range(days):
		day = base - timedelta(days=(days - 1 - i))
		series.append({
			"date": day.strftime("%Y-%m-%d"),
			"sleep_hours": 6 + (i % 3) * 0.5,
			"stress_level": max(1, 7 - (i % 5)),
			"exercise_frequency": (i % 4),
			"mood_score": 5 + (i % 4),
			"energy_level": 6 + (i % 3),
			"social_connections": 3 + (i % 3),
			"diet_quality": 3 + (i % 3),
		})
	return series


@router.get("/")
async def get_dashboard_root(
	current_user: Optional[Dict] = Depends(get_current_user_optional),
	user_id: Optional[str] = Query(None, description="User ID for temporary users")
) -> Dict[str, Any]:
	"""High-level dashboard summary for both authenticated and temporary users."""
	
	# Determine effective user ID
	effective_user_id = None
	user_info = {}
	
	if current_user:
		# Authenticated user
		effective_user_id = current_user["id"]
		user_info = {
			"id": current_user["id"],
			"email": current_user["email"],
			"full_name": current_user.get("full_name"),
			"last_login": current_user.get("last_login"),
			"is_authenticated": True
		}
		logger.info("Dashboard accessed by authenticated user: %s", current_user["email"])
	elif user_id:
		# Temporary user
		effective_user_id = user_id
		user_info = {
			"id": user_id,
			"email": None,
			"full_name": "Anonymous User",
			"last_login": None,
			"is_authenticated": False
		}
		logger.info("Dashboard accessed by temporary user: %s", user_id)
	else:
		raise HTTPException(status_code=400, detail="User ID required for unauthenticated access")
	
	# Fetch user's assessment history
	try:
		cursor = _assessments.find({"user_id": effective_user_id}).sort("created_at", -1).limit(10)
		assessments = []
		async for doc in cursor:
			assessment = {
				"id": str(doc.get("_id")),
				"phq9_score": doc.get("phq9", {}).get("total", 0),
				"risk_level": doc.get("assessment_result", {}).get("riskLevel", "unknown"),
				"created_at": doc.get("created_at").isoformat() if doc.get("created_at") else None,
				"sleep_hours": doc.get("sleep", {}).get("sleepHours"),
				"sleep_quality": doc.get("sleep", {}).get("sleepQuality")
			}
			assessments.append(assessment)
	except Exception as e:
		logger.warning("Failed to fetch assessments for user %s: %s", effective_user_id, e)
		assessments = []
	
	# Calculate overall health score and trends
	overall_health_score = 74  # Default
	current_risk_level = "normal"
	
	if assessments:
		latest_assessment = assessments[0]
		phq9_score = latest_assessment.get("phq9_score", 0)
		
		# Calculate health score based on latest PHQ-9
		if phq9_score <= 4:
			overall_health_score = 85
			current_risk_level = "low"
		elif phq9_score <= 9:
			overall_health_score = 75
			current_risk_level = "low"
		elif phq9_score <= 14:
			overall_health_score = 60
			current_risk_level = "moderate"
		elif phq9_score <= 19:
			overall_health_score = 40
			current_risk_level = "high"
		else:
			overall_health_score = 25
			current_risk_level = "high"
	
	return {
		"status": "ok",
		"message": "Dashboard summary",
		"data": {
			"userInfo": user_info,
			"overall_health_score": overall_health_score,
			"current_risk_level": current_risk_level,
			"history": assessments,
			"trends": _mock_trends(7),  # Last 7 days
			"personalizedInsights": {
				"encouragingMessage": "Your mental health journey is important. Keep taking care of yourself.",
				"psychologicalInsights": [
					"Regular self-assessment helps build self-awareness",
					"Small daily habits can lead to significant improvements"
				],
				"personalizedRecommendations": [
					"Continue monitoring your mental health regularly",
					"Practice mindfulness for 10 minutes daily",
					"Maintain a consistent sleep schedule"
				],
				"progressSummary": f"You have completed {len(assessments)} assessment(s)",
				"nextSteps": [
					"Take your next assessment in 1 week",
					"Focus on sleep quality improvement",
					"Consider speaking with a healthcare provider if symptoms persist"
				]
			},
			"widgets": {
				"sleep_avg": 7.1,
				"stress_avg": 4.2,
				"exercise_per_week": 3,
				"assessments_completed": len(assessments)
			},
			"recent_assessments": assessments[:3]  # Last 3 assessments
		}
	}


@router.post("")
async def save_dashboard_data(
	data: Dict[str, Any],
	current_user: Optional[Dict] = Depends(get_current_user_optional)
) -> Dict[str, Any]:
	"""Save dashboard-related data (assessments, etc.)"""
	try:
		logger.info("Dashboard POST data: %s", data)
		
		# Handle different payload formats
		if "assessment" in data:
			# Assessment submission
			assessment_data = data["assessment"]
			user_id = data.get("userId") or (current_user["id"] if current_user else "anonymous")
			
			# Add timestamp and user info
			assessment_record = {
				"user_id": user_id,
				"assessment_data": assessment_data,
				"timestamp": datetime.utcnow(),
				"type": "dashboard_submission"
			}
			
			# Save to database
			result = await _assessments.insert_one(assessment_record)
			
			return {
				"status": "success",
				"message": "Data saved successfully",
				"id": str(result.inserted_id)
			}
		else:
			# Generic data save
			return {
				"status": "success", 
				"message": "Data received successfully",
				"data": data
			}
			
	except Exception as e:
		logger.error("Error saving dashboard data: %s", e)
		raise HTTPException(status_code=500, detail="Failed to save data")


@router.get("/trends")
async def get_dashboard_trends(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
	logger.info("Trends requested by user: %s", current_user["email"]) 
	return {
		"status": "ok",
		"message": "Health trends (mock)",
		"data": _mock_trends(30)
	}


@router.get("/data")
async def get_dashboard_data(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
	logger.info("Dashboard data requested by user: %s", current_user["email"]) 
	return {
		"status": "ok",
		"message": "Dashboard data (mock)",
		"data": {
			"summary": {
				"overall_health_score": 74,
				"risk_level": "normal",
			},
			"trends": _mock_trends(14),
		}
	}

