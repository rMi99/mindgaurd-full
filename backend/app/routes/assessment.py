import logging
from datetime import datetime
from typing import Dict, Any, List
import os

from fastapi import APIRouter, Depends, HTTPException
from fastapi import Request
from motor.motor_asyncio import AsyncIOMotorClient
from bson import ObjectId

# Reuse auth dependency to protect routes
from app.routes.auth import get_current_user
from app.services.recommendation_service import RecommendationService

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assessment", tags=["assessment"])

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "mindguard_db")
_client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)
_db = _client[MONGO_DB_NAME]
_assessments = _db["assessments"]

# Initialize recommendation service
recommendation_service = RecommendationService()


def _sum_phq9(scores: Dict[str, Any]) -> int:
	"""Calculate PHQ-9 total score from individual question scores"""
	try:
		total = 0
		for key, value in scores.items():
			if value is not None and isinstance(value, (int, float)):
				# Ensure score is within valid range
				score = max(0, min(3, int(value)))
				total += score
		return total
	except Exception as e:
		logger.warning(f"Error calculating PHQ-9 score: {e}")
		return 0


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
	out = {**doc}
	# Convert ObjectId and datetime fields
	if isinstance(out.get("_id"), ObjectId):
		out["_id"] = str(out["_id"])
	if isinstance(out.get("created_at"), datetime):
		out["created_at"] = out["created_at"].isoformat()
	return out


def _determine_risk_level(phq9_score: int) -> str:
	"""Determine risk level based on PHQ-9 score"""
	if phq9_score >= 15:
		return "high"
	elif phq9_score >= 10:
		return "moderate"
	else:
		return "low"


def _generate_assessment_result(phq9_score: int, assessment_data: Dict[str, Any]) -> Dict[str, Any]:
	"""Generate comprehensive assessment result for frontend"""
	risk_level = _determine_risk_level(phq9_score)
	
	# Get recommendations from service
	recommendations = recommendation_service.get_personalized_recommendations(
		risk_level,
		assessment_data
	)
	
	# Generate risk factors
	risk_factors = []
	if phq9_score >= 15:
		risk_factors.append("Severe depression symptoms detected")
	elif phq9_score >= 10:
		risk_factors.append("Moderate depression symptoms detected")
	
	# Check for suicidal ideation (question 9)
	phq9_scores = assessment_data.get("phq9", {}).get("scores", {})
	if phq9_scores.get("9", 0) and phq9_scores["9"] > 0:
		risk_factors.append("Thoughts of self-harm reported - immediate attention needed")
	
	# Sleep-related risk factors
	sleep_data = assessment_data.get("sleep", {})
	if sleep_data.get("sleepHours") in ["<4", "4-5"]:
		risk_factors.append("Insufficient sleep (less than 6 hours)")
	
	if sleep_data.get("sleepQuality") in ["very-poor", "poor"]:
		risk_factors.append("Poor sleep quality")
	
	# Generate protective factors
	protective_factors = []
	if phq9_score <= 4:
		protective_factors.append("Minimal depression symptoms")
	if sleep_data.get("sleepHours") in ["7-8", "8-9"]:
		protective_factors.append("Good sleep duration")
	if sleep_data.get("exerciseFrequency") in ["often", "daily"]:
		protective_factors.append("Regular exercise routine")
	
	# Cultural considerations
	cultural_considerations = [
		"Mental health experiences vary across cultures",
		"Consider cultural factors in your mental health journey",
		"Seek culturally appropriate support when needed"
	]
	
	# Emergency resources for high risk
	emergency_resources = []
	if risk_level == "high":
		emergency_resources = [
			{
				"name": "National Suicide Prevention Lifeline",
				"phone": "988",
				"description": "24/7 crisis support and suicide prevention",
				"available24h": True
			},
			{
				"name": "Crisis Text Line",
				"phone": "Text HOME to 741741",
				"description": "Free 24/7 crisis counseling via text",
				"available24h": True
			}
		]
	
	return {
		"assessment_id": f"assess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
		"riskLevel": risk_level,
		"phq9Score": phq9_score,
		"confidenceScore": 0.85,  # Mock confidence score
		"riskFactors": risk_factors,
		"protectiveFactors": protective_factors,
		"recommendations": recommendations.get("general_recommendations", []),
		"culturalConsiderations": cultural_considerations,
		"emergencyResources": emergency_resources,
		"brainHealActivities": recommendations.get("brain_heal_activities", []),
		"weeklyPlan": recommendations.get("weekly_plan", {}),
		"created_at": datetime.utcnow().isoformat()
	}


@router.get("/")
async def assessment_root(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
	user_id = current_user["id"]
	last = await _assessments.find_one({"user_id": user_id}, sort=[("created_at", -1)])
	logger.info("Assessment home accessed by user: %s", current_user["email"])
	return {
		"status": "ok",
		"message": "Assessment API root",
		"data": {
			"available_forms": ["baseline", "weekly_checkin"],
			"last_submitted": (last.get("created_at").isoformat() if last and isinstance(last.get("created_at"), datetime) else None),
		}
	}


@router.post("/submit")
async def assessment_submit(request: Request, payload: Dict[str, Any], current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
	"""Persist assessment to MongoDB and return comprehensive results."""
	try:
		# Log minimal request header info for diagnostics
		auth_hdr = request.headers.get("authorization")
		ct_hdr = request.headers.get("content-type")
		logger.info("assessment_submit headers: Authorization=%s, Content-Type=%s", bool(auth_hdr), ct_hdr)
		logger.info("assessment_submit called for user=%s", current_user.get("email"))
		
		# Enhanced validation with detailed error messages
		validation_errors = []
		
		# Check for required top-level fields
		if not payload.get("demographics"):
			validation_errors.append("Missing demographics data")
		if not payload.get("phq9"):
			validation_errors.append("Missing PHQ-9 data")
		if not payload.get("sleep"):
			validation_errors.append("Missing sleep data")
		
		# Validate demographics structure
		demographics = payload.get("demographics", {})
		if isinstance(demographics, dict):
			required_demo_fields = ["age", "gender"]
			for field in required_demo_fields:
				if not demographics.get(field):
					validation_errors.append(f"Missing required demographic field: {field}")
		else:
			validation_errors.append("Demographics must be an object")
		
		# Validate PHQ-9 structure - handle both direct scores and nested scores format
		phq9_data = payload.get("phq9", {})
		if isinstance(phq9_data, dict):
			# Try to get scores from nested structure first, fallback to direct structure
			phq9_scores = phq9_data.get("scores", phq9_data if phq9_data else {})
			
			if not isinstance(phq9_scores, dict):
				validation_errors.append("PHQ-9 scores must be an object with question numbers as keys")
			else:
				# Check if we have at least some PHQ-9 scores
				valid_scores = {k: v for k, v in phq9_scores.items() if v is not None}
				if not valid_scores:
					validation_errors.append("At least one PHQ-9 question must be answered")
				else:
					# Validate that scores are in valid range (0-3)
					for key, value in valid_scores.items():
						if not isinstance(value, (int, float)) or value < 0 or value > 3:
							validation_errors.append(f"PHQ-9 question {key} has invalid score: {value}. Scores must be 0-3.")
		else:
			validation_errors.append("PHQ-9 data must be an object")
		
		# Validate sleep structure
		sleep_data = payload.get("sleep", {})
		if isinstance(sleep_data, dict):
			required_sleep_fields = ["sleepHours", "sleepQuality", "exerciseFrequency", "stressLevel"]
			for field in required_sleep_fields:
				if not sleep_data.get(field):
					validation_errors.append(f"Missing required sleep field: {field}")
		else:
			validation_errors.append("Sleep data must be an object")
		
		# If there are validation errors, return them
		if validation_errors:
			raise HTTPException(
				status_code=400, 
				detail=f"Validation failed: {'; '.join(validation_errors)}"
			)
		
		user_id = current_user["id"]
		# Handle both nested and direct PHQ-9 score formats
		phq9_scores = phq9_data.get("scores", phq9_data if phq9_data else {})
		phq9_total = _sum_phq9(phq9_scores)
		
		# Generate comprehensive assessment result
		assessment_result = _generate_assessment_result(phq9_total, payload)
		
		# Prepare database document
		doc = {
			"user_id": user_id,
			"demographics": demographics,
			"phq9": {"scores": phq9_scores, "total": phq9_total},
			"sleep": sleep_data,
			"language": payload.get("language", "en"),
			"created_at": datetime.utcnow(),
			"assessment_result": assessment_result
		}
		
		inserted_id = None
		try:
			result = await _assessments.insert_one(doc)
			inserted_id = str(result.inserted_id) if result and result.inserted_id else None
		except Exception as db_err:
			logger.warning("Mongo insert failed, returning generated ID. Error: %s", db_err)
			import uuid
			inserted_id = f"assess_{uuid.uuid4().hex[:12]}"
		
		# Update assessment result with the actual ID
		assessment_result["assessment_id"] = inserted_id
		
		logger.info("Assessment submitted by user: %s", current_user["email"]) 
		return assessment_result
		
	except HTTPException:
		raise
	except Exception as e:
		logger.exception("Error saving assessment for user %s: %s", current_user.get("email"), e)
		raise HTTPException(status_code=500, detail="Internal server error while saving assessment")


@router.get("/history")
async def assessment_history(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
	"""Return assessment history for current user."""
	try:
		user_id = current_user["id"]
		cursor = _assessments.find({"user_id": user_id}).sort("created_at", -1)
		history: List[Dict[str, Any]] = []
		async for doc in cursor:
			history.append(_serialize(doc))
		logger.info("Assessment history requested by user: %s", current_user["email"]) 
		return {
			"status": "ok",
			"message": "Assessment history retrieved",
			"data": history
		}
	except Exception as e:
		logger.exception("Error retrieving history for user %s: %s", current_user.get("email"), e)
		raise HTTPException(status_code=500, detail="Internal server error while retrieving history")

