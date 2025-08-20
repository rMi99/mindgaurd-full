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

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/assessment", tags=["assessment"])

# MongoDB connection
MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
MONGO_DB_NAME = os.getenv("MONGO_DB", "mindguard_db")
_client: AsyncIOMotorClient = AsyncIOMotorClient(MONGO_URI)
_db = _client[MONGO_DB_NAME]
_assessments = _db["assessments"]


def _sum_phq9(scores: Dict[str, Any]) -> int:
	try:
		return int(sum(int(v) for v in scores.values() if v is not None))
	except Exception:
		return 0


def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
	out = {**doc}
	# Convert ObjectId and datetime fields
	if isinstance(out.get("_id"), ObjectId):
		out["_id"] = str(out["_id"])
	if isinstance(out.get("created_at"), datetime):
		out["created_at"] = out["created_at"].isoformat()
	return out


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
	"""Persist assessment to MongoDB."""
	try:
		# Log minimal request header info for diagnostics
		auth_hdr = request.headers.get("authorization")
		ct_hdr = request.headers.get("content-type")
		logger.info("assessment_submit headers: Authorization=%s, Content-Type=%s", bool(auth_hdr), ct_hdr)
		logger.info("assessment_submit called for user=%s", current_user.get("email"))
		# Basic validation
		if not payload.get("demographics") or not payload.get("phq9") or not payload.get("sleep"):
			raise HTTPException(status_code=400, detail="Missing required fields: demographics, phq9, sleep")
		
		user_id = current_user["id"]
		phq9_scores = payload.get("phq9", {}).get("scores", {})
		phq9_total = _sum_phq9(phq9_scores)
		doc = {
			"user_id": user_id,
			"demographics": payload.get("demographics"),
			"phq9": {"scores": phq9_scores, "total": phq9_total},
			"sleep": payload.get("sleep"),
			"language": payload.get("language", "en"),
			"created_at": datetime.utcnow(),
		}
		result = await _assessments.insert_one(doc)
		if not result.inserted_id:
			raise HTTPException(status_code=500, detail="Failed to save assessment")
		
		logger.info("Assessment submitted by user: %s", current_user["email"]) 
		return {
			"status": "ok",
			"message": "Assessment saved successfully",
			"data": {"assessment_id": str(result.inserted_id)}
		}
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

