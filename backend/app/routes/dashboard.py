import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List

from fastapi import APIRouter, Depends

# Reuse auth dependency to protect routes
from app.routes.auth import get_current_user

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


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
async def get_dashboard_root(current_user: Dict = Depends(get_current_user)) -> Dict[str, Any]:
	"""High-level dashboard summary (mock)."""
	logger.info("Dashboard accessed by user: %s", current_user["email"]) 
	return {
		"status": "ok",
		"message": "Dashboard summary",
		"data": {
			"user": {
				"id": current_user["id"],
				"email": current_user["email"],
				"full_name": current_user.get("full_name"),
				"last_login": current_user.get("last_login"),
			},
			"overall_health_score": 74,
			"current_risk_level": "normal",
			"widgets": {
				"sleep_avg": 7.1,
				"stress_avg": 4.2,
				"exercise_per_week": 3,
			},
			"recent_assessments": [
				{"id": "assess_001", "risk_level": "low", "created_at": datetime.utcnow().isoformat()},
				{"id": "assess_002", "risk_level": "normal", "created_at": datetime.utcnow().isoformat()},
			],
		}
	}


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

