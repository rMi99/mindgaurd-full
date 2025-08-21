from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# Import our custom modules
from app.routes import auth
from app.routes import dashboard as dashboard_routes
from app.routes import assessment as assessment_routes
from app.routes import health as health_routes
from app.routes import research as research_routes
from app.routes import global_stats as global_stats_routes
from app.routes import history as history_routes
from app.routes import admin as admin_routes
from app.routes import predictions as predictions_routes
# from app.routes import predict as predict_routes  # Optional, guarded below
from app.routes import profile as profile_routes
from app.routes import user_management as user_mgmt_routes
from app.routes import settings as settings_routes
from app.services.recommendation_service import RecommendationService
# from app.models.enhanced_model import EnhancedHealthModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# Silence passlib bcrypt backend version warning (harmless but noisy)
logging.getLogger("passlib.handlers.bcrypt").setLevel(logging.ERROR)

# Initialize FastAPI app
app = FastAPI(
	title="MindGuard API",
	description="AI-Powered Health Risk Prediction System",
	version="1.0.0",
	docs_url="/docs",
	redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
	CORSMiddleware,
	allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Add your frontend URLs
	allow_origin_regex=".*",  # Allow all origins for testing
	allow_credentials=True,
	allow_methods=["*"],
	allow_headers=["*"],
)

# Initialize services
recommendation_service = RecommendationService()
# health_model = EnhancedHealthModel()

# Include routers
# Auth and assessment are under /api prefix for frontend API client compatibility
app.include_router(auth.router, prefix="/api")
app.include_router(assessment_routes.router, prefix="/api")
app.include_router(health_routes.router, prefix="/api")

# Public (no extra /api) routes for Next.js API proxies
app.include_router(dashboard_routes.router)          # /dashboard
app.include_router(research_routes.router)           # /research
app.include_router(global_stats_routes.router)       # /global-stats
app.include_router(history_routes.router)            # /history
app.include_router(admin_routes.router)              # /admin
app.include_router(predictions_routes.router)        # /predictions
# Optional torch-based predictions route
try:
	from app.routes import predict as predict_routes
	app.include_router(predict_routes.router)         # /predict
except Exception as e:
	logger.warning("Torch predict routes disabled: %s", e)
app.include_router(profile_routes.router, prefix="/user")  # /user/profile
app.include_router(user_mgmt_routes.router)          # /user/*
app.include_router(settings_routes.router, prefix="/api")  # /api/settings

# Health check endpoint
@app.get("/health")
async def health_check():
	"""Health check endpoint for monitoring."""
	return {
		"status": "healthy",
		"timestamp": datetime.utcnow().isoformat(),
		"service": "MindGuard API",
		"version": "1.0.0"
	}

# Root endpoint
@app.get("/")
async def root():
	"""Root endpoint with API information."""
	return {
		"message": "Welcome to MindGuard API",
		"version": "1.0.0",
		"description": "AI-Powered Health Risk Prediction System",
		"docs": "/docs",
		"health": "/health"
	}

# Recommendation endpoints
@app.get("/api/recommendations")
async def get_recommendations(risk_level: str):
	"""Get recommendations for a specific risk level."""
	try:
		recommendations = recommendation_service.get_personalized_recommendations(
			risk_level,
			{}  # Empty user data for general recommendations
		)
		return recommendations
	except Exception as e:
		logger.error(f"Error getting recommendations: {e}")
		raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/recommendations/brain-heal")
async def get_brain_heal_activities(risk_level: str = None):
	"""Get Brain Heal activities."""
	try:
		if risk_level:
			activities = recommendation_service._get_brain_heal_activities(risk_level)
		else:
			activities = recommendation_service.get_random_activity()
		return {"activities": activities}
	except Exception as e:
		logger.error(f"Error getting brain heal activities: {e}")
		raise HTTPException(status_code=500, detail=str(e))

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
	return JSONResponse(
		status_code=404,
		content={"detail": "Resource not found"}
	)

@app.exception_handler(500)
async def internal_error_handler(request, exc):
	return JSONResponse(
		status_code=500,
		content={"detail": "Internal server error"}
	)

if __name__ == "__main__":
	import uvicorn
	uvicorn.run(app, host="0.0.0.0", port=8000)

