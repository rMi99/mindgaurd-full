from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from datetime import datetime

# Import our custom modules
from app.routes import auth
from app.routes import dashboard as dashboard_routes
from app.routes import assessment as assessment_routes
from app.services.recommendation_service import RecommendationService
from app.models.enhanced_model import EnhancedHealthModel

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
health_model = EnhancedHealthModel()

# Include routers
app.include_router(auth.router, prefix="/api")
app.include_router(dashboard_routes.router, prefix="/api")
app.include_router(assessment_routes.router, prefix="/api")

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

# Assessment endpoints
@app.post("/api/assessment/submit")
async def submit_assessment(assessment_data: dict):
    """Submit health assessment and get predictions."""
    try:
        # Convert frontend data to model format
        model_data = {
            'age': assessment_data.get('age', 30),
            'gender': 1 if assessment_data.get('gender') == 'male' else 0,
            'sleep_hours': assessment_data.get('sleepHours', 7),
            'exercise_frequency': assessment_data.get('exerciseFrequency', 3),
            'stress_level': assessment_data.get('stressLevel', 5),
            'diet_quality': assessment_data.get('dietQuality', 3),
            'social_connections': assessment_data.get('socialConnections', 3),
            'work_life_balance': assessment_data.get('workLifeBalance', 3),
            'mental_health_history': 1 if assessment_data.get('mentalHealthHistory') else 0,
            'physical_health_history': 0,
            'substance_use': 0,
            'family_history': 0,
            'financial_stress': assessment_data.get('financialStress', 5),
            'relationship_status': 1
        }
        
        # Get prediction from model
        prediction = health_model.predict(model_data)
        
        # Get recommendations based on risk level
        recommendations = recommendation_service.get_personalized_recommendations(
            prediction['risk_level'].lower(),
            model_data
        )
        
        # Create assessment result
        result = {
            "assessment_id": f"assess_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            "risk_level": prediction['risk_level'].lower(),
            "confidence": prediction['confidence'],
            "health_score": int(prediction['confidence'] * 100),
            "recommendations": recommendations['general_recommendations'],
            "brain_heal_activities": recommendations['brain_heal_activities'],
            "weekly_plan": recommendations['weekly_plan'],
            "created_at": datetime.utcnow().isoformat()
        }
        
        logger.info(f"Assessment submitted successfully. Risk level: {result['risk_level']}")
        return result
        
    except Exception as e:
        logger.error(f"Error processing assessment: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/assessment/history")
async def get_assessment_history(limit: int = 10):
    """Get assessment history (placeholder for now)."""
    # This would typically query a database
    return {
        "assessments": [],
        "total": 0,
        "limit": limit
    }

@app.get("/api/assessment/{assessment_id}")
async def get_assessment_by_id(assessment_id: str):
    """Get specific assessment by ID (placeholder for now)."""
    # This would typically query a database
    raise HTTPException(status_code=404, detail="Assessment not found")

# Dashboard endpoints
@app.get("/api/dashboard/data")
async def get_dashboard_data():
    """Get dashboard data (placeholder for now)."""
    return {
        "user": {
            "id": "user_1",
            "email": "user@example.com",
            "full_name": "Example User",
            "created_at": datetime.utcnow().isoformat()
        },
        "recent_assessments": [],
        "health_trends": [],
        "current_risk_level": "normal",
        "overall_health_score": 75,
        "weekly_challenge": {
            "title": "Wellness Week",
            "description": "Focus on building healthy habits",
            "daily_challenges": [
                "Day 1: Take a 20-minute walk",
                "Day 2: Practice 10 minutes of meditation",
                "Day 3: Connect with a friend or family member",
                "Day 4: Try a new healthy recipe",
                "Day 5: Do 30 minutes of exercise",
                "Day 6: Spend time in nature",
                "Day 7: Reflect on your week and plan ahead"
            ],
            "goal": "Build a foundation of daily wellness practices"
        }
    }

@app.get("/api/dashboard/trends")
async def get_health_trends(range: str = "30d"):
    """Get health trends data (placeholder for now)."""
    # This would typically query a database for historical data
    return []

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

