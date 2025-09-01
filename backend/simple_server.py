"""
Simple FastAPI server for testing facial dashboard without complex dependencies.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging
from datetime import datetime

# Import simplified routes
from app.routes.facial_dashboard_simple import router as facial_dashboard_router

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="MindGuard Facial Dashboard API",
    description="Simplified API for testing facial dashboard",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include facial dashboard router
app.include_router(facial_dashboard_router, prefix="/api")

# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint for monitoring."""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "MindGuard Facial Dashboard API",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "Welcome to MindGuard Facial Dashboard API",
        "version": "1.0.0",
        "description": "Simplified API for testing facial dashboard",
        "docs": "/docs",
        "endpoints": {
            "facial_dashboard": "/api/facial-dashboard"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
