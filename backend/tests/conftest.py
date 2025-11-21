"""
Pytest configuration and shared fixtures for MindGuard backend tests
"""
import pytest
import asyncio
from typing import AsyncGenerator, Generator
from unittest.mock import Mock, AsyncMock, MagicMock
from datetime import datetime
import os
import sys
from pathlib import Path

# Ensure backend package is importable when running pytest from repo root
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment variables before any imports
os.environ["TESTING"] = "true"
os.environ["MONGO_DETAILS"] = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")
os.environ["DATABASE_NAME"] = os.getenv("TEST_DATABASE_NAME", "mindguard_test_db")


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_db():
    """Mock MongoDB database instance"""
    db = MagicMock()
    db.users = MagicMock()
    db.assessments = MagicMock()
    db.predictions = MagicMock()
    db.facial_analyses = MagicMock()
    return db


@pytest.fixture
def mock_mongo_client(mock_db):
    """Mock MongoDB client"""
    client = MagicMock()
    client.__getitem__ = MagicMock(return_value=mock_db)
    client.admin.command = AsyncMock(return_value={"ok": 1})
    client.close = Mock()
    return client


@pytest.fixture
def sample_user_data():
    """Sample user data for testing"""
    return {
        "user_id": "test_user_123",
        "email": "test@example.com",
        "username": "testuser",
        "age": 30,
        "gender": "other",
        "created_at": datetime.now().isoformat()
    }


@pytest.fixture
def sample_assessment_data():
    """Sample assessment data for testing"""
    return {
        "user_id": "test_user_123",
        "phq9_score": 12,
        "risk_level": "moderate",
        "timestamp": datetime.now().isoformat(),
        "responses": {
            "q1": 2,
            "q2": 1,
            "q3": 3
        }
    }


@pytest.fixture
def sample_facial_analysis_data():
    """Sample facial analysis data for testing"""
    return {
        "user_id": "test_user_123",
        "emotions": {
            "happy": 0.3,
            "sad": 0.2,
            "angry": 0.1,
            "neutral": 0.4
        },
        "timestamp": datetime.now().isoformat(),
        "confidence": 0.85
    }


@pytest.fixture
def mock_fastapi_request():
    """Mock FastAPI request object"""
    request = MagicMock()
    request.headers = {}
    request.cookies = {}
    request.client = MagicMock()
    request.client.host = "127.0.0.1"
    return request


@pytest.fixture
def mock_jwt_token():
    """Mock JWT token for authentication tests"""
    return "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidGVzdF91c2VyXzEyMyIsImV4cCI6OTk5OTk5OTk5OX0.test_signature"


@pytest.fixture
def mock_ml_model():
    """Mock ML model for testing"""
    model = MagicMock()
    model.predict = Mock(return_value={"risk_level": "moderate", "confidence": 0.75})
    model.get_accuracy = Mock(return_value=0.85)
    model.get_model_info = Mock(return_value={"type": "test_model", "version": "1.0"})
    return model


@pytest.fixture
def mock_external_api():
    """Mock external API calls"""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json = Mock(return_value={"success": True})
    mock_response.raise_for_status = Mock()
    return mock_response


@pytest.fixture(autouse=True)
def reset_environment():
    """Reset environment before each test"""
    yield
    # Cleanup after test if needed
    pass


# Pytest markers for test organization
pytest_plugins = []

