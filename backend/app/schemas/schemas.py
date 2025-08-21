from pydantic import BaseModel, EmailStr
from typing import List, Optional, Dict, Any
from datetime import datetime

class PredictRequest(BaseModel):
    user_id: Optional[str]
    features: List[float]

class PredictResponse(BaseModel):
    risk_level: str
    scores: List[float]

class ProfileResponse(BaseModel):
    user_id: str
    age: int
    gender: str
    language: str

class PredictionHistoryItem(BaseModel):
    timestamp: str
    risk_level: str
    scores: List[float]

# User Management Schemas
class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class EmailChangeRequest(BaseModel):
    new_email: EmailStr
    password: Optional[str] = None  # Only required for email users

class UsernameChangeRequest(BaseModel):
    new_username: str

class DeleteHistoryRequest(BaseModel):
    assessment_ids: Optional[List[str]] = None  # If None, delete all

class GoogleAuthRequest(BaseModel):
    google_token: str
    temp_user_id: Optional[str] = None

# User Response Models
class UserResponse(BaseModel):
    id: str
    email: str
    full_name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    username: Optional[str] = None
    created_at: Optional[datetime] = None
    last_login: Optional[datetime] = None

class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    username: Optional[str] = None
