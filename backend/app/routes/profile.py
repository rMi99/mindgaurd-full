from fastapi import APIRouter, HTTPException, Depends
from app.services.db import get_db
from app.schemas.schemas import ProfileResponse
from app.routes.auth import get_current_user
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ProfileUpdateRequest(BaseModel):
    full_name: Optional[str] = None
    username: Optional[str] = None
    age: Optional[int] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date_of_birth: Optional[str] = None
    gender: Optional[str] = None
    location: Optional[str] = None
    occupation: Optional[str] = None
    bio: Optional[str] = None
    preferences: Optional[Dict[str, Any]] = None

@router.get("/profile")
async def get_current_user_profile(
    current_user: dict = Depends(get_current_user)
):
    """Get the authenticated user's profile"""
    try:
        # Get user profile from database
        db = await get_db()
        users_collection = db["users"]
        user = await users_collection.find_one({"email": current_user["email"]})
        
        if not user:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        # Remove sensitive data
        profile_data = {
            "id": str(user.get("_id", "")),
            "email": user.get("email", ""),
            "full_name": user.get("full_name", ""),
            "username": user.get("username", ""),
            "age": user.get("age"),
            "phone": user.get("phone", ""),
            "date_of_birth": user.get("date_of_birth", ""),
            "gender": user.get("gender", ""),
            "location": user.get("location", ""),
            "occupation": user.get("occupation", ""),
            "bio": user.get("bio", ""),
            "preferences": user.get("preferences", {}),
            "created_at": user.get("created_at", ""),
            "last_login": user.get("last_login"),
            "is_authenticated": True,
            "account_status": user.get("account_status", "active"),
            "is_verified": user.get("is_verified", False),
            "profile_picture": user.get("profile_image", ""),
        }
        
        return {
            "status": "success",
            "data": profile_data
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error fetching user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch profile")

@router.put("/profile")
async def update_current_user_profile(
    profile_update: ProfileUpdateRequest,
    current_user: dict = Depends(get_current_user)
):
    """Update the authenticated user's profile"""
    try:
        db = await get_db()
        users_collection = db["users"]
        
        # Build update data
        update_data = {}
        for field, value in profile_update.dict(exclude_unset=True).items():
            if value is not None:
                update_data[field] = value
        
        if not update_data:
            raise HTTPException(status_code=400, detail="No valid update data provided")
        
        # Update user in database
        result = await users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": update_data}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Get updated user data
        updated_user = await users_collection.find_one({"email": current_user["email"]})
        
        profile_data = {
            "id": str(updated_user.get("_id", "")),
            "email": updated_user.get("email", ""),
            "full_name": updated_user.get("full_name", ""),
            "username": updated_user.get("username", ""),
            "age": updated_user.get("age"),
            "phone": updated_user.get("phone", ""),
            "date_of_birth": updated_user.get("date_of_birth", ""),
            "gender": updated_user.get("gender", ""),
            "location": updated_user.get("location", ""),
            "occupation": updated_user.get("occupation", ""),
            "bio": updated_user.get("bio", ""),
            "preferences": updated_user.get("preferences", {}),
            "created_at": updated_user.get("created_at", ""),
            "last_login": updated_user.get("last_login"),
            "is_authenticated": True,
            "account_status": updated_user.get("account_status", "active"),
            "is_verified": updated_user.get("is_verified", False),
            "profile_picture": updated_user.get("profile_image", ""),
        }
        
        return {
            "status": "success",
            "data": profile_data,
            "message": "Profile updated successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user profile: {e}")
        raise HTTPException(status_code=500, detail="Failed to update profile")

@router.get("/profile/{user_id}", response_model=ProfileResponse)
async def get_profile(user_id: str):
    db = await get_db()
    users_collection = db["users"]
    user = await users_collection.find_one({"user_id": user_id}, {'_id':0})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user
