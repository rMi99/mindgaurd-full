from fastapi import APIRouter, Depends, HTTPException, status
from typing import Optional, Dict, Any
from pydantic import BaseModel
from ..services.db import get_database
from .auth import get_current_user

router = APIRouter()

class NotificationSettings(BaseModel):
    email_notifications: bool = True
    push_notifications: bool = True
    assessment_reminders: bool = True
    weekly_reports: bool = True
    emergency_alerts: bool = True
    marketing_emails: bool = False

class PrivacySettings(BaseModel):
    data_sharing: bool = False
    analytics_tracking: bool = True
    profile_visibility: str = "private"  # "public", "private", "friends"
    show_activity: bool = False

class AppSettings(BaseModel):
    theme: str = "system"  # "light", "dark", "system"
    language: str = "en"
    timezone: str = "UTC"
    date_format: str = "DD/MM/YYYY"
    currency: str = "USD"

class UserSettings(BaseModel):
    notifications: NotificationSettings
    privacy: PrivacySettings
    app: AppSettings

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

@router.get("/settings")
async def get_user_settings(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Get user settings"""
    try:
        # Get settings from database
        settings_collection = db.user_settings
        user_settings = settings_collection.find_one({"user_id": current_user["id"]})
        
        if not user_settings:
            # Return default settings if none exist
            default_settings = {
                "notifications": {
                    "email_notifications": True,
                    "push_notifications": True,
                    "assessment_reminders": True,
                    "weekly_reports": True,
                    "emergency_alerts": True,
                    "marketing_emails": False,
                },
                "privacy": {
                    "data_sharing": False,
                    "analytics_tracking": True,
                    "profile_visibility": "private",
                    "show_activity": False,
                },
                "app": {
                    "theme": "system",
                    "language": "en",
                    "timezone": "UTC",
                    "date_format": "DD/MM/YYYY",
                    "currency": "USD",
                },
            }
            return default_settings
        
        # Remove MongoDB _id field
        if "_id" in user_settings:
            del user_settings["_id"]
        if "user_id" in user_settings:
            del user_settings["user_id"]
            
        return user_settings
        
    except Exception as e:
        print(f"Error fetching settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch settings"
        )

@router.put("/settings")
async def update_user_settings(
    settings: UserSettings,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Update user settings"""
    try:
        settings_collection = db.user_settings
        
        # Convert to dict
        settings_dict = settings.dict()
        settings_dict["user_id"] = current_user["id"]
        
        # Update or insert settings
        result = settings_collection.update_one(
            {"user_id": current_user["id"]},
            {"$set": settings_dict},
            upsert=True
        )
        
        # Return updated settings (without user_id and _id)
        updated_settings = settings_dict.copy()
        if "user_id" in updated_settings:
            del updated_settings["user_id"]
            
        return updated_settings
        
    except Exception as e:
        print(f"Error updating settings: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update settings"
        )

@router.post("/auth/change-password")
async def change_password(
    password_request: PasswordChangeRequest,
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Change user password"""
    try:
        from passlib.context import CryptContext
        
        pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        users_collection = db.users
        
        # Get current user from database
        user = users_collection.find_one({"email": current_user["email"]})
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        # Verify current password
        if not pwd_context.verify(password_request.current_password, user["hashed_password"]):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid current password"
            )
        
        # Validate new password
        if len(password_request.new_password) < 8:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="New password must be at least 8 characters long"
            )
        
        # Hash new password
        new_hashed_password = pwd_context.hash(password_request.new_password)
        
        # Update password in database
        users_collection.update_one(
            {"email": current_user["email"]},
            {"$set": {"hashed_password": new_hashed_password}}
        )
        
        return {"message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error changing password: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to change password"
        )

@router.delete("/auth/delete-account")
async def delete_account(
    current_user: dict = Depends(get_current_user),
    db = Depends(get_database)
):
    """Delete user account and all associated data"""
    try:
        user_id = current_user["id"]
        email = current_user["email"]
        
        # Delete from all collections
        db.users.delete_one({"email": email})
        db.assessments.delete_many({"user_id": user_id})
        db.user_settings.delete_many({"user_id": user_id})
        db.user_profiles.delete_many({"user_id": user_id})
        
        return {"message": "Account deleted successfully"}
        
    except Exception as e:
        print(f"Error deleting account: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete account"
        )
