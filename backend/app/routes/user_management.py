from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from datetime import datetime
from typing import Optional, Dict, Any
from pydantic import BaseModel, EmailStr
import bcrypt
import httpx
import os
from app.services.db import db
from app.routes.auth import get_current_user, hash_password, verify_password, create_access_token

router = APIRouter()

# Google OAuth Configuration
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

class GoogleAuthRequest(BaseModel):
    google_token: str
    temp_user_id: Optional[str] = None

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str

class EmailChangeRequest(BaseModel):
    new_email: EmailStr
    password: Optional[str] = None

class UsernameChangeRequest(BaseModel):
    new_username: str

class DeleteHistoryRequest(BaseModel):
    assessment_ids: Optional[list] = None  # If None, delete all

async def verify_google_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify Google OAuth token and return user info"""
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://www.googleapis.com/oauth2/v1/userinfo?access_token={token}"
            )
            
            if response.status_code == 200:
                return response.json()
            else:
                return None
    except Exception:
        return None

@router.post("/user/google-auth")
async def google_auth(request: GoogleAuthRequest):
    """Authenticate user with Google OAuth"""
    try:
        # Verify Google token
        google_user_info = await verify_google_token(request.google_token)
        if not google_user_info:
            raise HTTPException(status_code=401, detail="Invalid Google token")
        
        email = google_user_info.get("email")
        name = google_user_info.get("name", "")
        google_id = google_user_info.get("id")
        
        if not email:
            raise HTTPException(status_code=400, detail="Email not provided by Google")
        
        # Check if user exists
        user = await db.users.find_one({"email": email})
        
        if user:
            # Update last login
            await db.users.update_one(
                {"user_id": user["user_id"]},
                {"$set": {"last_login": datetime.utcnow().isoformat()}}
            )
            user_id = user["user_id"]
            username = user.get("username")
        else:
            # Create new user
            import uuid
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            username = name or email.split("@")[0]
            
            new_user = {
                "user_id": user_id,
                "email": email,
                "username": username,
                "google_id": google_id,
                "created_at": datetime.utcnow().isoformat(),
                "last_login": datetime.utcnow().isoformat(),
                "account_status": "active",
                "auth_provider": "google",
                "is_temporary": False
            }
            
            await db.users.insert_one(new_user)
            
            # Link temporary user data if provided
            if request.temp_user_id:
                await link_temp_user_data(request.temp_user_id, user_id)
        
        # Create access token
        from datetime import timedelta
        access_token_expires = timedelta(minutes=30)
        access_token = create_access_token(
            data={"sub": user_id}, expires_delta=access_token_expires
        )
        
        return {
            "success": True,
            "access_token": access_token,
            "token_type": "bearer",
            "user_id": user_id,
            "email": email,
            "username": username,
            "is_temporary": False,
            "message": "Google authentication successful"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Google authentication failed: {str(e)}")

async def link_temp_user_data(temp_user_id: str, user_id: str):
    """Helper function to link temporary user data to registered account"""
    try:
        # Verify temp user exists
        temp_user = await db.temp_users.find_one({"temp_user_id": temp_user_id})
        if not temp_user:
            return  # Silently ignore if temp user doesn't exist
        
        # Transfer assessments from temp user to registered user
        await db.user_assessments.update_many(
            {"user_id": temp_user_id},
            {"$set": {"user_id": user_id, "transferred_from_temp": temp_user_id}}
        )
        
        # Mark temp user as linked
        await db.temp_users.update_one(
            {"temp_user_id": temp_user_id},
            {"$set": {"linked_to_registered": True, "linked_user_id": user_id}}
        )
    except Exception as e:
        print(f"Error linking temp user data: {str(e)}")

@router.get("/user/profile")
async def get_user_profile(current_user: str = Depends(get_current_user)):
    """Get current user profile information"""
    try:
        user = await db.users.find_one({"user_id": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        return {
            "user_id": user["user_id"],
            "email": user["email"],
            "username": user.get("username"),
            "created_at": user.get("created_at"),
            "last_login": user.get("last_login"),
            "auth_provider": user.get("auth_provider", "email"),
            "account_status": user.get("account_status", "active")
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get user profile: {str(e)}")

@router.put("/user/change-password")
async def change_password(request: PasswordChangeRequest, current_user: str = Depends(get_current_user)):
    """Change user password"""
    try:
        user = await db.users.find_one({"user_id": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Check if user has password (Google users might not)
        if not user.get("password_hash"):
            raise HTTPException(status_code=400, detail="Password change not available for Google accounts")
        
        # Verify current password
        if not verify_password(request.current_password, user["password_hash"]):
            raise HTTPException(status_code=401, detail="Current password is incorrect")
        
        # Hash new password
        new_password_hash = hash_password(request.new_password)
        
        # Update password
        await db.users.update_one(
            {"user_id": current_user},
            {"$set": {"password_hash": new_password_hash}}
        )
        
        return {"success": True, "message": "Password changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to change password: {str(e)}")

@router.put("/user/change-email")
async def change_email(request: EmailChangeRequest, current_user: str = Depends(get_current_user)):
    """Change user email"""
    try:
        user = await db.users.find_one({"user_id": current_user})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Verify password for email users (if they have a password)
        if user.get("password_hash") and request.password:
            if not verify_password(request.password, user["password_hash"]):
                raise HTTPException(status_code=401, detail="Password is incorrect")
        elif user.get("password_hash") and not request.password:
            raise HTTPException(status_code=400, detail="Password required for email change")
        
        # Check if new email is already taken
        existing_user = await db.users.find_one({"email": request.new_email})
        if existing_user and existing_user["user_id"] != current_user:
            raise HTTPException(status_code=400, detail="Email already in use")
        
        # Update email
        await db.users.update_one(
            {"user_id": current_user},
            {"$set": {"email": request.new_email}}
        )
        
        return {"success": True, "message": "Email changed successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to change email: {str(e)}")

@router.put("/user/change-username")
async def change_username(request: UsernameChangeRequest, current_user: str = Depends(get_current_user)):
    """Change username"""
    try:
        # Update username
        await db.users.update_one(
            {"user_id": current_user},
            {"$set": {"username": request.new_username}}
        )
        
        return {"success": True, "message": "Username changed successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to change username: {str(e)}")

@router.delete("/user/delete-history")
async def delete_user_history(request: DeleteHistoryRequest, current_user: str = Depends(get_current_user)):
    """Delete user's mental health history"""
    try:
        if request.assessment_ids:
            # Delete specific assessments
            from bson import ObjectId
            object_ids = []
            for assessment_id in request.assessment_ids:
                try:
                    object_ids.append(ObjectId(assessment_id))
                except:
                    continue  # Skip invalid IDs
            
            result = await db.user_assessments.delete_many({
                "_id": {"$in": object_ids},
                "user_id": current_user
            })
            message = f"Deleted {result.deleted_count} assessments"
        else:
            # Delete all assessments
            result = await db.user_assessments.delete_many({"user_id": current_user})
            message = f"Deleted all {result.deleted_count} assessments"
        
        return {"success": True, "message": message}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete history: {str(e)}")

@router.delete("/user/delete-account")
async def delete_user_account(current_user: str = Depends(get_current_user)):
    """Delete user account and all associated data"""
    try:
        # Delete all user assessments
        await db.user_assessments.delete_many({"user_id": current_user})
        
        # Delete user record
        await db.users.delete_one({"user_id": current_user})
        
        # Log deletion for audit
        deletion_log = {
            "user_id": current_user,
            "deleted_at": datetime.utcnow().isoformat(),
            "action": "account_deletion"
        }
        await db.deletion_logs.insert_one(deletion_log)
        
        return {"success": True, "message": "Account deleted successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete account: {str(e)}")

@router.get("/user/assessment-history")
async def get_assessment_history(current_user: str = Depends(get_current_user)):
    """Get user's assessment history with IDs for deletion"""
    try:
        assessments = await db.user_assessments.find({"user_id": current_user}).to_list(length=None)
        
        history = []
        for assessment in assessments:
            history.append({
                "id": str(assessment["_id"]),
                "timestamp": assessment.get("timestamp"),
                "phq9_score": assessment.get("phq9_score"),
                "risk_level": assessment.get("risk_level"),
                "sleep_data": assessment.get("sleep_data", {})
            })
        
        return {"assessments": history}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get assessment history: {str(e)}")

@router.delete("/user/assessment/{assessment_id}")
async def delete_single_assessment(assessment_id: str, current_user: str = Depends(get_current_user)):
    """Delete a single assessment"""
    try:
        from bson import ObjectId
        
        result = await db.user_assessments.delete_one({
            "_id": ObjectId(assessment_id),
            "user_id": current_user
        })
        
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Assessment not found or access denied")
        
        return {"success": True, "message": "Assessment deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete assessment: {str(e)}")

