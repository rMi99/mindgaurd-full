from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.db import db
import hashlib
import secrets

router = APIRouter()

# Admin authentication (simple token-based for demo)
ADMIN_TOKEN = "mindguard_admin_2024"

def verify_admin_token(token: str = Query(...)):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return token

class UserSummary(BaseModel):
    user_id: str
    total_assessments: int
    last_assessment_date: Optional[str]
    average_phq9_score: float
    current_risk_level: str
    registration_date: Optional[str]
    is_active: bool

class SystemStats(BaseModel):
    total_users: int
    total_assessments: int
    active_users_last_30_days: int
    average_assessments_per_user: float
    risk_level_distribution: Dict[str, int]
    daily_assessment_counts: Dict[str, int]

class UserDetail(BaseModel):
    user_id: str
    assessments: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    last_activity: Optional[str]
    account_status: str

class PasswordResetRequest(BaseModel):
    user_id: str
    new_password_hash: str
    reset_reason: str

@router.get("/admin/stats", response_model=SystemStats)
async def get_system_statistics(token: str = Depends(verify_admin_token)):
    """Get comprehensive system statistics for admin dashboard"""
    try:
        # Get all assessments
        all_assessments = await db.user_assessments.find({}).to_list(length=None)
        
        # Calculate basic stats
        total_assessments = len(all_assessments)
        unique_users = set(assessment.get("user_id") for assessment in all_assessments)
        total_users = len(unique_users)
        
        # Active users in last 30 days
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        recent_assessments = [a for a in all_assessments if a.get("timestamp", "") > thirty_days_ago]
        active_users_last_30_days = len(set(a.get("user_id") for a in recent_assessments))
        
        # Average assessments per user
        avg_assessments_per_user = total_assessments / total_users if total_users > 0 else 0
        
        # Risk level distribution
        risk_levels = [a.get("risk_level", "unknown") for a in all_assessments]
        risk_distribution = {}
        for level in ["low", "moderate", "high", "unknown"]:
            risk_distribution[level] = risk_levels.count(level)
        
        # Daily assessment counts (last 30 days)
        daily_counts = {}
        for assessment in recent_assessments:
            date_str = assessment.get("timestamp", "")
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    day_key = date_obj.strftime("%Y-%m-%d")
                    daily_counts[day_key] = daily_counts.get(day_key, 0) + 1
                except:
                    pass
        
        return SystemStats(
            total_users=total_users,
            total_assessments=total_assessments,
            active_users_last_30_days=active_users_last_30_days,
            average_assessments_per_user=round(avg_assessments_per_user, 2),
            risk_level_distribution=risk_distribution,
            daily_assessment_counts=daily_counts
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch system statistics: {str(e)}")

@router.get("/admin/users", response_model=List[UserSummary])
async def get_users_summary(
    token: str = Depends(verify_admin_token),
    limit: int = Query(100, le=1000),
    offset: int = Query(0, ge=0),
    risk_level: Optional[str] = Query(None),
    active_only: bool = Query(False)
):
    """Get summary of all users for admin management"""
    try:
        # Get all assessments
        all_assessments = await db.user_assessments.find({}).to_list(length=None)
        
        # Group by user
        user_data = {}
        for assessment in all_assessments:
            user_id = assessment.get("user_id")
            if not user_id:
                continue
                
            if user_id not in user_data:
                user_data[user_id] = []
            user_data[user_id].append(assessment)
        
        # Create user summaries
        user_summaries = []
        for user_id, assessments in user_data.items():
            if not assessments:
                continue
                
            # Sort assessments by date
            sorted_assessments = sorted(assessments, key=lambda x: x.get("timestamp", ""), reverse=True)
            latest_assessment = sorted_assessments[0]
            
            # Calculate statistics
            phq9_scores = [a.get("phq9_score", 0) for a in assessments]
            avg_phq9 = sum(phq9_scores) / len(phq9_scores) if phq9_scores else 0
            
            # Check if user is active (assessment in last 30 days)
            thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
            is_active = latest_assessment.get("timestamp", "") > thirty_days_ago
            
            # Apply filters
            if risk_level and latest_assessment.get("risk_level") != risk_level:
                continue
            if active_only and not is_active:
                continue
            
            user_summary = UserSummary(
                user_id=user_id,
                total_assessments=len(assessments),
                last_assessment_date=latest_assessment.get("timestamp"),
                average_phq9_score=round(avg_phq9, 2),
                current_risk_level=latest_assessment.get("risk_level", "unknown"),
                registration_date=sorted_assessments[-1].get("timestamp"),  # First assessment
                is_active=is_active
            )
            user_summaries.append(user_summary)
        
        # Sort by last assessment date (most recent first)
        user_summaries.sort(key=lambda x: x.last_assessment_date or "", reverse=True)
        
        # Apply pagination
        return user_summaries[offset:offset + limit]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch users summary: {str(e)}")

@router.get("/admin/users/{user_id}", response_model=UserDetail)
async def get_user_detail(user_id: str, token: str = Depends(verify_admin_token)):
    """Get detailed information about a specific user"""
    try:
        # Get user's assessments
        cursor = db.user_assessments.find({"user_id": user_id}).sort("timestamp", -1)
        assessments = await cursor.to_list(length=None)
        
        if not assessments:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Calculate user statistics
        phq9_scores = [a.get("phq9_score", 0) for a in assessments]
        risk_levels = [a.get("risk_level", "unknown") for a in assessments]
        
        statistics = {
            "total_assessments": len(assessments),
            "average_phq9_score": round(sum(phq9_scores) / len(phq9_scores), 2) if phq9_scores else 0,
            "risk_level_distribution": {level: risk_levels.count(level) for level in set(risk_levels)},
            "first_assessment": assessments[-1].get("timestamp") if assessments else None,
            "assessment_frequency": len(assessments) / max(1, (datetime.utcnow() - datetime.fromisoformat(assessments[-1].get("timestamp", datetime.utcnow().isoformat()).replace('Z', '+00:00'))).days) if assessments else 0
        }
        
        # Determine account status
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        last_activity = assessments[0].get("timestamp") if assessments else None
        account_status = "active" if last_activity and last_activity > thirty_days_ago else "inactive"
        
        return UserDetail(
            user_id=user_id,
            assessments=[{
                "id": str(a.get("_id", "")),
                "date": a.get("timestamp"),
                "phq9_score": a.get("phq9_score"),
                "risk_level": a.get("risk_level"),
                "sleep_data": a.get("sleep_data", {}),
                "notes": a.get("notes", "")
            } for a in assessments],
            statistics=statistics,
            last_activity=last_activity,
            account_status=account_status
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch user detail: {str(e)}")

@router.post("/admin/users/{user_id}/reset-password")
async def reset_user_password(
    user_id: str,
    request: PasswordResetRequest,
    token: str = Depends(verify_admin_token)
):
    """Reset password for a user (placeholder - would integrate with actual auth system)"""
    try:
        # In a real system, this would integrate with your authentication system
        # For now, we'll just log the password reset request
        
        reset_record = {
            "user_id": user_id,
            "admin_token": hashlib.sha256(token.encode()).hexdigest()[:16],  # Partial hash for audit
            "reset_timestamp": datetime.utcnow().isoformat(),
            "reset_reason": request.reset_reason,
            "new_password_hash": request.new_password_hash
        }
        
        # Store password reset record
        await db.password_resets.insert_one(reset_record)
        
        return {
            "success": True,
            "message": f"Password reset initiated for user {user_id}",
            "reset_id": str(reset_record.get("_id", ""))
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset password: {str(e)}")

@router.delete("/admin/users/{user_id}")
async def delete_user_account(
    user_id: str,
    reason: str = Query(...),
    token: str = Depends(verify_admin_token)
):
    """Delete a user account and all associated data"""
    try:
        # Delete all user assessments
        assessment_result = await db.user_assessments.delete_many({"user_id": user_id})
        
        # Delete any password reset records
        reset_result = await db.password_resets.delete_many({"user_id": user_id})
        
        # Log the deletion
        deletion_record = {
            "deleted_user_id": user_id,
            "admin_token": hashlib.sha256(token.encode()).hexdigest()[:16],
            "deletion_timestamp": datetime.utcnow().isoformat(),
            "deletion_reason": reason,
            "assessments_deleted": assessment_result.deleted_count,
            "reset_records_deleted": reset_result.deleted_count
        }
        
        await db.user_deletions.insert_one(deletion_record)
        
        return {
            "success": True,
            "message": f"User {user_id} and all associated data deleted",
            "assessments_deleted": assessment_result.deleted_count,
            "reset_records_deleted": reset_result.deleted_count
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user account: {str(e)}")

@router.get("/admin/analytics/risk-trends")
async def get_risk_trends(
    token: str = Depends(verify_admin_token),
    days: int = Query(30, ge=1, le=365)
):
    """Get risk level trends over time for analytics"""
    try:
        # Get assessments from the specified time period
        start_date = (datetime.utcnow() - timedelta(days=days)).isoformat()
        cursor = db.user_assessments.find({"timestamp": {"$gte": start_date}}).sort("timestamp", 1)
        assessments = await cursor.to_list(length=None)
        
        # Group by date and risk level
        daily_risk_counts = {}
        for assessment in assessments:
            date_str = assessment.get("timestamp", "")
            risk_level = assessment.get("risk_level", "unknown")
            
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    day_key = date_obj.strftime("%Y-%m-%d")
                    
                    if day_key not in daily_risk_counts:
                        daily_risk_counts[day_key] = {"low": 0, "moderate": 0, "high": 0, "unknown": 0}
                    
                    daily_risk_counts[day_key][risk_level] = daily_risk_counts[day_key].get(risk_level, 0) + 1
                except:
                    pass
        
        return {
            "period_days": days,
            "daily_risk_counts": daily_risk_counts,
            "total_assessments": len(assessments),
            "analysis_date": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch risk trends: {str(e)}")

@router.post("/admin/users/{user_id}/flag")
async def flag_user_for_review(
    user_id: str,
    reason: str = Query(...),
    priority: str = Query("medium"),
    token: str = Depends(verify_admin_token)
):
    """Flag a user account for manual review"""
    try:
        flag_record = {
            "user_id": user_id,
            "flag_reason": reason,
            "priority": priority,
            "flagged_by_admin": hashlib.sha256(token.encode()).hexdigest()[:16],
            "flag_timestamp": datetime.utcnow().isoformat(),
            "status": "pending_review",
            "resolved": False
        }
        
        await db.user_flags.insert_one(flag_record)
        
        return {
            "success": True,
            "message": f"User {user_id} flagged for review",
            "flag_id": str(flag_record.get("_id", "")),
            "priority": priority
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to flag user: {str(e)}")

@router.get("/admin/flags")
async def get_flagged_users(
    token: str = Depends(verify_admin_token),
    status: str = Query("pending_review"),
    priority: Optional[str] = Query(None)
):
    """Get list of flagged users for review"""
    try:
        query_filter = {"status": status}
        if priority:
            query_filter["priority"] = priority
        
        cursor = db.user_flags.find(query_filter).sort("flag_timestamp", -1)
        flags = await cursor.to_list(length=100)
        
        return {
            "flagged_users": [{
                "flag_id": str(flag.get("_id", "")),
                "user_id": flag.get("user_id"),
                "reason": flag.get("flag_reason"),
                "priority": flag.get("priority"),
                "flag_date": flag.get("flag_timestamp"),
                "status": flag.get("status")
            } for flag in flags],
            "total_flags": len(flags)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch flagged users: {str(e)}")

