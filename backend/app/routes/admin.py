from fastapi import APIRouter, HTTPException, Query, Depends
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.db import get_db
import hashlib
import secrets

router = APIRouter()

# Admin authentication (simple token-based for demo)
ADMIN_TOKEN = "123"

def verify_admin_token(token: str = Query(...)):
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Invalid admin token")
    return token

class TempUserSummary(BaseModel):
    temp_user_id: str
    total_assessments: int
    last_assessment_date: Optional[str]
    average_phq9_score: float
    current_risk_level: str
    created_date: str
    is_active: bool
    days_since_creation: int

class RegisteredUserSummary(BaseModel):
    user_id: str
    email: str
    username: Optional[str]
    registration_date: str
    last_login: Optional[str]
    consent_for_admin_access: bool
    account_status: str

class SystemStats(BaseModel):
    total_registered_users: int
    total_temp_users: int
    total_assessments: int
    temp_user_assessments: int
    registered_user_assessments: int
    active_temp_users_last_30_days: int
    active_registered_users_last_30_days: int
    temp_users_converted_to_registered: int
    risk_level_distribution_temp_users: Dict[str, int]
    daily_temp_user_activity: Dict[str, int]

class TempUserDetail(BaseModel):
    temp_user_id: str
    assessments: List[Dict[str, Any]]
    statistics: Dict[str, Any]
    last_activity: Optional[str]
    created_date: str
    status: str
    linked_to_registered: bool

class ConsentRequest(BaseModel):
    user_id: str
    admin_reason: str
    requested_access_type: str

@router.get("/admin/stats", response_model=SystemStats)
async def get_system_statistics(token: str = Depends(verify_admin_token)):
    """Get comprehensive system statistics for admin dashboard with focus on temporary users"""
    try:
        db = await get_db()
        
        # Count registered users
        total_registered_users = await db["users"].count_documents({})
        
        # Count temporary users (not linked)
        total_temp_users = await db["temp_users"].count_documents({"linked_to_registered": {"$ne": True}})
        
        # Count assessments
        total_assessments = await db["user_assessments"].count_documents({})
        
        # Get temp user IDs
        temp_users = await db["temp_users"].find({"linked_to_registered": {"$ne": True}}).to_list(length=None)
        temp_user_ids = [user["temp_user_id"] for user in temp_users]
        
        # Count assessments by user type
        temp_user_assessments = await db["user_assessments"].count_documents({"user_id": {"$in": temp_user_ids}})
        registered_user_assessments = total_assessments - temp_user_assessments
        
        # Active users in last 30 days
        thirty_days_ago = (datetime.utcnow() - timedelta(days=30)).isoformat()
        
        # Active temp users
        active_temp_assessments = await db["user_assessments"].find({
            "user_id": {"$in": temp_user_ids},
            "timestamp": {"$gte": thirty_days_ago}
        }).to_list(length=None)
        active_temp_user_ids = list(set([a["user_id"] for a in active_temp_assessments]))
        active_temp_users_last_30_days = len(active_temp_user_ids)
        
        # Active registered users (count only, no detailed access)
        active_registered_assessments = await db["user_assessments"].find({
            "user_id": {"$nin": temp_user_ids},
            "timestamp": {"$gte": thirty_days_ago}
        }).to_list(length=None)
        active_registered_user_ids = list(set([a["user_id"] for a in active_registered_assessments]))
        active_registered_users_last_30_days = len(active_registered_user_ids)
        
        # Count temp users converted to registered
        temp_users_converted = await db["temp_users"].count_documents({"linked_to_registered": True})
        
        # Risk level distribution for temp users only
        risk_distribution_temp = {"low": 0, "moderate": 0, "high": 0}
        temp_assessments = await db["user_assessments"].find({"user_id": {"$in": temp_user_ids}}).to_list(length=None)
        
        for assessment in temp_assessments:
            risk_level = assessment.get("risk_level", "low")
            if risk_level in risk_distribution_temp:
                risk_distribution_temp[risk_level] += 1
        
        # Daily temp user activity (last 7 days)
        daily_activity = {}
        for i in range(7):
            date = (datetime.utcnow() - timedelta(days=i)).strftime("%Y-%m-%d")
            day_start = date + "T00:00:00"
            day_end = date + "T23:59:59"
            
            count = await db["user_assessments"].count_documents({
                "user_id": {"$in": temp_user_ids},
                "timestamp": {"$gte": day_start, "$lte": day_end}
            })
            daily_activity[date] = count
        
        return SystemStats(
            total_registered_users=total_registered_users,
            total_temp_users=total_temp_users,
            total_assessments=total_assessments,
            temp_user_assessments=temp_user_assessments,
            registered_user_assessments=registered_user_assessments,
            active_temp_users_last_30_days=active_temp_users_last_30_days,
            active_registered_users_last_30_days=active_registered_users_last_30_days,
            temp_users_converted_to_registered=temp_users_converted,
            risk_level_distribution_temp_users=risk_distribution_temp,
            daily_temp_user_activity=daily_activity
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch system statistics: {str(e)}")

@router.get("/admin/temp-users", response_model=List[TempUserSummary])
async def get_temp_users(
    token: str = Depends(verify_admin_token),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """Get list of temporary users with their basic information"""
    try:
        # Get temporary users (not linked)
        temp_users = await db["temp_users"].find(
            {"linked_to_registered": {"$ne": True}}
        ).skip(offset).limit(limit).to_list(length=limit)
        
        temp_user_summaries = []
        
        for temp_user in temp_users:
            temp_user_id = temp_user["temp_user_id"]
            created_date = temp_user.get("created_at", "")
            
            # Get assessments for this temp user
            assessments = await db["user_assessments"].find({"user_id": temp_user_id}).to_list(length=None)
            
            total_assessments = len(assessments)
            
            if assessments:
                # Calculate statistics
                phq9_scores = [a.get("phq9_score", 0) for a in assessments]
                average_phq9_score = sum(phq9_scores) / len(phq9_scores)
                
                # Get most recent assessment
                latest_assessment = max(assessments, key=lambda x: x.get("timestamp", ""))
                last_assessment_date = latest_assessment.get("timestamp")
                current_risk_level = latest_assessment.get("risk_level", "low")
                
                # Check if active (assessment in last 7 days)
                seven_days_ago = (datetime.utcnow() - timedelta(days=7)).isoformat()
                is_active = any(a.get("timestamp", "") >= seven_days_ago for a in assessments)
            else:
                average_phq9_score = 0.0
                last_assessment_date = None
                current_risk_level = "unknown"
                is_active = False
            
            # Calculate days since creation
            try:
                created_dt = datetime.fromisoformat(created_date.replace('Z', '+00:00'))
                days_since_creation = (datetime.utcnow() - created_dt.replace(tzinfo=None)).days
            except:
                days_since_creation = 0
            
            temp_user_summaries.append(TempUserSummary(
                temp_user_id=temp_user_id,
                total_assessments=total_assessments,
                last_assessment_date=last_assessment_date,
                average_phq9_score=round(average_phq9_score, 1),
                current_risk_level=current_risk_level,
                created_date=created_date,
                is_active=is_active,
                days_since_creation=days_since_creation
            ))
        
        return temp_user_summaries
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch temporary users: {str(e)}")

@router.get("/admin/temp-users/{temp_user_id}", response_model=TempUserDetail)
async def get_temp_user_detail(temp_user_id: str, token: str = Depends(verify_admin_token)):
    """Get detailed information for a specific temporary user"""
    db = await get_db()
    try:
        # Verify this is a temporary user
        temp_user = await db["temp_users"].find_one({"temp_user_id": temp_user_id})
        if not temp_user:
            raise HTTPException(status_code=404, detail="Temporary user not found")
        
        if temp_user.get("linked_to_registered"):
            raise HTTPException(status_code=403, detail="User has been converted to registered - access denied")
        
        # Get all assessments for this temp user
        assessments = await db["user_assessments"].find({"user_id": temp_user_id}).to_list(length=None)
        
        # Calculate statistics
        statistics = {}
        if assessments:
            phq9_scores = [a.get("phq9_score", 0) for a in assessments]
            statistics = {
                "total_assessments": len(assessments),
                "average_phq9_score": round(sum(phq9_scores) / len(phq9_scores), 1),
                "min_phq9_score": min(phq9_scores),
                "max_phq9_score": max(phq9_scores),
                "first_assessment": min(assessments, key=lambda x: x.get("timestamp", "")).get("timestamp"),
                "last_assessment": max(assessments, key=lambda x: x.get("timestamp", "")).get("timestamp")
            }
            last_activity = statistics["last_assessment"]
        else:
            statistics = {
                "total_assessments": 0,
                "average_phq9_score": 0.0,
                "min_phq9_score": 0,
                "max_phq9_score": 0,
                "first_assessment": None,
                "last_assessment": None
            }
            last_activity = None
        
        # Remove sensitive data from assessments (keep only necessary fields)
        cleaned_assessments = []
        for assessment in assessments:
            cleaned_assessments.append({
                "timestamp": assessment.get("timestamp"),
                "phq9_score": assessment.get("phq9_score"),
                "risk_level": assessment.get("risk_level"),
                "sleep_data": assessment.get("sleep_data", {})
            })
        
        return TempUserDetail(
            temp_user_id=temp_user_id,
            assessments=cleaned_assessments,
            statistics=statistics,
            last_activity=last_activity,
            created_date=temp_user.get("created_at", ""),
            status="active" if not temp_user.get("linked_to_registered") else "converted",
            linked_to_registered=temp_user.get("linked_to_registered", False)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch temporary user details: {str(e)}")

@router.get("/admin/registered-users", response_model=List[RegisteredUserSummary])
async def get_registered_users_summary(
    token: str = Depends(verify_admin_token),
    limit: int = Query(50, le=100),
    offset: int = Query(0, ge=0)
):
    """Get summary of registered users (no private data access without consent)"""
    try:
        # Get registered users with only basic information
        users = await db["users"].find({}).skip(offset).limit(limit).to_list(length=limit)
        
        user_summaries = []
        
        for user in users:
            # Only include non-sensitive information
            user_summaries.append(RegisteredUserSummary(
                user_id=user["user_id"],
                email=user.get("email", ""),
                username=user.get("username"),
                registration_date=user.get("created_at", ""),
                last_login=user.get("last_login"),  # This would need to be tracked separately
                consent_for_admin_access=user.get("admin_access_consent", False),
                account_status=user.get("account_status", "active")
            ))
        
        return user_summaries
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch registered users: {str(e)}")

@router.post("/admin/request-user-consent")
async def request_user_data_access(
    request: ConsentRequest,
    token: str = Depends(verify_admin_token)
):
    """Request consent from a registered user to access their data"""
    try:
        # Verify user exists
        user = await db["users"].find_one({"user_id": request.user_id})
        if not user:
            raise HTTPException(status_code=404, detail="User not found")
        
        # Create consent request record
        consent_request = {
            "user_id": request.user_id,
            "admin_reason": request.admin_reason,
            "requested_access_type": request.requested_access_type,
            "request_date": datetime.utcnow().isoformat(),
            "status": "pending",
            "admin_token_hash": hashlib.sha256(token.encode()).hexdigest()
        }
        
        await db["admin_consent_requests"].insert_one(consent_request)
        
        # In a real implementation, this would trigger an email/notification to the user
        
        return {
            "success": True,
            "message": "Consent request created. User will be notified to approve or deny access.",
            "request_id": str(consent_request.get("_id"))
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create consent request: {str(e)}")

@router.delete("/admin/temp-users/{temp_user_id}")
async def delete_temp_user(temp_user_id: str, token: str = Depends(verify_admin_token)):
    """Delete a temporary user and their data"""
    db = await get_db()
    try:
        # Verify this is a temporary user
        temp_user = await db["temp_users"].find_one({"temp_user_id": temp_user_id})
        if not temp_user:
            raise HTTPException(status_code=404, detail="Temporary user not found")
        
        if temp_user.get("linked_to_registered"):
            raise HTTPException(status_code=403, detail="Cannot delete converted user data")
        
        # Delete user assessments
        await db["user_assessments"].delete_many({"user_id": temp_user_id})
        
        # Delete temp user record
        await db["temp_users"].delete_one({"temp_user_id": temp_user_id})
        
        # Log admin action
        admin_log = {
            "action": "delete_temp_user",
            "temp_user_id": temp_user_id,
            "admin_token_hash": hashlib.sha256(token.encode()).hexdigest(),
            "timestamp": datetime.utcnow().isoformat()
        }
        await db["admin_logs"].insert_one(admin_log)
        
        return {"success": True, "message": "Temporary user and associated data deleted successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete temporary user: {str(e)}")

@router.get("/admin/flagged-temp-users")
async def get_flagged_temp_users(token: str = Depends(verify_admin_token)):
    """Get temporary users that may need attention (high risk, inactive, etc.)"""
    db = await get_db()
    try:
        # Get temp users with high risk assessments
        high_risk_assessments = await db["user_assessments"].find({"risk_level": "high"}).to_list(length=None)
        high_risk_temp_user_ids = []
        
        # Filter for temp users only
        temp_users = await db["temp_users"].find({"linked_to_registered": {"$ne": True}}).to_list(length=None)
        temp_user_ids = [user["temp_user_id"] for user in temp_users]
        
        for assessment in high_risk_assessments:
            if assessment["user_id"] in temp_user_ids:
                high_risk_temp_user_ids.append(assessment["user_id"])
        
        # Get unique high-risk temp users
        flagged_users = []
        for temp_user_id in set(high_risk_temp_user_ids):
            # Get recent assessments
            recent_assessments = await db["user_assessments"].find({
                "user_id": temp_user_id,
                "timestamp": {"$gte": (datetime.utcnow() - timedelta(days=7)).isoformat()}
            }).to_list(length=None)
            
            if recent_assessments:
                high_risk_count = sum(1 for a in recent_assessments if a.get("risk_level") == "high")
                if high_risk_count > 0:
                    flagged_users.append({
                        "temp_user_id": temp_user_id,
                        "flag_reason": "high_risk_assessments",
                        "high_risk_count": high_risk_count,
                        "total_recent_assessments": len(recent_assessments),
                        "last_assessment": max(recent_assessments, key=lambda x: x.get("timestamp", "")).get("timestamp")
                    })
        
        return {"flagged_temp_users": flagged_users}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch flagged temporary users: {str(e)}")
    try:
        # Get all assessments
        all_assessments = await db["user_assessments"].find({}).to_list(length=None)
        
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

