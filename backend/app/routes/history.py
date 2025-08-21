from fastapi import APIRouter, HTTPException, Query
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from app.services.db import get_db

router = APIRouter()

class DetailedAssessmentHistory(BaseModel):
    id: str
    date: str
    phq9Score: int
    riskLevel: str
    sleepData: Optional[Dict[str, Any]] = None
    demographicData: Optional[Dict[str, Any]] = None
    responses: Optional[Dict[str, Any]] = None
    aiInsights: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

class HistoryFilter(BaseModel):
    startDate: Optional[str] = None
    endDate: Optional[str] = None
    riskLevels: Optional[List[str]] = None
    limit: Optional[int] = 50

class HistoryStats(BaseModel):
    totalAssessments: int
    averagePhq9Score: float
    riskLevelDistribution: Dict[str, int]
    assessmentFrequency: Dict[str, int]
    improvementTrend: str
    lastAssessmentDate: str

class ExportData(BaseModel):
    userId: str
    exportDate: str
    assessments: List[DetailedAssessmentHistory]
    statistics: HistoryStats
    metadata: Dict[str, Any]

@router.get("/history/{user_id}", response_model=List[DetailedAssessmentHistory])
async def get_detailed_history(
    user_id: str,
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    risk_levels: Optional[str] = Query(None),
    limit: int = Query(50, le=200)
):
    """Get detailed assessment history for a user with filtering options"""
    db = await get_db()
    try:
        # Build query filter
        query_filter = {"user_id": user_id}
        
        # Add date range filter
        if start_date or end_date:
            date_filter = {}
            if start_date:
                date_filter["$gte"] = start_date
            if end_date:
                date_filter["$lte"] = end_date
            query_filter["timestamp"] = date_filter
        
        # Add risk level filter
        if risk_levels:
            risk_level_list = risk_levels.split(",")
            query_filter["risk_level"] = {"$in": risk_level_list}
        
        # Fetch assessments from database
        cursor = db["user_assessments"].find(query_filter).sort("timestamp", -1).limit(limit)
        assessments = await cursor.to_list(length=limit)
        
        # Convert to detailed format
        detailed_history = []
        for assessment in assessments:
            detailed_item = DetailedAssessmentHistory(
                id=str(assessment.get("_id", "")),
                date=assessment.get("timestamp", datetime.utcnow().isoformat()),
                phq9Score=assessment.get("phq9_score", 0),
                riskLevel=assessment.get("risk_level", "low"),
                sleepData=assessment.get("sleep_data", {}),
                demographicData=assessment.get("demographic_data", {}),
                responses=assessment.get("responses", {}),
                aiInsights=assessment.get("ai_insights", {}),
                notes=assessment.get("notes", "")
            )
            detailed_history.append(detailed_item)
        
        return detailed_history
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch detailed history: {str(e)}")

@router.get("/history/{user_id}/stats", response_model=HistoryStats)
async def get_history_statistics(user_id: str):
    db = await get_db()
    """Get comprehensive statistics for user's assessment history"""
    try:
        # Get all assessments for the user
        cursor = db["user_assessments"].find({"user_id": user_id}).sort("timestamp", 1)
        assessments = await cursor.to_list(length=None)
        
        if not assessments:
            raise HTTPException(status_code=404, detail="No assessment history found")
        
        # Calculate statistics
        total_assessments = len(assessments)
        phq9_scores = [a.get("phq9_score", 0) for a in assessments]
        average_phq9 = sum(phq9_scores) / len(phq9_scores) if phq9_scores else 0
        
        # Risk level distribution
        risk_levels = [a.get("risk_level", "low") for a in assessments]
        risk_distribution = {}
        for level in ["low", "moderate", "high"]:
            risk_distribution[level] = risk_levels.count(level)
        
        # Assessment frequency by month
        frequency = {}
        for assessment in assessments:
            date_str = assessment.get("timestamp", "")
            if date_str:
                try:
                    date_obj = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    month_key = date_obj.strftime("%Y-%m")
                    frequency[month_key] = frequency.get(month_key, 0) + 1
                except:
                    pass
        
        # Improvement trend
        if len(phq9_scores) >= 2:
            recent_scores = phq9_scores[-3:]  # Last 3 scores
            early_scores = phq9_scores[:3]    # First 3 scores
            recent_avg = sum(recent_scores) / len(recent_scores)
            early_avg = sum(early_scores) / len(early_scores)
            
            if recent_avg < early_avg - 2:
                improvement_trend = "improving"
            elif recent_avg > early_avg + 2:
                improvement_trend = "declining"
            else:
                improvement_trend = "stable"
        else:
            improvement_trend = "insufficient_data"
        
        # Last assessment date
        last_assessment_date = assessments[-1].get("timestamp", "") if assessments else ""
        
        return HistoryStats(
            totalAssessments=total_assessments,
            averagePhq9Score=round(average_phq9, 2),
            riskLevelDistribution=risk_distribution,
            assessmentFrequency=frequency,
            improvementTrend=improvement_trend,
            lastAssessmentDate=last_assessment_date
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to calculate statistics: {str(e)}")

@router.post("/history/{user_id}/export")
async def export_user_data(user_id: str):
    db = await get_db()
    """Export complete user data for download"""
    try:
        # Get detailed history
        detailed_history = await get_detailed_history(user_id, limit=1000)
        
        # Get statistics
        stats = await get_history_statistics(user_id)
        
        # Create export data
        export_data = ExportData(
            userId=user_id,
            exportDate=datetime.utcnow().isoformat(),
            assessments=detailed_history,
            statistics=stats,
            metadata={
                "export_version": "1.0",
                "platform": "MindGuard",
                "privacy_notice": "This data export contains your personal mental health assessment history. Please handle with care and in accordance with your local privacy regulations.",
                "data_retention": "You may request deletion of this data at any time by contacting support.",
                "format": "JSON"
            }
        )
        
        return export_data
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to export user data: {str(e)}")

@router.delete("/history/{user_id}")
async def delete_user_history(user_id: str, confirm: bool = Query(False)):
    db = await get_db()
    """Delete all assessment history for a user (requires confirmation)"""
    try:
        if not confirm:
            raise HTTPException(
                status_code=400, 
                detail="Deletion requires confirmation. Add ?confirm=true to the request."
            )
        
        # Delete all assessments for the user
        result = await db["user_assessments"].delete_many({"user_id": user_id})
        
        return {
            "success": True,
            "message": f"Deleted {result.deleted_count} assessment records for user {user_id}",
            "deleted_count": result.deleted_count
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete user history: {str(e)}")

@router.post("/history/{user_id}/note")
async def add_assessment_note(user_id: str, assessment_id: str, note: str):
    db = await get_db()
    """Add a personal note to a specific assessment"""
    try:
        from bson import ObjectId
        
        # Update the assessment with the note
        result = await db["user_assessments"].update_one(
            {"_id": ObjectId(assessment_id), "user_id": user_id},
            {"$set": {"notes": note, "note_updated": datetime.utcnow().isoformat()}}
        )
        
        if result.matched_count == 0:
            raise HTTPException(status_code=404, detail="Assessment not found")
        
        return {"success": True, "message": "Note added successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to add note: {str(e)}")

@router.get("/history/{user_id}/insights")
async def get_ai_insights(user_id: str):
    db = await get_db()
    """Get AI-generated insights based on user's complete history"""
    try:
        # Get user's assessment history
        cursor = db["user_assessments"].find({"user_id": user_id}).sort("timestamp", 1)
        assessments = await cursor.to_list(length=None)
        
        if len(assessments) < 2:
            return {
                "insights": ["Take more assessments to receive personalized insights"],
                "recommendations": ["Complete regular assessments to track your progress"],
                "patterns": {}
            }
        
        # Analyze patterns
        phq9_scores = [a.get("phq9_score", 0) for a in assessments]
        sleep_data = [a.get("sleep_data", {}) for a in assessments]
        
        insights = []
        recommendations = []
        patterns = {}
        
        # PHQ-9 trend analysis
        if len(phq9_scores) >= 3:
            recent_trend = sum(phq9_scores[-3:]) / 3 - sum(phq9_scores[:3]) / 3
            if recent_trend < -2:
                insights.append("Your mental health scores show significant improvement over time")
                recommendations.append("Continue with your current self-care practices")
            elif recent_trend > 2:
                insights.append("Your mental health scores indicate increasing concerns")
                recommendations.append("Consider reaching out to a mental health professional")
            
            patterns["phq9_trend"] = recent_trend
        
        # Sleep pattern analysis
        sleep_qualities = [s.get("sleepQuality", "") for s in sleep_data if s.get("sleepQuality")]
        if sleep_qualities:
            poor_sleep_count = sleep_qualities.count("poor")
            if poor_sleep_count > len(sleep_qualities) * 0.5:
                insights.append("Poor sleep quality appears to be a recurring pattern")
                recommendations.append("Focus on improving sleep hygiene and establishing a consistent bedtime routine")
            
            patterns["sleep_quality_issues"] = poor_sleep_count / len(sleep_qualities)
        
        # Stress level analysis
        stress_levels = [s.get("stressLevel", "") for s in sleep_data if s.get("stressLevel")]
        if stress_levels:
            high_stress_count = stress_levels.count("high")
            if high_stress_count > len(stress_levels) * 0.4:
                insights.append("High stress levels are frequently reported in your assessments")
                recommendations.append("Consider stress management techniques such as meditation or exercise")
            
            patterns["high_stress_frequency"] = high_stress_count / len(stress_levels)
        
        return {
            "insights": insights,
            "recommendations": recommendations,
            "patterns": patterns,
            "analysis_date": datetime.utcnow().isoformat(),
            "assessments_analyzed": len(assessments)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate insights: {str(e)}")

