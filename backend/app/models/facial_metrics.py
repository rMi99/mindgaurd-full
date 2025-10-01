"""
Models for comprehensive facial analysis metrics including mood, sleepiness, 
fatigue, stress indicators, and PHQ-9 estimation.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class EyeMetrics(BaseModel):
    """Eye-related measurements for sleepiness detection."""
    left_ear: float = Field(description="Left Eye Aspect Ratio")
    right_ear: float = Field(description="Right Eye Aspect Ratio") 
    avg_ear: float = Field(description="Average Eye Aspect Ratio")
    blink_rate: float = Field(description="Blinks per minute")
    blink_duration: float = Field(description="Average blink duration in ms")

class HeadPoseMetrics(BaseModel):
    """Head pose measurements for fatigue detection."""
    pitch: float = Field(description="Head pitch angle in degrees")
    yaw: float = Field(description="Head yaw angle in degrees")
    roll: float = Field(description="Head roll angle in degrees")
    stability: float = Field(description="Head stability score (0-1)")

class MicroExpressionMetrics(BaseModel):
    """Micro-expression analysis for stress detection."""
    muscle_tension: float = Field(description="Facial muscle tension score (0-1)")
    asymmetry: float = Field(description="Facial asymmetry score (0-1)")
    micro_movement_frequency: float = Field(description="Micro-movements per second")
    expression_variability: float = Field(description="Expression variability score (0-1)")

class EmotionDistribution(BaseModel):
    """Detailed emotion probability distribution."""
    happy: float = Field(ge=0, le=1)
    sad: float = Field(ge=0, le=1)
    angry: float = Field(ge=0, le=1)
    fear: float = Field(ge=0, le=1)
    surprise: float = Field(ge=0, le=1)
    disgust: float = Field(ge=0, le=1)
    neutral: float = Field(ge=0, le=1)

class SleepinessLevel(BaseModel):
    """Sleepiness assessment result."""
    level: str = Field(description="Alert, Slightly tired, Very tired")
    confidence: float = Field(ge=0, le=1)
    contributing_factors: List[str] = Field(description="Factors contributing to sleepiness")

class FatigueIndicators(BaseModel):
    """Fatigue detection result."""
    yawning_detected: bool
    head_droop_detected: bool
    overall_fatigue: bool
    confidence: float = Field(ge=0, le=1)

class StressLevel(BaseModel):
    """Stress level assessment."""
    level: str = Field(description="Low, Medium, High")
    confidence: float = Field(ge=0, le=1)
    indicators: List[str] = Field(description="Detected stress indicators")

class PHQ9Estimation(BaseModel):
    """PHQ-9 score estimation from facial analysis."""
    estimated_score: int = Field(ge=0, le=27, description="Estimated PHQ-9 score")
    confidence: float = Field(ge=0, le=1)
    severity_level: str = Field(description="Minimal, Mild, Moderate, Moderately severe, Severe")
    contributing_expressions: List[str] = Field(description="Facial expressions contributing to score")

class ComprehensiveFacialAnalysis(BaseModel):
    """Complete facial analysis result with all metrics."""
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    face_detected: bool
    
    # Basic emotion analysis
    primary_emotion: str
    emotion_confidence: float
    emotion_distribution: EmotionDistribution
    
    # Raw facial metrics
    eye_metrics: Optional[EyeMetrics] = None
    head_pose: Optional[HeadPoseMetrics] = None
    micro_expressions: Optional[MicroExpressionMetrics] = None
    
    # Processed insights
    mood_assessment: str  # Happy, Sad, Angry, Neutral, Surprised
    sleepiness: SleepinessLevel
    fatigue: FatigueIndicators
    stress: StressLevel
    phq9_estimation: PHQ9Estimation
    
    # Overall analysis quality
    analysis_quality: float = Field(ge=0, le=1, description="Quality of analysis (0-1)")
    frame_quality: float = Field(ge=0, le=1, description="Quality of input frame (0-1)")

class FacialAnalysisSession(BaseModel):
    """Session tracking for continuous facial analysis."""
    session_id: str
    user_id: str
    start_time: datetime
    end_time: Optional[datetime] = None
    total_frames_analyzed: int = 0
    average_quality: float = 0.0
    session_summary: Optional[Dict] = None

class FacialAnalysisRequest(BaseModel):
    """Request model for comprehensive facial analysis."""
    image: str = Field(description="Base64 encoded image")
    session_id: Optional[str] = None
    user_id: Optional[str] = None
    include_raw_metrics: bool = True
    include_phq9_estimation: bool = True

class FacialAnalysisHistory(BaseModel):
    """Historical facial analysis data for trend analysis."""
    user_id: str
    date: str
    analyses: List[ComprehensiveFacialAnalysis]
    daily_summary: Dict = Field(description="Aggregated daily metrics")
