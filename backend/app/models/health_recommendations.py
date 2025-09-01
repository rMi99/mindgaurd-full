"""
Enhanced models for health recommendations, exercises, games, and real-time monitoring
based on facial expression analysis.
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Union, Any
from datetime import datetime
from enum import Enum

class HealthStatus(str, Enum):
    """Overall health status levels."""
    EXCELLENT = "excellent"
    GOOD = "good"
    MODERATE = "moderate"
    CONCERNING = "concerning"
    CRITICAL = "critical"

class MoodCategory(str, Enum):
    """Mood categories for targeted recommendations."""
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    STRESSED = "stressed"
    ANXIOUS = "anxious"
    DEPRESSED = "depressed"
    FATIGUED = "fatigued"
    ALERT = "alert"

class ExerciseType(str, Enum):
    """Types of exercises for recommendations."""
    BREATHING = "breathing"
    FACIAL = "facial"
    NECK_SHOULDER = "neck_shoulder"
    EYE = "eye"
    MEDITATION = "meditation"
    STRETCHING = "stretching"
    COGNITIVE = "cognitive"

class GameCategory(str, Enum):
    """Categories of games for mental stimulation."""
    MEMORY = "memory"
    ATTENTION = "attention"
    PROBLEM_SOLVING = "problem_solving"
    RELAXATION = "relaxation"
    FOCUS = "focus"
    CREATIVITY = "creativity"

class Exercise(BaseModel):
    """Individual exercise recommendation."""
    id: str = Field(description="Unique exercise identifier")
    name: str = Field(description="Exercise name")
    type: ExerciseType = Field(description="Type of exercise")
    description: str = Field(description="Detailed exercise description")
    duration_minutes: int = Field(description="Recommended duration in minutes")
    difficulty: str = Field(description="Easy, Medium, Hard")
    instructions: List[str] = Field(description="Step-by-step instructions")
    benefits: List[str] = Field(description="Expected benefits")
    target_conditions: List[str] = Field(description="Conditions this exercise helps with")

class Game(BaseModel):
    """Mental stimulation game recommendation."""
    id: str = Field(description="Unique game identifier")
    name: str = Field(description="Game name")
    category: GameCategory = Field(description="Game category")
    description: str = Field(description="Game description")
    estimated_time: int = Field(description="Estimated play time in minutes")
    difficulty: str = Field(description="Easy, Medium, Hard")
    instructions: str = Field(description="How to play")
    benefits: List[str] = Field(description="Cognitive benefits")
    target_mood: List[MoodCategory] = Field(description="Best moods for this game")

class BiometricIndicators(BaseModel):
    """Real-time biometric monitoring indicators."""
    stress_level: float = Field(ge=0, le=1, description="Current stress level (0-1)")
    focus_level: float = Field(ge=0, le=1, description="Current focus level (0-1)")
    energy_level: float = Field(ge=0, le=1, description="Current energy level (0-1)")
    emotional_stability: float = Field(ge=0, le=1, description="Emotional stability (0-1)")
    cognitive_load: float = Field(ge=0, le=1, description="Mental workload (0-1)")
    alertness: float = Field(ge=0, le=1, description="Alertness level (0-1)")

class ProgressMetrics(BaseModel):
    """Progress tracking metrics."""
    baseline_mood: float = Field(description="Initial mood score")
    current_mood: float = Field(description="Current mood score")
    post_exercise_mood: Optional[float] = Field(description="Mood after exercise")
    improvement_percentage: float = Field(description="Percentage improvement")
    session_duration: int = Field(description="Session duration in minutes")
    exercises_completed: int = Field(description="Number of exercises completed")
    games_played: int = Field(description="Number of games played")

class HealthRecommendations(BaseModel):
    """Comprehensive health recommendations based on facial analysis."""
    health_status: HealthStatus = Field(description="Overall health assessment")
    mood_category: MoodCategory = Field(description="Detected mood category")
    priority_level: str = Field(description="Low, Medium, High, Critical")
    
    # Exercise recommendations
    recommended_exercises: List[Exercise] = Field(description="Personalized exercise recommendations")
    immediate_actions: List[str] = Field(description="Immediate actions to take")
    
    # Game recommendations
    suggested_games: List[Game] = Field(description="Mental stimulation games")
    
    # Monitoring insights
    biometric_indicators: BiometricIndicators = Field(description="Real-time biometric data")
    
    # Progress tracking
    progress_metrics: Optional[ProgressMetrics] = Field(description="Progress tracking data")
    
    # General recommendations
    lifestyle_tips: List[str] = Field(description="General lifestyle recommendations")
    when_to_seek_help: Optional[str] = Field(description="When to seek professional help")

class RealtimeMonitoring(BaseModel):
    """Real-time monitoring dashboard data."""
    timestamp: datetime = Field(description="Current timestamp")
    session_id: str = Field(description="Current session identifier")
    
    # Current status
    current_health_status: HealthStatus = Field(description="Current health status")
    mood_trend: List[float] = Field(description="Mood trend over last 10 readings")
    stress_trend: List[float] = Field(description="Stress trend over last 10 readings")
    
    # Brain activity indicators
    brain_activity: BiometricIndicators = Field(description="Current brain activity metrics")
    
    # Alerts and notifications
    active_alerts: List[str] = Field(description="Current active alerts")
    recommendations_queue: List[str] = Field(description="Queued recommendations")
    
    # Session summary
    session_start_time: datetime = Field(description="Session start time")
    total_analysis_frames: int = Field(description="Total frames analyzed")
    average_mood_score: float = Field(description="Session average mood")

class EnhancedFacialAnalysisResult(BaseModel):
    """Enhanced facial analysis result with health recommendations."""
    # Original analysis data
    emotions: Dict[str, float] = Field(description="Emotion probabilities")
    dominant_emotion: str = Field(description="Primary detected emotion")
    confidence: float = Field(description="Analysis confidence")
    
    # Enhanced health analysis
    health_recommendations: HealthRecommendations = Field(description="Personalized health recommendations")
    realtime_monitoring: RealtimeMonitoring = Field(description="Real-time monitoring data")
    
    # Analysis metadata
    analysis_timestamp: datetime = Field(description="Analysis timestamp")
    user_id: Optional[str] = Field(description="User identifier")
    session_id: str = Field(description="Session identifier")
    image_quality: float = Field(description="Input image quality score")

class SessionProgress(BaseModel):
    """Session progress tracking."""
    session_id: str = Field(description="Session identifier")
    user_id: Optional[str] = Field(description="User identifier")
    start_time: datetime = Field(description="Session start time")
    current_time: datetime = Field(description="Current time")
    
    # Progress metrics
    initial_assessment: HealthRecommendations = Field(description="Initial health assessment")
    current_assessment: HealthRecommendations = Field(description="Current health assessment")
    progress_metrics: ProgressMetrics = Field(description="Progress tracking metrics")
    
    # Activities completed
    exercises_completed: List[Dict[str, Union[str, int, float, datetime]]] = Field(description="Completed exercises with timestamps")
    games_played: List[Dict[str, Union[str, int, float, datetime]]] = Field(description="Played games with scores")
    
    # Recommendations effectiveness
    recommendation_effectiveness: Dict[str, float] = Field(description="Effectiveness scores for recommendations")
    next_recommendations: List[str] = Field(description="Next recommended actions")
