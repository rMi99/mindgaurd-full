"""
Health Recommendation Service - Provides personalized health recommendations,
exercises, games, and real-time monitoring based on facial expression analysis.
"""

import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import json
import uuid

from app.models.health_recommendations import (
    HealthRecommendations, HealthStatus, MoodCategory, Exercise, ExerciseType,
    Game, GameCategory, BiometricIndicators, RealtimeMonitoring, ProgressMetrics,
    SessionProgress
)
from app.models.facial_metrics import ComprehensiveFacialAnalysis

logger = logging.getLogger(__name__)

class HealthRecommendationService:
    """Service for generating health recommendations based on facial analysis."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.session_data = {}  # Store session tracking data
        self.exercises = []  # Will be loaded from database or defaults
        self.games = []  # Will be loaded from database or defaults
        self.initialized = False

    def _get_emotions_dict(self, analysis: ComprehensiveFacialAnalysis) -> dict:
        """Helper method to safely extract emotions dict from analysis."""
        try:
            if hasattr(analysis, 'emotion_distribution'):
                emotion_dist = analysis.emotion_distribution
                # Handle both direct dict access and object attribute access
                if hasattr(emotion_dist, 'model_dump'):
                    return emotion_dist.model_dump()
                elif isinstance(emotion_dist, dict):
                    return emotion_dist
                else:
                    # Fallback: try to extract attributes with proper field names
                    emotions = {}
                    # Map both 'fearful' and 'fear' to 'fear' for consistency
                    fear_val = getattr(emotion_dist, 'fear', None) or getattr(emotion_dist, 'fearful', 0.0)
                    
                    emotions = {
                        'happy': getattr(emotion_dist, 'happy', 0.0),
                        'sad': getattr(emotion_dist, 'sad', 0.0),
                        'angry': getattr(emotion_dist, 'angry', 0.0),
                        'fear': fear_val,
                        'surprise': getattr(emotion_dist, 'surprise', 0.0),
                        'disgust': getattr(emotion_dist, 'disgust', 0.0),
                        'neutral': getattr(emotion_dist, 'neutral', 0.0)
                    }
                    return emotions
            else:
                # Fallback to basic emotions from analysis
                return {
                    'happy': 0.1,
                    'sad': 0.1,
                    'angry': 0.1,
                    'fear': 0.1,
                    'surprise': 0.1,
                    'disgust': 0.1,
                    'neutral': 0.5
                }
        except Exception as e:
            print(f"Error extracting emotions: {e}")
            return {
                'happy': 0.1,
                'sad': 0.1,
                'angry': 0.1,
                'fear': 0.1,
                'surprise': 0.1,
                'disgust': 0.1,
                'neutral': 0.5
            }

    def _get_fatigue_level(self, analysis: ComprehensiveFacialAnalysis) -> float:
        """Helper method to safely extract fatigue level."""
        try:
            if hasattr(analysis, 'fatigue'):
                fatigue = analysis.fatigue
                if hasattr(fatigue, 'overall_fatigue'):
                    # Convert boolean to float (True = 1.0, False = 0.0)
                    if isinstance(fatigue.overall_fatigue, bool):
                        return 1.0 if fatigue.overall_fatigue else 0.0
                    else:
                        return float(fatigue.overall_fatigue)
                # Fallback: check for other fatigue indicators
                if hasattr(fatigue, 'yawning_detected') and fatigue.yawning_detected:
                    return 0.8
                if hasattr(fatigue, 'head_droop_detected') and fatigue.head_droop_detected:
                    return 0.7
            return 0.0
        except Exception as e:
            print(f"Error getting fatigue level: {e}")
            return 0.0

    def _create_progress_metrics(self, analysis: ComprehensiveFacialAnalysis) -> ProgressMetrics:
        """Create progress metrics from current analysis."""
        try:
            emotions = self._get_emotions_dict(analysis)
            current_mood = emotions.get('happy', 0) + emotions.get('neutral', 0) - emotions.get('sad', 0)
            
            return ProgressMetrics(
                baseline_mood=current_mood,  # For now, using current as baseline
                current_mood=current_mood,
                post_exercise_mood=None,  # Will be updated after exercises
                improvement_percentage=0.0,  # Initial session
                session_duration=0,  # Will be updated during session
                exercises_completed=0,
                games_played=0
            )
        except Exception as e:
            self.logger.error(f"Error creating progress metrics: {e}")
            return ProgressMetrics(
                baseline_mood=0.5,
                current_mood=0.5,
                post_exercise_mood=None,
                improvement_percentage=0.0,
                session_duration=0,
                exercises_completed=0,
                games_played=0
            )
        
    def analyze_health_status(self, facial_analysis: ComprehensiveFacialAnalysis) -> HealthRecommendations:
        """Generate comprehensive health recommendations based on facial analysis."""
        try:
            # Ensure service is initialized
            if not self.initialized:
                # Fallback to synchronous initialization
                self.exercises = self._initialize_exercise_database()
                self.games = self._initialize_game_database()
                self.initialized = True
            
            # Determine overall health status
            health_status = self._assess_health_status(facial_analysis)
            mood_category = self._categorize_mood(facial_analysis)
            priority_level = self._determine_priority(health_status, facial_analysis)
            
            # Generate recommendations
            recommended_exercises = self._recommend_exercises(mood_category, health_status, facial_analysis)
            suggested_games = self._recommend_games(mood_category, health_status)
            immediate_actions = self._generate_immediate_actions(health_status, facial_analysis)
            
            # Calculate biometric indicators
            biometric_indicators = self._calculate_biometric_indicators(facial_analysis)
            
            # Generate lifestyle tips
            lifestyle_tips = self._generate_lifestyle_tips(mood_category, health_status)
            when_to_seek_help = self._assess_help_needed(health_status, facial_analysis)
            
            # Create progress metrics
            progress_metrics = self._create_progress_metrics(facial_analysis)
            
            return HealthRecommendations(
                health_status=health_status,
                mood_category=mood_category,
                priority_level=priority_level,
                recommended_exercises=recommended_exercises,
                immediate_actions=immediate_actions,
                suggested_games=suggested_games,
                biometric_indicators=biometric_indicators,
                progress_metrics=progress_metrics,
                lifestyle_tips=lifestyle_tips,
                when_to_seek_help=when_to_seek_help
            )
            
        except Exception as e:
            logger.error(f"Error generating health recommendations: {e}")
            return self._generate_fallback_recommendations()
    
    def create_realtime_monitoring(self, session_id: str, facial_analysis: ComprehensiveFacialAnalysis, 
                                 health_recommendations: HealthRecommendations) -> RealtimeMonitoring:
        """Create real-time monitoring data."""
        try:
            current_time = datetime.utcnow()
            
            # Get or create session data
            if session_id not in self.session_data:
                self.session_data[session_id] = {
                    'start_time': current_time,
                    'mood_history': [],
                    'stress_history': [],
                    'total_frames': 0,
                    'mood_scores': []
                }
            
            session = self.session_data[session_id]
            session['total_frames'] += 1
            
            # Calculate current mood score
            mood_score = self._calculate_mood_score(facial_analysis)
            session['mood_scores'].append(mood_score)
            session['mood_history'].append(mood_score)
            session['stress_history'].append(health_recommendations.biometric_indicators.stress_level)
            
            # Keep only last 10 readings for trends
            mood_trend = session['mood_history'][-10:]
            stress_trend = session['stress_history'][-10:]
            
            # Generate alerts
            active_alerts = self._generate_alerts(health_recommendations, mood_trend, stress_trend)
            recommendations_queue = self._generate_recommendation_queue(health_recommendations)
            
            # Calculate average mood
            avg_mood = sum(session['mood_scores']) / len(session['mood_scores'])
            
            return RealtimeMonitoring(
                timestamp=current_time,
                session_id=session_id,
                current_health_status=health_recommendations.health_status,
                mood_trend=mood_trend,
                stress_trend=stress_trend,
                brain_activity=health_recommendations.biometric_indicators,
                active_alerts=active_alerts,
                recommendations_queue=recommendations_queue,
                session_start_time=session['start_time'],
                total_analysis_frames=session['total_frames'],
                average_mood_score=avg_mood
            )
            
        except Exception as e:
            logger.error(f"Error creating realtime monitoring: {e}")
            return self._generate_fallback_monitoring(session_id)
    
    def track_progress(self, session_id: str, user_id: Optional[str] = None) -> Optional[SessionProgress]:
        """Track user progress throughout the session."""
        try:
            if session_id not in self.session_data:
                return None
            
            session = self.session_data[session_id]
            current_time = datetime.utcnow()
            
            # Calculate progress metrics
            if len(session['mood_scores']) > 1:
                baseline_mood = session['mood_scores'][0]
                current_mood = session['mood_scores'][-1]
                improvement = ((current_mood - baseline_mood) / baseline_mood) * 100 if baseline_mood > 0 else 0
                
                progress_metrics = ProgressMetrics(
                    baseline_mood=baseline_mood,
                    current_mood=current_mood,
                    post_exercise_mood=session.get('post_exercise_mood'),
                    improvement_percentage=improvement,
                    session_duration=int((current_time - session['start_time']).total_seconds() / 60),
                    exercises_completed=len(session.get('exercises_completed', [])),
                    games_played=len(session.get('games_played', []))
                )
                
                return SessionProgress(
                    session_id=session_id,
                    user_id=user_id,
                    start_time=session['start_time'],
                    current_time=current_time,
                    initial_assessment=session.get('initial_assessment'),
                    current_assessment=session.get('current_assessment'),
                    progress_metrics=progress_metrics,
                    exercises_completed=session.get('exercises_completed', []),
                    games_played=session.get('games_played', []),
                    recommendation_effectiveness=session.get('recommendation_effectiveness', {}),
                    next_recommendations=self._generate_next_recommendations(progress_metrics)
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error tracking progress: {e}")
            return None
    
    def _assess_health_status(self, analysis: ComprehensiveFacialAnalysis) -> HealthStatus:
        """Assess overall health status from facial analysis."""
        try:
            # Calculate composite health score using correct field names
            stress_score = 1 - analysis.stress.confidence if hasattr(analysis, 'stress') and hasattr(analysis.stress, 'confidence') else 0.5
            fatigue_score = 1 - analysis.fatigue.overall_fatigue if hasattr(analysis, 'fatigue') and hasattr(analysis.fatigue, 'overall_fatigue') else 0.8
            
            # Access emotions through helper method
            emotions = self._get_emotions_dict(analysis)
            
            mood_score = emotions.get('happy', 0) + emotions.get('neutral', 0)
            
            overall_score = (stress_score + fatigue_score + mood_score) / 3
            
            if overall_score >= 0.8:
                return HealthStatus.EXCELLENT
            elif overall_score >= 0.6:
                return HealthStatus.GOOD
            elif overall_score >= 0.4:
                return HealthStatus.MODERATE
            elif overall_score >= 0.2:
                return HealthStatus.CONCERNING
            else:
                return HealthStatus.CRITICAL
                
        except Exception as e:
            logger.error(f"Error assessing health status: {e}")
            return HealthStatus.MODERATE
    
    def _categorize_mood(self, analysis: ComprehensiveFacialAnalysis) -> MoodCategory:
        """Categorize mood from facial analysis."""
        try:
            # Access emotions through helper method
            emotions = self._get_emotions_dict(analysis)
            
            if not emotions:
                return MoodCategory.NEUTRAL
                
            dominant = max(emotions.items(), key=lambda x: x[1])
            
            if dominant[0] == 'happy' and dominant[1] > 0.6:
                return MoodCategory.POSITIVE
            elif dominant[0] in ['sad'] and dominant[1] > 0.4:
                return MoodCategory.DEPRESSED
            elif dominant[0] in ['angry', 'fear'] and dominant[1] > 0.4:
                return MoodCategory.ANXIOUS
            elif hasattr(analysis, 'stress') and hasattr(analysis.stress, 'confidence') and analysis.stress.confidence > 0.6:
                return MoodCategory.STRESSED
            elif self._get_fatigue_level(analysis) > 0.5:
                return MoodCategory.FATIGUED
            elif hasattr(analysis, 'sleepiness') and analysis.sleepiness.level == "Alert":
                return MoodCategory.ALERT
            else:
                return MoodCategory.NEUTRAL
                
        except Exception as e:
            logger.error(f"Error categorizing mood: {e}")
            return MoodCategory.NEUTRAL
    
    def _determine_priority(self, health_status: HealthStatus, analysis: ComprehensiveFacialAnalysis) -> str:
        """Determine priority level for recommendations."""
        if health_status == HealthStatus.CRITICAL:
            return "Critical"
        elif health_status == HealthStatus.CONCERNING:
            return "High"
        elif self._get_fatigue_level(analysis) > 0.7 or (hasattr(analysis, 'sleepiness') and analysis.sleepiness.level == "Very tired"):
            return "Medium"
        else:
            return "Low"
    
    def _recommend_exercises(self, mood: MoodCategory, health_status: HealthStatus, 
                           analysis: ComprehensiveFacialAnalysis) -> List[Exercise]:
        """Recommend exercises based on mood and health status."""
        try:
            recommendations = []
            
            # Filter exercises based on mood and health status
            for exercise in self.exercises:
                if self._is_exercise_suitable(exercise, mood, health_status, analysis):
                    recommendations.append(exercise)
            
            # Prioritize and limit to top 3-5 exercises
            return sorted(recommendations, key=lambda x: self._calculate_exercise_priority(x, mood, health_status))[:5]
            
        except Exception as e:
            logger.error(f"Error recommending exercises: {e}")
            return self._get_default_exercises()
    
    def _recommend_games(self, mood: MoodCategory, health_status: HealthStatus) -> List[Game]:
        """Recommend games based on mood and health status."""
        try:
            recommendations = []
            
            for game in self.games:
                if mood in game.target_mood or self._is_game_suitable(game, mood, health_status):
                    recommendations.append(game)
            
            return sorted(recommendations, key=lambda x: self._calculate_game_priority(x, mood, health_status))[:3]
            
        except Exception as e:
            logger.error(f"Error recommending games: {e}")
            return self._get_default_games()
    
    def _calculate_biometric_indicators(self, analysis: ComprehensiveFacialAnalysis) -> BiometricIndicators:
        """Calculate biometric indicators from facial analysis."""
        try:
            # Use helper methods for safe field access
            emotions = self._get_emotions_dict(analysis)
            fatigue_level = self._get_fatigue_level(analysis)
            
            stress_level = analysis.stress.confidence if hasattr(analysis, 'stress') and hasattr(analysis.stress, 'confidence') else 0.3
            focus_level = 1 - stress_level if hasattr(analysis, 'sleepiness') and analysis.sleepiness.level == "Alert" else 0.4
            energy_level = 1 - fatigue_level
            emotional_stability = emotions.get('neutral', 0) + emotions.get('happy', 0)
            cognitive_load = stress_level + (0.3 if fatigue_level > 0.5 else 0)
            alertness = 1 - (0.8 if hasattr(analysis, 'sleepiness') and analysis.sleepiness.level == "Very tired" else 0.3 if hasattr(analysis, 'sleepiness') and analysis.sleepiness.level == "Slightly tired" else 0)
            
            return BiometricIndicators(
                stress_level=min(1.0, stress_level),
                focus_level=min(1.0, focus_level),
                energy_level=min(1.0, energy_level),
                emotional_stability=min(1.0, emotional_stability),
                cognitive_load=min(1.0, cognitive_load),
                alertness=min(1.0, alertness)
            )
            
        except Exception as e:
            logger.error(f"Error calculating biometric indicators: {e}")
            return BiometricIndicators(
                stress_level=0.5, focus_level=0.5, energy_level=0.5,
                emotional_stability=0.5, cognitive_load=0.5, alertness=0.5
            )
    
    def _calculate_mood_score(self, analysis: ComprehensiveFacialAnalysis) -> float:
        """Calculate overall mood score."""
        try:
            emotions = self._get_emotions_dict(analysis)
            positive_emotions = emotions.get('happy', 0) + emotions.get('surprise', 0)
            negative_emotions = emotions.get('sad', 0) + emotions.get('angry', 0) + emotions.get('fear', 0)
            neutral = emotions.get('neutral', 0)
            
            return (positive_emotions + neutral * 0.5) - negative_emotions * 0.5
            
        except Exception as e:
            logger.error(f"Error calculating mood score: {e}")
            return 0.5
    
    def _generate_alerts(self, recommendations: HealthRecommendations, mood_trend: List[float], 
                        stress_trend: List[float]) -> List[str]:
        """Generate active alerts based on current status."""
        alerts = []
        
        if recommendations.health_status == HealthStatus.CRITICAL:
            alerts.append("⚠️ Critical health status detected - immediate attention recommended")
        
        if recommendations.priority_level == "Critical":
            alerts.append("🚨 High priority recommendations available")
        
        if len(mood_trend) >= 3 and all(mood_trend[i] > mood_trend[i+1] for i in range(len(mood_trend)-1)):
            alerts.append("📉 Declining mood trend detected")
        
        if len(stress_trend) >= 3 and all(s > 0.7 for s in stress_trend[-3:]):
            alerts.append("😰 Elevated stress levels detected")
        
        if recommendations.biometric_indicators.energy_level < 0.3:
            alerts.append("🔋 Low energy levels - rest recommended")
        
        return alerts
    
    def _generate_immediate_actions(self, health_status: HealthStatus, analysis: ComprehensiveFacialAnalysis) -> List[str]:
        """Generate immediate actions to take."""
        actions = []
        
        if health_status == HealthStatus.CRITICAL:
            actions.extend([
                "Take a 5-minute break from current activity",
                "Practice deep breathing for 2 minutes",
                "Consider seeking professional support if feelings persist"
            ])
        elif self._get_fatigue_level(analysis) > 0.6:
            actions.extend([
                "Take a short 2-3 minute break",
                "Do gentle neck and shoulder rolls",
                "Hydrate with a glass of water"
            ])
        elif hasattr(analysis, 'sleepiness') and analysis.sleepiness.level == "Very tired":
            actions.extend([
                "Consider a 10-15 minute power nap if possible",
                "Do eye exercises to reduce strain",
                "Get some fresh air or natural light"
            ])
        else:
            actions.extend([
                "Continue current activity with regular breaks",
                "Maintain good posture",
                "Stay hydrated"
            ])
        
        return actions
    
    async def _load_exercises_from_db(self):
        """Load exercises from MongoDB database."""
        try:
            from .db import get_db
            db = await get_db()
            
            exercises_collection = db['exercises']
            exercises_cursor = exercises_collection.find({})
            
            self.exercises = []
            async for exercise_doc in exercises_cursor:
                try:
                    # Remove MongoDB _id field for Pydantic validation
                    if '_id' in exercise_doc:
                        exercise_doc['id'] = exercise_doc.pop('_id')
                    
                    exercise = Exercise(**exercise_doc)
                    self.exercises.append(exercise)
                except Exception as e:
                    print(f"Error loading exercise {exercise_doc.get('id', 'unknown')}: {e}")
            
            print(f"Loaded {len(self.exercises)} exercises from database")
            
        except Exception as e:
            print(f"Error loading exercises from database: {e}")
            # Fallback to default exercises
            self.exercises = self._initialize_exercise_database()
    
    async def _load_games_from_db(self):
        """Load games from MongoDB database."""
        try:
            from .db import get_db
            db = await get_db()
            
            games_collection = db['games']
            games_cursor = games_collection.find({})
            
            self.games = []
            async for game_doc in games_cursor:
                try:
                    # Remove MongoDB _id field for Pydantic validation
                    if '_id' in game_doc:
                        game_doc['id'] = game_doc.pop('_id')
                    
                    game = Game(**game_doc)
                    self.games.append(game)
                except Exception as e:
                    print(f"Error loading game {game_doc.get('id', 'unknown')}: {e}")
            
            print(f"Loaded {len(self.games)} games from database")
            
        except Exception as e:
            print(f"Error loading games from database: {e}")
            # Fallback to default games
            self.games = self._initialize_game_database()
    
    async def initialize_from_database(self):
        """Initialize exercises and games from database, fallback to defaults if needed."""
        await self._load_exercises_from_db()
        await self._load_games_from_db()
        
        # If no data loaded, initialize with defaults
        if not self.exercises:
            print("No exercises found in database, using defaults")
            self.exercises = self._initialize_exercise_database()
        
        if not self.games:
            print("No games found in database, using defaults")
            self.games = self._initialize_game_database()

    def _initialize_exercise_database(self) -> List[Exercise]:
        """Initialize the exercise database with predefined exercises."""
        return [
            Exercise(
                id="breathing_4_7_8",
                name="4-7-8 Breathing Technique",
                type=ExerciseType.BREATHING,
                description="A calming breathing exercise to reduce stress and anxiety",
                duration_minutes=3,
                difficulty="Easy",
                instructions=[
                    "Sit comfortably with your back straight",
                    "Exhale completely through your mouth",
                    "Inhale through your nose for 4 counts",
                    "Hold your breath for 7 counts",
                    "Exhale through your mouth for 8 counts",
                    "Repeat 3-4 times"
                ],
                benefits=["Reduces stress", "Calms nervous system", "Improves focus"],
                target_conditions=["stress", "anxiety", "high blood pressure"]
            ),
            Exercise(
                id="progressive_muscle_relaxation",
                name="Progressive Muscle Relaxation",
                type=ExerciseType.STRETCHING,
                description="Systematic tensing and relaxing of muscle groups",
                duration_minutes=10,
                difficulty="Easy",
                instructions=[
                    "Sit or lie down comfortably",
                    "Start with your toes - tense for 5 seconds, then relax",
                    "Move up to calves, thighs, abdomen, arms, and face",
                    "Focus on the contrast between tension and relaxation",
                    "End with 2 minutes of deep breathing"
                ],
                benefits=["Reduces muscle tension", "Promotes relaxation", "Improves sleep"],
                target_conditions=["stress", "fatigue", "insomnia"]
            ),
            Exercise(
                id="eye_focus_exercise",
                name="20-20-20 Eye Exercise",
                type=ExerciseType.EYE,
                description="Eye exercise to reduce strain and improve focus",
                duration_minutes=2,
                difficulty="Easy",
                instructions=[
                    "Every 20 minutes, look away from your screen",
                    "Focus on something 20 feet away",
                    "Hold focus for 20 seconds",
                    "Blink deliberately 20 times",
                    "Return to your activity"
                ],
                benefits=["Reduces eye strain", "Improves focus", "Prevents dry eyes"],
                target_conditions=["eye strain", "fatigue", "focus issues"]
            ),
            Exercise(
                id="facial_tension_release",
                name="Facial Tension Release",
                type=ExerciseType.FACIAL,
                description="Release tension from facial muscles",
                duration_minutes=5,
                difficulty="Easy",
                instructions=[
                    "Sit comfortably and close your eyes",
                    "Scrunch up your face tightly for 5 seconds",
                    "Release and let your face go completely slack",
                    "Massage your temples in circular motions",
                    "Gently massage your jaw muscles",
                    "End with a gentle smile"
                ],
                benefits=["Reduces facial tension", "Improves circulation", "Promotes relaxation"],
                target_conditions=["stress", "tension headaches", "jaw pain"]
            ),
            Exercise(
                id="mindful_meditation",
                name="5-Minute Mindfulness",
                type=ExerciseType.MEDITATION,
                description="Brief mindfulness meditation for mental clarity",
                duration_minutes=5,
                difficulty="Medium",
                instructions=[
                    "Sit quietly and close your eyes",
                    "Focus on your breath naturally flowing",
                    "When thoughts arise, acknowledge them and return to breath",
                    "Notice sensations in your body without judgment",
                    "End by slowly opening your eyes"
                ],
                benefits=["Improves focus", "Reduces anxiety", "Enhances self-awareness"],
                target_conditions=["anxiety", "stress", "poor focus"]
            )
        ]
    
    def _initialize_game_database(self) -> List[Game]:
        """Initialize the game database with mental stimulation games."""
        return [
            Game(
                id="memory_sequence",
                name="Memory Sequence Challenge",
                category=GameCategory.MEMORY,
                description="Remember and repeat increasingly complex sequences",
                estimated_time=5,
                difficulty="Medium",
                instructions="Watch the sequence of colors/numbers and repeat it back in order",
                benefits=["Improves working memory", "Enhances concentration", "Boosts cognitive flexibility"],
                target_mood=[MoodCategory.ALERT, MoodCategory.NEUTRAL]
            ),
            Game(
                id="breathing_rhythm",
                name="Rhythmic Breathing Game",
                category=GameCategory.RELAXATION,
                description="Follow visual cues for optimal breathing patterns",
                estimated_time=3,
                difficulty="Easy",
                instructions="Match your breathing to the expanding and contracting circle",
                benefits=["Reduces stress", "Improves focus", "Promotes relaxation"],
                target_mood=[MoodCategory.STRESSED, MoodCategory.ANXIOUS]
            ),
            Game(
                id="color_attention",
                name="Color Focus Challenge",
                category=GameCategory.ATTENTION,
                description="Identify specific colors while ignoring distractors",
                estimated_time=4,
                difficulty="Medium",
                instructions="Click only on the target color while ignoring other colors",
                benefits=["Improves selective attention", "Enhances processing speed", "Boosts focus"],
                target_mood=[MoodCategory.NEUTRAL, MoodCategory.ALERT]
            ),
            Game(
                id="puzzle_solving",
                name="Quick Logic Puzzles",
                category=GameCategory.PROBLEM_SOLVING,
                description="Solve simple logic puzzles to stimulate thinking",
                estimated_time=7,
                difficulty="Medium",
                instructions="Use logical reasoning to solve each puzzle step by step",
                benefits=["Enhances problem-solving", "Improves logical thinking", "Boosts confidence"],
                target_mood=[MoodCategory.ALERT, MoodCategory.POSITIVE]
            ),
            Game(
                id="creative_visualization",
                name="Guided Visualization",
                category=GameCategory.CREATIVITY,
                description="Creative visualization exercise for mental stimulation",
                estimated_time=6,
                difficulty="Easy",
                instructions="Follow the guided imagery to create vivid mental pictures",
                benefits=["Enhances creativity", "Reduces stress", "Improves mood"],
                target_mood=[MoodCategory.DEPRESSED, MoodCategory.FATIGUED, MoodCategory.STRESSED]
            )
        ]
    
    def _is_exercise_suitable(self, exercise: Exercise, mood: MoodCategory, 
                            health_status: HealthStatus, analysis: ComprehensiveFacialAnalysis) -> bool:
        """Check if an exercise is suitable for current conditions."""
        # Check health status compatibility
        if health_status == HealthStatus.CRITICAL and exercise.difficulty == "Hard":
            return False
        
        # Check mood-specific suitability
        if mood == MoodCategory.STRESSED and exercise.type in [ExerciseType.BREATHING, ExerciseType.MEDITATION]:
            return True
        elif mood == MoodCategory.FATIGUED and exercise.type == ExerciseType.EYE:
            return True
        elif self._get_fatigue_level(analysis) > 0.5 and exercise.type == ExerciseType.STRETCHING:
            return True
        
        return True  # Default to suitable
    
    def _calculate_exercise_priority(self, exercise: Exercise, mood: MoodCategory, health_status: HealthStatus) -> float:
        """Calculate priority score for exercise recommendation."""
        priority = 0.5  # Base priority
        
        # Mood-based priority adjustments
        if mood == MoodCategory.STRESSED and exercise.type == ExerciseType.BREATHING:
            priority += 0.3
        elif mood == MoodCategory.FATIGUED and exercise.type in [ExerciseType.EYE, ExerciseType.STRETCHING]:
            priority += 0.2
        
        # Health status adjustments
        if health_status in [HealthStatus.CONCERNING, HealthStatus.CRITICAL]:
            if exercise.difficulty == "Easy":
                priority += 0.2
            else:
                priority -= 0.1
        
        return priority
    
    def _is_game_suitable(self, game: Game, mood: MoodCategory, health_status: HealthStatus) -> bool:
        """Check if a game is suitable for current conditions."""
        if health_status == HealthStatus.CRITICAL and game.difficulty == "Hard":
            return False
        return True
    
    def _calculate_game_priority(self, game: Game, mood: MoodCategory, health_status: HealthStatus) -> float:
        """Calculate priority score for game recommendation."""
        priority = 0.5
        
        if mood in game.target_mood:
            priority += 0.3
        
        if health_status in [HealthStatus.CONCERNING, HealthStatus.CRITICAL] and game.category == GameCategory.RELAXATION:
            priority += 0.2
        
        return priority
    
    def _generate_lifestyle_tips(self, mood: MoodCategory, health_status: HealthStatus) -> List[str]:
        """Generate lifestyle tips based on current status."""
        tips = [
            "Maintain regular sleep schedule (7-9 hours per night)",
            "Stay hydrated throughout the day",
            "Take regular breaks from screen time"
        ]
        
        if mood == MoodCategory.STRESSED:
            tips.extend([
                "Practice stress management techniques daily",
                "Limit caffeine intake in the afternoon",
                "Consider talking to someone about your stress"
            ])
        elif mood == MoodCategory.FATIGUED:
            tips.extend([
                "Ensure adequate nutrition throughout the day",
                "Get natural sunlight exposure",
                "Consider a brief walk outside"
            ])
        
        return tips
    
    def _assess_help_needed(self, health_status: HealthStatus, analysis: ComprehensiveFacialAnalysis) -> Optional[str]:
        """Assess when professional help might be needed."""
        if health_status == HealthStatus.CRITICAL:
            return "Consider speaking with a healthcare professional if these feelings persist for more than a few days."
        elif health_status == HealthStatus.CONCERNING:
            return "If symptoms continue or worsen, consider reaching out to a mental health professional."
        return None
    
    def _generate_next_recommendations(self, progress: ProgressMetrics) -> List[str]:
        """Generate next recommendations based on progress."""
        recommendations = []
        
        if progress.improvement_percentage > 10:
            recommendations.append("Great progress! Continue with current activities")
        elif progress.improvement_percentage < -10:
            recommendations.append("Consider trying different exercises or taking a longer break")
        else:
            recommendations.append("Steady progress - maintain current routine")
        
        if progress.exercises_completed < 2:
            recommendations.append("Try completing 1-2 more exercises this session")
        
        return recommendations
    
    # Fallback methods for error handling
    def _generate_fallback_recommendations(self) -> HealthRecommendations:
        """Generate fallback recommendations when analysis fails."""
        return HealthRecommendations(
            health_status=HealthStatus.MODERATE,
            mood_category=MoodCategory.NEUTRAL,
            priority_level="Medium",
            recommended_exercises=self._get_default_exercises()[:3],
            immediate_actions=["Take a short break", "Practice deep breathing"],
            suggested_games=self._get_default_games()[:2],
            biometric_indicators=BiometricIndicators(
                stress_level=0.5, focus_level=0.5, energy_level=0.5,
                emotional_stability=0.5, cognitive_load=0.5, alertness=0.5
            ),
            progress_metrics=ProgressMetrics(
                baseline_mood=0.5,
                current_mood=0.5,
                post_exercise_mood=None,
                improvement_percentage=0.0,
                session_duration=0,
                exercises_completed=0,
                games_played=0
            ),
            lifestyle_tips=["Stay hydrated", "Take regular breaks", "Get adequate sleep"],
            when_to_seek_help="Consider consulting a healthcare professional if symptoms persist"
        )
    
    def _generate_fallback_monitoring(self, session_id: str) -> RealtimeMonitoring:
        """Generate fallback monitoring data."""
        return RealtimeMonitoring(
            timestamp=datetime.utcnow(),
            session_id=session_id,
            current_health_status=HealthStatus.MODERATE,
            mood_trend=[0.5],
            stress_trend=[0.5],
            brain_activity=BiometricIndicators(
                stress_level=0.5, focus_level=0.5, energy_level=0.5,
                emotional_stability=0.5, cognitive_load=0.5, alertness=0.5
            ),
            active_alerts=[],
            recommendations_queue=["Take regular breaks", "Stay hydrated"],
            session_start_time=datetime.utcnow(),
            total_analysis_frames=1,
            average_mood_score=0.5
        )
    
    def _get_default_exercises(self) -> List[Exercise]:
        """Get default exercises for fallback."""
        return self.exercises[:3] if self.exercises else []
    
    def _get_default_games(self) -> List[Game]:
        """Get default games for fallback."""
        return self.games[:2] if self.games else []
    
    def _generate_recommendation_queue(self, recommendations: HealthRecommendations) -> List[str]:
        """Generate recommendation queue for display."""
        queue = []
        
        if recommendations.immediate_actions:
            queue.extend(recommendations.immediate_actions[:2])
        
        if recommendations.recommended_exercises:
            queue.append(f"Try: {recommendations.recommended_exercises[0].name}")
        
        if recommendations.suggested_games:
            queue.append(f"Play: {recommendations.suggested_games[0].name}")
        
        return queue[:5]  # Limit to 5 items
    
    def _initialize_exercise_database(self) -> List[Exercise]:
        """Initialize the exercise database with predefined exercises."""
        return [
            Exercise(
                id="breathing_basic",
                name="Deep Breathing Exercise",
                type=ExerciseType.BREATHING,
                description="A simple breathing exercise to reduce stress and improve focus",
                duration_minutes=3,
                difficulty="easy",
                instructions=[
                    "Sit comfortably with your back straight",
                    "Inhale slowly through your nose for 4 counts",
                    "Hold your breath for 4 counts",
                    "Exhale slowly through your mouth for 6 counts",
                    "Repeat 5-10 times"
                ],
                benefits=["Reduces stress", "Improves focus", "Lowers heart rate"],
                target_conditions=[MoodCategory.STRESSED.value, MoodCategory.ANXIOUS.value]
            ),
            Exercise(
                id="eye_relaxation",
                name="Eye Strain Relief",
                type=ExerciseType.EYE,
                description="Eye exercises to reduce strain from screen time",
                duration_minutes=2,
                difficulty="easy",
                instructions=[
                    "Look away from your screen",
                    "Focus on an object 20 feet away for 20 seconds",
                    "Blink slowly 10 times",
                    "Close your eyes gently for 10 seconds"
                ],
                benefits=["Reduces eye strain", "Improves focus", "Prevents dry eyes"],
                target_conditions=[MoodCategory.FATIGUED.value]
            ),
            Exercise(
                id="neck_stretch",
                name="Neck and Shoulder Stretch",
                type=ExerciseType.NECK_SHOULDER,
                description="Gentle stretches to relieve neck and shoulder tension",
                duration_minutes=3,
                difficulty="easy",
                instructions=[
                    "Gently tilt your head to the right",
                    "Hold for 10 seconds",
                    "Return to center and tilt to the left",
                    "Roll your shoulders backward 5 times",
                    "Roll your shoulders forward 5 times"
                ],
                benefits=["Relieves tension", "Improves posture", "Reduces stiffness"],
                target_conditions=[MoodCategory.STRESSED.value, MoodCategory.FATIGUED.value]
            )
        ]
    
    def _initialize_game_database(self) -> List[Game]:
        """Initialize the game database with predefined games."""
        return [
            Game(
                id="memory_sequence",
                name="Memory Sequence Challenge",
                category=GameCategory.MEMORY,
                description="Remember and repeat sequences of colors or numbers",
                estimated_time=5,
                difficulty="medium",
                instructions="Watch the sequence, then repeat it by clicking the correct order",
                benefits=["Improves working memory", "Enhances attention", "Boosts cognitive flexibility"],
                target_mood=[MoodCategory.FATIGUED, MoodCategory.NEUTRAL]
            ),
            Game(
                id="focus_breathing",
                name="Breathing Focus Game",
                category=GameCategory.FOCUS,
                description="Match your breathing to visual cues on screen",
                estimated_time=3,
                difficulty="easy",
                instructions="Follow the breathing guide on screen - inhale when it expands, exhale when it contracts",
                benefits=["Improves focus", "Reduces stress", "Enhances self-awareness"],
                target_mood=[MoodCategory.STRESSED, MoodCategory.ANXIOUS]
            ),
            Game(
                id="pattern_recognition",
                name="Pattern Recognition",
                category=GameCategory.ATTENTION,
                description="Identify patterns in sequences of shapes or colors",
                estimated_time=4,
                difficulty="medium",
                instructions="Look at the sequence and identify the missing pattern or next item",
                benefits=["Enhances pattern recognition", "Improves attention to detail", "Boosts analytical thinking"],
                target_mood=[MoodCategory.NEUTRAL, MoodCategory.ALERT]
            )
        ]
        
        return queue
