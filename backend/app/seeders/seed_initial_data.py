#!/usr/bin/env python3
"""
Initial data seeder for MindGuard application.
Seeds exercises, games, and sample facial analysis data.
"""

import sys
import os
import logging
from typing import List

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.models.health_recommendations import (
    Exercise, ExerciseType, Game, GameCategory, MoodCategory,
    BiometricIndicators, ProgressMetrics
)
from app.models.facial_metrics import (
    ComprehensiveFacialAnalysis, EmotionDistribution, SleepinessLevel,
    FatigueIndicators, StressLevel, PHQ9Estimation, EyeMetrics,
    HeadPoseMetrics, MicroExpressionMetrics
)
from datetime import datetime
import asyncio

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataSeeder:
    """Handles seeding of initial application data."""
    
    def __init__(self):
        self.exercises = []
        self.games = []
        
    def seed_exercises(self) -> List[Exercise]:
        """Seed initial exercise data."""
        logger.info("Seeding exercise data...")
        
        exercises = [
            Exercise(
                id="breathing_basic",
                name="Basic Breathing Exercise",
                type=ExerciseType.BREATHING,
                description="A guided breathing exercise to reduce stress and improve focus",
                duration_minutes=3,
                difficulty="easy",
                instructions=[
                    "Sit comfortably with your back straight",
                    "Close your eyes or soften your gaze",
                    "Inhale slowly through your nose for 4 counts",
                    "Hold your breath for 4 counts",
                    "Exhale slowly through your mouth for 6 counts",
                    "Repeat 5-10 times"
                ],
                benefits=["Reduces stress", "Improves focus", "Lowers heart rate", "Calms nervous system"],
                target_conditions=[MoodCategory.STRESSED.value, MoodCategory.ANXIOUS.value]
            ),
            Exercise(
                id="eye_strain_relief",
                name="Eye Strain Relief",
                type=ExerciseType.EYE,
                description="Quick eye exercises to reduce digital eye strain",
                duration_minutes=2,
                difficulty="easy",
                instructions=[
                    "Look away from your screen",
                    "Focus on an object 20 feet away for 20 seconds (20-20-20 rule)",
                    "Blink slowly and deliberately 10 times",
                    "Close your eyes gently for 10 seconds",
                    "Look up, down, left, right - hold each for 3 seconds"
                ],
                benefits=["Reduces eye strain", "Prevents dry eyes", "Improves focus", "Relaxes eye muscles"],
                target_conditions=[MoodCategory.FATIGUED.value, MoodCategory.STRESSED.value]
            ),
            Exercise(
                id="neck_shoulder_stretch",
                name="Neck and Shoulder Release",
                type=ExerciseType.NECK_SHOULDER,
                description="Gentle stretches to relieve neck and shoulder tension",
                duration_minutes=4,
                difficulty="easy",
                instructions=[
                    "Sit up straight with shoulders relaxed",
                    "Slowly tilt your head to the right, hold for 15 seconds",
                    "Return to center and tilt to the left, hold for 15 seconds",
                    "Look down, gently pulling chin to chest, hold for 15 seconds",
                    "Roll shoulders backward 5 times slowly",
                    "Roll shoulders forward 5 times slowly"
                ],
                benefits=["Relieves tension", "Improves posture", "Reduces stiffness", "Increases mobility"],
                target_conditions=[MoodCategory.STRESSED.value, MoodCategory.FATIGUED.value]
            ),
            Exercise(
                id="facial_tension_release",
                name="Facial Tension Release",
                type=ExerciseType.FACIAL,
                description="Facial massage and exercises to release tension",
                duration_minutes=3,
                difficulty="easy",
                instructions=[
                    "Place fingers on your temples",
                    "Gently massage in small circular motions for 30 seconds",
                    "Move to jaw muscles and massage gently",
                    "Open mouth wide, then close slowly",
                    "Smile as wide as possible, hold for 5 seconds",
                    "Raise eyebrows high, hold for 5 seconds"
                ],
                benefits=["Reduces facial tension", "Improves circulation", "Promotes relaxation", "Relieves jaw pain"],
                target_conditions=[MoodCategory.STRESSED.value, MoodCategory.ANXIOUS.value]
            ),
            Exercise(
                id="mindfulness_meditation",
                name="Quick Mindfulness",
                type=ExerciseType.MEDITATION,
                description="A brief mindfulness meditation for mental clarity",
                duration_minutes=5,
                difficulty="medium",
                instructions=[
                    "Find a comfortable seated position",
                    "Close your eyes and take 3 deep breaths",
                    "Focus your attention on your breathing",
                    "Notice thoughts without judgment, let them pass",
                    "Return focus to breath when mind wanders",
                    "Continue for the full 5 minutes"
                ],
                benefits=["Improves mindfulness", "Reduces anxiety", "Enhances focus", "Promotes emotional regulation"],
                target_conditions=[MoodCategory.ANXIOUS.value, MoodCategory.DEPRESSED.value, MoodCategory.STRESSED.value]
            ),
            Exercise(
                id="progressive_relaxation",
                name="Progressive Muscle Relaxation",
                type=ExerciseType.STRETCHING,
                description="Systematic tension and relaxation of muscle groups",
                duration_minutes=8,
                difficulty="medium",
                instructions=[
                    "Lie down or sit comfortably",
                    "Start with your toes - tense for 5 seconds, then relax",
                    "Move to calves - tense and relax",
                    "Continue with thighs, abdomen, arms, shoulders",
                    "Finally, tense and relax facial muscles",
                    "Notice the difference between tension and relaxation"
                ],
                benefits=["Reduces muscle tension", "Promotes deep relaxation", "Improves body awareness", "Reduces stress"],
                target_conditions=[MoodCategory.STRESSED.value, MoodCategory.FATIGUED.value, MoodCategory.ANXIOUS.value]
            )
        ]
        
        self.exercises = exercises
        logger.info(f"Successfully seeded {len(exercises)} exercises")
        return exercises
    
    def seed_games(self) -> List[Game]:
        """Seed initial game data."""
        logger.info("Seeding game data...")
        
        games = [
            Game(
                id="memory_sequence",
                name="Memory Sequence Challenge",
                category=GameCategory.MEMORY,
                description="Test and improve your working memory by remembering sequences of colors, numbers, or shapes",
                estimated_time=5,
                difficulty="medium",
                instructions="Watch the sequence carefully, then repeat it by clicking in the same order. Sequences get longer as you progress.",
                benefits=["Improves working memory", "Enhances attention span", "Boosts cognitive flexibility", "Increases concentration"],
                target_mood=[MoodCategory.FATIGUED, MoodCategory.NEUTRAL, MoodCategory.ALERT]
            ),
            Game(
                id="breathing_rhythm",
                name="Breathing Rhythm Game",
                category=GameCategory.FOCUS,
                description="A calming game that helps you focus on your breathing rhythm with visual and audio cues",
                estimated_time=3,
                difficulty="easy",
                instructions="Follow the breathing guide on screen. Inhale when the circle expands, exhale when it contracts. Try to match the rhythm perfectly.",
                benefits=["Improves focus", "Reduces stress", "Enhances self-awareness", "Promotes relaxation"],
                target_mood=[MoodCategory.STRESSED, MoodCategory.ANXIOUS, MoodCategory.NEUTRAL]
            ),
            Game(
                id="pattern_finder",
                name="Pattern Recognition",
                category=GameCategory.ATTENTION,
                description="Identify patterns and complete sequences in various visual puzzles",
                estimated_time=6,
                difficulty="medium",
                instructions="Look at the sequence of shapes, colors, or numbers and identify the pattern. Select the next item that should come in the sequence.",
                benefits=["Enhances pattern recognition", "Improves attention to detail", "Boosts analytical thinking", "Increases problem-solving skills"],
                target_mood=[MoodCategory.NEUTRAL, MoodCategory.ALERT, MoodCategory.POSITIVE]
            ),
            Game(
                id="word_association",
                name="Creative Word Links",
                category=GameCategory.CREATIVITY,
                description="Create meaningful connections between seemingly unrelated words to boost creativity",
                estimated_time=7,
                difficulty="easy",
                instructions="You'll see two random words. Create a story, find a connection, or explain how they could be related. Be as creative as possible!",
                benefits=["Boosts creativity", "Improves verbal fluency", "Enhances divergent thinking", "Stimulates imagination"],
                target_mood=[MoodCategory.DEPRESSED, MoodCategory.NEUTRAL, MoodCategory.POSITIVE]
            ),
            Game(
                id="attention_training",
                name="Sustained Attention Task",
                category=GameCategory.ATTENTION,
                description="Train your ability to maintain focus over extended periods",
                estimated_time=8,
                difficulty="hard",
                instructions="Watch for specific targets among distractors. Click when you see the target, but ignore everything else. Stay focused for the entire session.",
                benefits=["Improves sustained attention", "Reduces mind wandering", "Enhances concentration", "Builds mental stamina"],
                target_mood=[MoodCategory.FATIGUED, MoodCategory.NEUTRAL, MoodCategory.ALERT]
            ),
            Game(
                id="zen_garden",
                name="Digital Zen Garden",
                category=GameCategory.RELAXATION,
                description="Create beautiful patterns in a virtual sand garden for meditation and relaxation",
                estimated_time=10,
                difficulty="easy",
                instructions="Use your mouse or finger to draw patterns in the sand. There's no right or wrong way - just focus on the moment and create something beautiful.",
                benefits=["Promotes relaxation", "Reduces anxiety", "Enhances mindfulness", "Provides creative outlet"],
                target_mood=[MoodCategory.STRESSED, MoodCategory.ANXIOUS, MoodCategory.DEPRESSED]
            ),
            Game(
                id="math_puzzles",
                name="Quick Math Challenges",
                category=GameCategory.PROBLEM_SOLVING,
                description="Solve arithmetic problems to keep your mind sharp and engaged",
                estimated_time=4,
                difficulty="medium",
                instructions="Solve the math problems as quickly and accurately as possible. Problems start easy and get progressively harder.",
                benefits=["Improves numerical reasoning", "Enhances problem-solving speed", "Boosts confidence", "Keeps mind active"],
                target_mood=[MoodCategory.FATIGUED, MoodCategory.NEUTRAL, MoodCategory.ALERT]
            )
        ]
        
        self.games = games
        logger.info(f"Successfully seeded {len(games)} games")
        return games
    
    def create_sample_facial_analysis(self) -> ComprehensiveFacialAnalysis:
        """Create a sample ComprehensiveFacialAnalysis for testing."""
        logger.info("Creating sample facial analysis data...")
        
        # Create sample emotion distribution
        emotion_dist = EmotionDistribution(
            happy=0.3,
            sad=0.1,
            angry=0.05,
            fear=0.05,
            surprise=0.1,
            disgust=0.05,
            neutral=0.35
        )
        
        # Create sample sleepiness data
        sleepiness = SleepinessLevel(
            level="Alert",
            confidence=0.8,
            contributing_factors=["Good lighting", "Adequate rest"]
        )
        
        # Create sample fatigue indicators
        fatigue = FatigueIndicators(
            yawning_detected=False,
            head_droop_detected=False,
            overall_fatigue=False,
            confidence=0.7
        )
        
        # Create sample stress level
        stress = StressLevel(
            level="Low",
            confidence=0.75,
            indicators=["Relaxed facial muscles", "Steady breathing"]
        )
        
        # Create sample PHQ-9 estimation
        phq9 = PHQ9Estimation(
            estimated_score=2,
            confidence=0.6,
            severity_level="Minimal",
            contributing_expressions=["Neutral expression", "Occasional smile"]
        )
        
        # Create sample eye metrics
        eye_metrics = EyeMetrics(
            left_ear=0.25,
            right_ear=0.25,
            avg_ear=0.25,
            blink_rate=15.0,
            blink_duration=100.0
        )
        
        # Create sample head pose
        head_pose = HeadPoseMetrics(
            pitch=0.0,
            yaw=0.0,
            roll=0.0,
            stability=0.9
        )
        
        # Create sample micro expressions
        micro_expressions = MicroExpressionMetrics(
            muscle_tension=0.2,
            asymmetry=0.1,
            micro_movement_frequency=0.5,
            expression_variability=0.3
        )
        
        analysis = ComprehensiveFacialAnalysis(
            timestamp=datetime.utcnow(),
            face_detected=True,
            primary_emotion="neutral",
            emotion_confidence=0.8,
            emotion_distribution=emotion_dist,
            eye_metrics=eye_metrics,
            head_pose=head_pose,
            micro_expressions=micro_expressions,
            mood_assessment="Neutral",
            sleepiness=sleepiness,
            fatigue=fatigue,
            stress=stress,
            phq9_estimation=phq9,
            analysis_quality=0.85,
            frame_quality=0.9
        )
        
        logger.info("Sample facial analysis created successfully")
        return analysis
    
    def seed_all(self):
        """Seed all initial data."""
        logger.info("Starting comprehensive data seeding...")
        
        try:
            exercises = self.seed_exercises()
            games = self.seed_games()
            sample_analysis = self.create_sample_facial_analysis()
            
            logger.info("✅ All data seeded successfully!")
            logger.info(f"   📋 {len(exercises)} exercises")
            logger.info(f"   🎮 {len(games)} games")
            logger.info(f"   🔍 1 sample facial analysis")
            
            return {
                'exercises': exercises,
                'games': games,
                'sample_analysis': sample_analysis
            }
            
        except Exception as e:
            logger.error(f"❌ Error seeding data: {e}")
            raise

def main():
    """Main seeder function."""
    seeder = DataSeeder()
    seeder.seed_all()

if __name__ == "__main__":
    main()
