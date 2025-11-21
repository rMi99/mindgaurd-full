from typing import Dict, List, Optional
import logging
from datetime import datetime, timedelta
import random

logger = logging.getLogger(__name__)

class RecommendationService:
    """
    Service for providing personalized health recommendations and Brain Heal activities
    based on user's health risk assessment.
    """
    
    def __init__(self):
        self.brain_heal_activities = {
            'high': [
                {
                    'name': 'Deep Breathing Exercise',
                    'duration': '10 minutes',
                    'description': 'Practice diaphragmatic breathing to reduce stress and anxiety',
                    'steps': [
                        'Find a comfortable seated position',
                        'Place one hand on your chest and one on your belly',
                        'Breathe in slowly through your nose for 4 counts',
                        'Hold for 4 counts',
                        'Exhale slowly through your mouth for 6 counts',
                        'Repeat for 10 minutes'
                    ],
                    'benefits': ['Reduces cortisol levels', 'Improves focus', 'Calms nervous system'],
                    'difficulty': 'Beginner'
                },
                {
                    'name': 'Therapeutic Journaling',
                    'duration': '15-20 minutes',
                    'description': 'Write about your thoughts and feelings to process emotions',
                    'steps': [
                        'Set aside quiet time without distractions',
                        'Write freely without worrying about grammar or structure',
                        'Focus on your current emotional state',
                        'Explore what might be causing stress or anxiety',
                        'End with positive affirmations or gratitude'
                    ],
                    'benefits': ['Emotional processing', 'Stress reduction', 'Self-awareness'],
                    'difficulty': 'Beginner'
                },
                {
                    'name': 'Professional Support Consultation',
                    'duration': 'Varies',
                    'description': 'Consider reaching out to mental health professionals',
                    'steps': [
                        'Research local mental health resources',
                        'Contact your healthcare provider for referrals',
                        'Look into online therapy options',
                        'Schedule an initial consultation'
                    ],
                    'benefits': ['Professional guidance', 'Personalized treatment', 'Long-term support'],
                    'difficulty': 'Professional'
                }
            ],
            'moderate': [
                {
                    'name': 'Progressive Muscle Relaxation',
                    'duration': '15-20 minutes',
                    'description': 'Systematically tense and relax different muscle groups',
                    'steps': [
                        'Lie down or sit comfortably in a quiet space',
                        'Start with your feet - tense for 5 seconds, then relax',
                        'Move up through your legs, torso, arms, and face',
                        'Focus on the contrast between tension and relaxation',
                        'End with a few minutes of deep breathing',
                        'Notice the overall sense of calm in your body'
                    ],
                    'benefits': ['Physical tension relief', 'Stress reduction', 'Better sleep'],
                    'difficulty': 'Beginner'
                },
                {
                    'name': 'Mindful Walking Meditation',
                    'duration': '20-30 minutes',
                    'description': 'Combine gentle exercise with mindfulness practice',
                    'steps': [
                        'Choose a quiet path or route in nature if possible',
                        'Walk at a slower pace than usual',
                        'Focus on the sensation of your feet touching the ground',
                        'Notice your breathing, the air, sounds, and sights around you',
                        'When your mind wanders, gently bring attention back to walking',
                        'End with a few minutes of gratitude'
                    ],
                    'benefits': ['Physical activity', 'Mental clarity', 'Stress relief'],
                    'difficulty': 'Beginner'
                },
                {
                    'name': 'Cognitive Restructuring Practice',
                    'duration': '15-20 minutes',
                    'description': 'Challenge and reframe negative thought patterns',
                    'steps': [
                        'Identify a stressful thought or worry',
                        'Write down the thought and rate your belief in it (1-10)',
                        'Look for evidence that supports and contradicts the thought',
                        'Generate a more balanced, realistic alternative thought',
                        'Rate your belief in the new thought',
                        'Notice how your emotions change with the new perspective'
                    ],
                    'benefits': ['Improved emotional regulation', 'Reduced anxiety', 'Better perspective'],
                    'difficulty': 'Intermediate'
                },
                {
                    'name': 'Structured Social Connection',
                    'duration': '30-60 minutes',
                    'description': 'Intentionally connect with supportive people in your life',
                    'steps': [
                        'Identify 2-3 people who make you feel supported and understood',
                        'Reach out to one of them via call, text, or in-person meeting',
                        'Share something meaningful about your current experience',
                        'Listen actively to their response and perspective',
                        'Express gratitude for their support',
                        'Schedule regular check-ins if helpful'
                    ],
                    'benefits': ['Social support', 'Reduced isolation', 'Emotional validation'],
                    'difficulty': 'Beginner'
                }
            ],
            'normal': [
                {
                    'name': 'Mindful Walking',
                    'duration': '30 minutes',
                    'description': 'Take a walk while practicing mindfulness and awareness',
                    'steps': [
                        'Choose a safe walking route',
                        'Walk at a comfortable pace',
                        'Focus on your breathing and footsteps',
                        'Notice the sights, sounds, and sensations around you',
                        'If your mind wanders, gently return to the present moment'
                    ],
                    'benefits': ['Physical exercise', 'Mental clarity', 'Stress reduction'],
                    'difficulty': 'Beginner'
                },
                {
                    'name': 'Calming Music Therapy',
                    'duration': '20-30 minutes',
                    'description': 'Listen to soothing music to relax and unwind',
                    'steps': [
                        'Create a playlist of calming music',
                        'Find a quiet, comfortable space',
                        'Close your eyes and focus on the music',
                        'Practice deep breathing while listening',
                        'Allow yourself to fully relax'
                    ],
                    'benefits': ['Mood improvement', 'Stress reduction', 'Better sleep'],
                    'difficulty': 'Beginner'
                },
                {
                    'name': 'Social Connection Time',
                    'duration': '30-60 minutes',
                    'description': 'Reach out to friends, family, or loved ones',
                    'steps': [
                        'Identify people you want to connect with',
                        'Choose your preferred method (call, video chat, meet in person)',
                        'Share your feelings and listen to theirs',
                        'Engage in activities you both enjoy',
                        'Express gratitude for the connection'
                    ],
                    'benefits': ['Social support', 'Emotional well-being', 'Reduced isolation'],
                    'difficulty': 'Beginner'
                }
            ],
            'low': [
                {
                    'name': 'New Hobby Exploration',
                    'duration': '1-2 hours',
                    'description': 'Try something new that interests you',
                    'steps': [
                        'Make a list of activities you\'ve always wanted to try',
                        'Choose one that feels exciting and manageable',
                        'Gather necessary materials or resources',
                        'Set aside dedicated time to practice',
                        'Be patient with yourself as you learn'
                    ],
                    'benefits': ['Mental stimulation', 'Joy and fulfillment', 'Personal growth'],
                    'difficulty': 'Beginner to Intermediate'
                },
                {
                    'name': 'Mindfulness Meditation',
                    'duration': '5-10 minutes',
                    'description': 'Practice present-moment awareness and acceptance',
                    'steps': [
                        'Find a quiet, comfortable space',
                        'Sit with your back straight but relaxed',
                        'Close your eyes or focus on a point',
                        'Focus on your breath or a mantra',
                        'When thoughts arise, observe them without judgment',
                        'Gently return to your focus point'
                    ],
                    'benefits': ['Reduced stress', 'Improved focus', 'Emotional balance'],
                    'difficulty': 'Beginner'
                },
                {
                    'name': 'Daily Micro-Goal Setting',
                    'duration': '5 minutes daily',
                    'description': 'Set and achieve small, meaningful daily goals',
                    'steps': [
                        'Reflect on what would make today meaningful',
                        'Choose 1-3 small, achievable goals',
                        'Write them down in a visible place',
                        'Break them into specific, actionable steps',
                        'Celebrate when you complete them'
                    ],
                    'benefits': ['Sense of accomplishment', 'Motivation', 'Progress tracking'],
                    'difficulty': 'Beginner'
                }
            ]
        }
        
        self.general_recommendations = {
            'high': [
                'Prioritize sleep hygiene - aim for 7-9 hours of quality sleep',
                'Consider reducing caffeine and alcohol intake',
                'Practice stress management techniques daily',
                'Maintain regular meal times and balanced nutrition',
                'Limit screen time, especially before bedtime',
                'Seek professional help if symptoms persist'
            ],
            'moderate': [
                'Implement stress management techniques like meditation or deep breathing',
                'Establish a consistent sleep routine and aim for 7-8 hours nightly',
                'Incorporate regular physical activity into your weekly schedule',
                'Practice mindfulness and present-moment awareness',
                'Consider talking to a trusted friend or counselor about your feelings',
                'Limit overwhelming activities and create time for relaxation'
            ],
            'normal': [
                'Maintain consistent sleep schedule',
                'Include moderate exercise in your routine',
                'Practice stress management techniques',
                'Maintain balanced diet with regular meals',
                'Stay connected with friends and family',
                'Monitor stress levels and adjust activities accordingly'
            ],
            'low': [
                'Continue maintaining healthy habits',
                'Consider challenging yourself with new activities',
                'Share your positive habits with others',
                'Regular health check-ups and preventive care',
                'Continue learning and personal development',
                'Maintain work-life balance'
            ]
        }
    
    def get_personalized_recommendations(self, 
                                      risk_level: str, 
                                      user_data: Dict,
                                      include_brain_heal: bool = False) -> Dict:
        """
        Generate personalized recommendations based on risk level and user data.
        
        Args:
            risk_level: User's mental health risk level ('low', 'normal', 'moderate', 'high')
            user_data: Dictionary containing user's health and demographic data
            include_brain_heal: Whether to include brain healing activities
            
        Returns:
            Dictionary containing personalized recommendations
        """
        try:
            risk_level = risk_level.lower()
            
            # Define valid risk levels
            valid_risk_levels = ['low', 'normal', 'moderate', 'high']
            
            if risk_level not in valid_risk_levels:
                raise ValueError(f"Invalid risk level: {risk_level}. Valid levels are: {valid_risk_levels}")
            
            # Build recommendations using the general recommendations
            recommendations = {
                'risk_level': risk_level,
                'timestamp': datetime.now().isoformat(),
                'general_recommendations': self.general_recommendations[risk_level],
                'personalized_insights': self._generate_personalized_insights(user_data, risk_level),
                'weekly_plan': self._create_weekly_plan(risk_level),
                'progress_tracking': self._get_progress_tracking_tips(risk_level)
            }
            
            if include_brain_heal:
                recommendations['brain_heal_activities'] = self._get_brain_heal_activities(risk_level)
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            raise
    
    def _generate_personalized_insights(self, user_data: Dict, risk_level: str) -> List[str]:
        """
        Generate personalized insights based on user's specific health data.
        """
        insights = []
        
        # Sleep insights
        if 'sleep_hours' in user_data:
            sleep_hours = user_data['sleep_hours']
            if sleep_hours is None:
                pass
            elif sleep_hours < 6:
                insights.append(f"Your sleep duration of {sleep_hours} hours is below recommended levels. Consider improving sleep hygiene.")
            elif sleep_hours > 9:
                insights.append(f"Your sleep duration of {sleep_hours} hours is above average. Ensure quality sleep over quantity.")
        
        # Exercise insights
        if 'exercise_frequency' in user_data:
            exercise_freq = user_data['exercise_frequency']
            if exercise_freq is None:
                pass
            elif exercise_freq < 3:
                insights.append("Increasing your exercise frequency could significantly improve your health outcomes.")
            elif exercise_freq >= 5:
                insights.append("Great job maintaining regular exercise! Consider adding variety to your routine.")
        
        # Stress insights
        if 'stress_level' in user_data:
            stress_level = user_data['stress_level']
            if stress_level is None:
                pass
            elif stress_level > 7:
                insights.append("Your stress levels are elevated. Consider implementing daily stress management techniques.")
            elif stress_level < 3:
                insights.append("You're managing stress well. Continue these positive practices.")
        
        # Social connections
        if 'social_connections' in user_data:
            social_score = user_data['social_connections']
            if social_score is None:
                pass
            elif social_score < 5:
                insights.append("Strengthening social connections could improve your mental well-being.")
            elif social_score > 8:
                insights.append("Strong social connections are a great foundation for mental health.")
        
        return insights
    
    def _create_weekly_plan(self, risk_level: str) -> Dict:
        """
        Create a weekly wellness plan based on risk level.
        """
        weekly_plans = {
            'high': {
                'monday': ['Deep breathing (10 min)', 'Light stretching', 'Journaling'],
                'tuesday': ['Mindful walking (15 min)', 'Gratitude practice', 'Reduce screen time'],
                'wednesday': ['Progressive muscle relaxation', 'Healthy meal prep', 'Early bedtime'],
                'thursday': ['Gentle yoga', 'Social connection call', 'Stress journaling'],
                'friday': ['Nature walk', 'Creative activity', 'Weekend planning'],
                'saturday': ['Rest day', 'Light hobby time', 'Family/friend time'],
                'sunday': ['Mindfulness meditation', 'Weekly reflection', 'Prepare for week ahead']
            },
            'moderate': {
                'monday': ['Morning meditation (10 min)', 'Moderate exercise (20 min)', 'Stress check-in'],
                'tuesday': ['Mindful breathing', 'Social connection', 'Healthy meal planning'],
                'wednesday': ['Gentle yoga or stretching', 'Journaling practice', 'Evening relaxation'],
                'thursday': ['Walking or light exercise', 'Creative activity', 'Stress management'],
                'friday': ['Mindfulness practice', 'Weekend planning', 'Social activity'],
                'saturday': ['Moderate activity', 'Rest and relaxation', 'Quality time with others'],
                'sunday': ['Reflection and gratitude', 'Meal prep', 'Week preparation']
            },
            'normal': {
                'monday': ['Morning walk (20 min)', 'Healthy breakfast', 'Work organization'],
                'tuesday': ['Exercise session (30 min)', 'Social connection', 'Stress management'],
                'wednesday': ['Mindfulness practice', 'Balanced meals', 'Hobby time'],
                'thursday': ['Cardio workout', 'Learning activity', 'Evening relaxation'],
                'friday': ['Strength training', 'Weekend planning', 'Social activity'],
                'saturday': ['Active recovery', 'Creative pursuit', 'Quality time with others'],
                'sunday': ['Rest and reflection', 'Meal prep', 'Week preparation']
            },
            'low': {
                'monday': ['Morning routine', 'Goal setting', 'Productive work'],
                'tuesday': ['Exercise variety', 'Learning new skill', 'Social engagement'],
                'wednesday': ['Wellness check-in', 'Creative activity', 'Personal development'],
                'thursday': ['Fitness challenge', 'Community involvement', 'Skill building'],
                'friday': ['Weekend planning', 'Achievement celebration', 'Social connections'],
                'saturday': ['Adventure/exploration', 'Hobby development', 'Relationship building'],
                'sunday': ['Rest and recharge', 'Weekly planning', 'Goal review']
            }
        }
        
        return weekly_plans.get(risk_level, weekly_plans['normal'])
    
    def _get_progress_tracking_tips(self, risk_level: str) -> List[str]:
        """
        Get tips for tracking progress based on risk level.
        """
        tracking_tips = {
            'high': [
                'Track daily mood and stress levels',
                'Monitor sleep quality and duration',
                'Record stress management activities',
                'Note any physical symptoms',
                'Track professional support sessions',
                'Celebrate small improvements'
            ],
            'moderate': [
                'Track mood patterns and stress levels 3-4 times per week',
                'Monitor sleep quality and consistency',
                'Record stress management and relaxation activities',
                'Note physical and emotional changes',
                'Track social connections and support',
                'Record mindfulness and meditation practice',
                'Celebrate progress and positive changes'
            ],
            'normal': [
                'Weekly wellness check-ins',
                'Track exercise and activity levels',
                'Monitor stress management effectiveness',
                'Record social connection activities',
                'Track sleep patterns',
                'Note positive changes'
            ],
            'low': [
                'Monthly wellness assessments',
                'Track personal development goals',
                'Monitor habit consistency',
                'Record new experiences and learning',
                'Track overall life satisfaction',
                'Set and review quarterly goals'
            ]
        }
        
        return tracking_tips.get(risk_level, tracking_tips['normal'])
    
    def _get_brain_heal_activities(self, risk_level: str) -> List[Dict]:
        """
        Get Brain Heal activities for the specified risk level.
        """
        activities = self.brain_heal_activities.get(risk_level, [])
        
        # Randomize the order for variety
        randomized_activities = activities.copy()
        random.shuffle(randomized_activities)
        
        return randomized_activities
    
    def get_activity_by_name(self, activity_name: str) -> Optional[Dict]:
        """
        Get a specific Brain Heal activity by name.
        
        Args:
            activity_name: Name of the activity to retrieve
            
        Returns:
            Activity details or None if not found
        """
        for risk_level in self.brain_heal_activities:
            for activity in self.brain_heal_activities[risk_level]:
                if activity['name'].lower() == activity_name.lower():
                    return activity
        return None
    
    def get_random_activity(self, risk_level: Optional[str] = None) -> Dict:
        """
        Get a random Brain Heal activity.
        
        Args:
            risk_level: Optional risk level filter
            
        Returns:
            Random activity details
        """
        if risk_level and risk_level.lower() in self.brain_heal_activities:
            activities = self.brain_heal_activities[risk_level.lower()]
        else:
            # Flatten all activities if no specific risk level
            activities = []
            for risk_activities in self.brain_heal_activities.values():
                activities.extend(risk_activities)
        
        return random.choice(activities)
    
    def get_weekly_challenge(self, risk_level: str) -> Dict:
        """
        Generate a weekly wellness challenge based on risk level.
        
        Args:
            risk_level: User's health risk level
            
        Returns:
            Weekly challenge details
        """
        challenges = {
            'high': {
                'title': 'Mindful Recovery Week',
                'description': 'Focus on gentle, healing activities that support your recovery',
                'daily_challenges': [
                    'Day 1: Practice 10 minutes of deep breathing',
                    'Day 2: Write 3 things you\'re grateful for',
                    'Day 3: Take a 15-minute mindful walk',
                    'Day 4: Try progressive muscle relaxation',
                    'Day 5: Connect with a supportive person',
                    'Day 6: Practice self-compassion meditation',
                    'Day 7: Reflect on your progress and plan next steps'
                ],
                'goal': 'Build a foundation of daily wellness practices'
            },
            'normal': {
                'title': 'Balanced Wellness Week',
                'description': 'Maintain and enhance your current wellness practices',
                'daily_challenges': [
                    'Day 1: Try a new form of exercise',
                    'Day 2: Practice active listening in conversations',
                    'Day 3: Learn a new stress management technique',
                    'Day 4: Cook a healthy meal from scratch',
                    'Day 5: Engage in a creative activity',
                    'Day 6: Plan a social activity with friends',
                    'Day 7: Set goals for the upcoming week'
                ],
                'goal': 'Enhance your wellness routine with new experiences'
            },
            'low': {
                'title': 'Thriving and Growing Week',
                'description': 'Push your boundaries and explore new wellness frontiers',
                'daily_challenges': [
                    'Day 1: Try an advanced fitness challenge',
                    'Day 2: Learn a completely new skill',
                    'Day 3: Mentor someone in wellness',
                    'Day 4: Participate in a community wellness event',
                    'Day 5: Create a wellness challenge for others',
                    'Day 6: Explore a new wellness modality',
                    'Day 7: Plan a wellness retreat or intensive'
                ],
                'goal': 'Expand your wellness impact and personal growth'
            }
        }
        
        return challenges.get(risk_level.lower(), challenges['normal']) 