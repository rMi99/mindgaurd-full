"""
Unit tests for RecommendationService

Tests cover:
- Personalized recommendations generation
- Brain Heal activities retrieval
- Edge cases (invalid risk levels, empty data)
- Weekly plan generation
- Progress tracking tips
"""
import pytest
from datetime import datetime
from app.services.recommendation_service import RecommendationService


@pytest.fixture
def recommendation_service():
    """Create a RecommendationService instance for testing"""
    return RecommendationService()


@pytest.mark.unit
class TestRecommendationService:
    """Test suite for RecommendationService"""
    
    def test_initialization(self, recommendation_service):
        """Test that service initializes with correct data structures"""
        assert recommendation_service is not None
        assert hasattr(recommendation_service, 'brain_heal_activities')
        assert hasattr(recommendation_service, 'general_recommendations')
        
        # Check that all risk levels are present
        expected_levels = ['high', 'moderate', 'normal', 'low']
        assert all(level in recommendation_service.brain_heal_activities for level in expected_levels)
        assert all(level in recommendation_service.general_recommendations for level in expected_levels)
    
    def test_get_personalized_recommendations_high_risk(self, recommendation_service):
        """Test recommendations for high risk level"""
        user_data = {
            'sleep_hours': 5,
            'exercise_frequency': 1,
            'stress_level': 8
        }
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='high',
            user_data=user_data
        )
        
        assert result['risk_level'] == 'high'
        assert 'general_recommendations' in result
        assert 'personalized_insights' in result
        assert 'weekly_plan' in result
        assert 'progress_tracking' in result
        assert 'timestamp' in result
        assert len(result['general_recommendations']) > 0
        assert len(result['personalized_insights']) > 0
    
    def test_get_personalized_recommendations_moderate_risk(self, recommendation_service):
        """Test recommendations for moderate risk level"""
        user_data = {
            'sleep_hours': 7,
            'exercise_frequency': 2,
            'stress_level': 5
        }
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        assert result['risk_level'] == 'moderate'
        assert len(result['general_recommendations']) > 0
        assert 'weekly_plan' in result
    
    def test_get_personalized_recommendations_normal_risk(self, recommendation_service):
        """Test recommendations for normal risk level"""
        user_data = {
            'sleep_hours': 8,
            'exercise_frequency': 4,
            'stress_level': 3
        }
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='normal',
            user_data=user_data
        )
        
        assert result['risk_level'] == 'normal'
        assert len(result['general_recommendations']) > 0
    
    def test_get_personalized_recommendations_low_risk(self, recommendation_service):
        """Test recommendations for low risk level"""
        user_data = {
            'sleep_hours': 8,
            'exercise_frequency': 5,
            'stress_level': 2
        }
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='low',
            user_data=user_data
        )
        
        assert result['risk_level'] == 'low'
        assert len(result['general_recommendations']) > 0
    
    def test_get_personalized_recommendations_case_insensitive(self, recommendation_service):
        """Test that risk level is case-insensitive"""
        user_data = {}
        
        result_upper = recommendation_service.get_personalized_recommendations(
            risk_level='HIGH',
            user_data=user_data
        )
        result_lower = recommendation_service.get_personalized_recommendations(
            risk_level='high',
            user_data=user_data
        )
        
        assert result_upper['risk_level'] == 'high'
        assert result_lower['risk_level'] == 'high'
    
    def test_get_personalized_recommendations_with_brain_heal(self, recommendation_service):
        """Test recommendations with Brain Heal activities included"""
        user_data = {}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data,
            include_brain_heal=True
        )
        
        assert 'brain_heal_activities' in result
        assert len(result['brain_heal_activities']) > 0
        assert all('name' in activity for activity in result['brain_heal_activities'])
        assert all('duration' in activity for activity in result['brain_heal_activities'])
    
    def test_get_personalized_recommendations_without_brain_heal(self, recommendation_service):
        """Test recommendations without Brain Heal activities"""
        user_data = {}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data,
            include_brain_heal=False
        )
        
        assert 'brain_heal_activities' not in result
    
    def test_invalid_risk_level_raises_error(self, recommendation_service):
        """Test that invalid risk level raises ValueError"""
        user_data = {}
        
        with pytest.raises(ValueError, match="Invalid risk level"):
            recommendation_service.get_personalized_recommendations(
                risk_level='invalid_level',
                user_data=user_data
            )
    
    def test_empty_user_data(self, recommendation_service):
        """Test recommendations with empty user data"""
        result = recommendation_service.get_personalized_recommendations(
            risk_level='normal',
            user_data={}
        )
        
        assert result['risk_level'] == 'normal'
        assert 'personalized_insights' in result
        # Should still work, just with fewer personalized insights
    
    def test_personalized_insights_sleep_low(self, recommendation_service):
        """Test personalized insights for low sleep hours"""
        user_data = {'sleep_hours': 5}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        insights = result['personalized_insights']
        assert any('sleep' in insight.lower() for insight in insights)
        assert any('below' in insight.lower() or 'recommended' in insight.lower() for insight in insights)
    
    def test_personalized_insights_sleep_high(self, recommendation_service):
        """Test personalized insights for high sleep hours"""
        user_data = {'sleep_hours': 10}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        insights = result['personalized_insights']
        assert any('sleep' in insight.lower() for insight in insights)
    
    def test_personalized_insights_exercise_low(self, recommendation_service):
        """Test personalized insights for low exercise frequency"""
        user_data = {'exercise_frequency': 1}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        insights = result['personalized_insights']
        assert any('exercise' in insight.lower() for insight in insights)
    
    def test_personalized_insights_stress_high(self, recommendation_service):
        """Test personalized insights for high stress level"""
        user_data = {'stress_level': 9}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        insights = result['personalized_insights']
        assert any('stress' in insight.lower() for insight in insights)
    
    def test_weekly_plan_structure(self, recommendation_service):
        """Test that weekly plan has correct structure"""
        user_data = {}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        weekly_plan = result['weekly_plan']
        days = ['monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday']
        assert all(day in weekly_plan for day in days)
        assert all(isinstance(weekly_plan[day], list) for day in days)
    
    def test_progress_tracking_tips(self, recommendation_service):
        """Test progress tracking tips are included"""
        user_data = {}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='high',
            user_data=user_data
        )
        
        assert 'progress_tracking' in result
        assert isinstance(result['progress_tracking'], list)
        assert len(result['progress_tracking']) > 0
    
    def test_get_brain_heal_activities(self, recommendation_service):
        """Test getting Brain Heal activities for a risk level"""
        activities = recommendation_service._get_brain_heal_activities('moderate')
        
        assert isinstance(activities, list)
        assert len(activities) > 0
        assert all('name' in activity for activity in activities)
        assert all('duration' in activity for activity in activities)
        assert all('description' in activity for activity in activities)
        assert all('steps' in activity for activity in activities)
        assert all('benefits' in activity for activity in activities)
    
    def test_get_brain_heal_activities_invalid_level(self, recommendation_service):
        """Test getting Brain Heal activities for invalid risk level"""
        activities = recommendation_service._get_brain_heal_activities('invalid')
        
        # Should return empty list for invalid level
        assert isinstance(activities, list)
        assert len(activities) == 0
    
    def test_get_activity_by_name_exists(self, recommendation_service):
        """Test getting activity by name when it exists"""
        activity = recommendation_service.get_activity_by_name('Deep Breathing Exercise')
        
        assert activity is not None
        assert activity['name'] == 'Deep Breathing Exercise'
        assert 'duration' in activity
        assert 'description' in activity
    
    def test_get_activity_by_name_not_exists(self, recommendation_service):
        """Test getting activity by name when it doesn't exist"""
        activity = recommendation_service.get_activity_by_name('Non-existent Activity')
        
        assert activity is None
    
    def test_get_activity_by_name_case_insensitive(self, recommendation_service):
        """Test that activity name search is case-insensitive"""
        activity_lower = recommendation_service.get_activity_by_name('deep breathing exercise')
        activity_upper = recommendation_service.get_activity_by_name('DEEP BREATHING EXERCISE')
        
        assert activity_lower is not None
        assert activity_upper is not None
        assert activity_lower['name'] == activity_upper['name']
    
    def test_get_random_activity_with_risk_level(self, recommendation_service):
        """Test getting random activity with specific risk level"""
        activity = recommendation_service.get_random_activity(risk_level='moderate')
        
        assert activity is not None
        assert 'name' in activity
        assert 'duration' in activity
    
    def test_get_random_activity_without_risk_level(self, recommendation_service):
        """Test getting random activity without risk level filter"""
        activity = recommendation_service.get_random_activity()
        
        assert activity is not None
        assert 'name' in activity
        assert 'duration' in activity
    
    def test_get_weekly_challenge(self, recommendation_service):
        """Test getting weekly challenge"""
        challenge = recommendation_service.get_weekly_challenge('high')
        
        assert 'title' in challenge
        assert 'description' in challenge
        assert 'daily_challenges' in challenge
        assert 'goal' in challenge
        assert isinstance(challenge['daily_challenges'], list)
        assert len(challenge['daily_challenges']) == 7
    
    def test_get_weekly_challenge_invalid_level(self, recommendation_service):
        """Test getting weekly challenge for invalid level defaults to normal"""
        challenge = recommendation_service.get_weekly_challenge('invalid')
        
        # Should default to 'normal' challenge
        assert 'title' in challenge
        assert challenge['title'] == 'Balanced Wellness Week'
    
    def test_timestamp_in_recommendations(self, recommendation_service):
        """Test that recommendations include timestamp"""
        user_data = {}
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='normal',
            user_data=user_data
        )
        
        assert 'timestamp' in result
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(result['timestamp'])


@pytest.mark.unit
class TestRecommendationServiceEdgeCases:
    """Edge case tests for RecommendationService"""
    
    def test_very_large_user_data(self, recommendation_service):
        """Test with very large user data dictionary"""
        large_data = {f'key_{i}': f'value_{i}' for i in range(1000)}
        large_data['sleep_hours'] = 7
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='normal',
            user_data=large_data
        )
        
        assert result['risk_level'] == 'normal'
    
    def test_none_values_in_user_data(self, recommendation_service):
        """Test with None values in user data"""
        user_data = {
            'sleep_hours': None,
            'exercise_frequency': None,
            'stress_level': None
        }
        
        # Should not crash, just skip None values
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        assert result['risk_level'] == 'moderate'
    
    def test_negative_values_in_user_data(self, recommendation_service):
        """Test with negative values in user data"""
        user_data = {
            'sleep_hours': -5,
            'exercise_frequency': -1,
            'stress_level': -2
        }
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        # Should handle gracefully
        assert result['risk_level'] == 'moderate'
    
    def test_extremely_high_values_in_user_data(self, recommendation_service):
        """Test with extremely high values in user data"""
        user_data = {
            'sleep_hours': 24,
            'exercise_frequency': 100,
            'stress_level': 100
        }
        
        result = recommendation_service.get_personalized_recommendations(
            risk_level='moderate',
            user_data=user_data
        )
        
        assert result['risk_level'] == 'moderate'

