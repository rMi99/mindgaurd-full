"""
PHQ-9 Integration Service for auto-filling depression screening questionnaires
based on facial analysis results.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from app.models.facial_metrics import ComprehensiveFacialAnalysis, PHQ9Estimation

logger = logging.getLogger(__name__)

class PHQ9IntegrationService:
    """Service for integrating facial analysis with PHQ-9 questionnaire auto-fill."""
    
    def __init__(self):
        # PHQ-9 question mappings to facial indicators
        self.question_mappings = {
            "q1_interest": {
                "facial_indicators": ["sad", "neutral", "angry"],
                "sleepiness_weight": 0.3,
                "stress_weight": 0.2,
                "fatigue_weight": 0.4
            },
            "q2_mood": {
                "facial_indicators": ["sad", "angry", "fear"],
                "sleepiness_weight": 0.1,
                "stress_weight": 0.5,
                "fatigue_weight": 0.2
            },
            "q3_sleep": {
                "facial_indicators": ["sad", "neutral"],
                "sleepiness_weight": 0.8,
                "stress_weight": 0.1,
                "fatigue_weight": 0.6
            },
            "q4_energy": {
                "facial_indicators": ["sad", "neutral"],
                "sleepiness_weight": 0.4,
                "stress_weight": 0.2,
                "fatigue_weight": 0.7
            },
            "q5_appetite": {
                "facial_indicators": ["sad", "disgust"],
                "sleepiness_weight": 0.1,
                "stress_weight": 0.3,
                "fatigue_weight": 0.2
            },
            "q6_self_worth": {
                "facial_indicators": ["sad", "fear", "angry"],
                "sleepiness_weight": 0.1,
                "stress_weight": 0.6,
                "fatigue_weight": 0.1
            },
            "q7_concentration": {
                "facial_indicators": ["sad", "neutral"],
                "sleepiness_weight": 0.5,
                "stress_weight": 0.4,
                "fatigue_weight": 0.6
            },
            "q8_psychomotor": {
                "facial_indicators": ["sad", "neutral"],
                "sleepiness_weight": 0.3,
                "stress_weight": 0.2,
                "fatigue_weight": 0.8
            },
            "q9_suicidal": {
                "facial_indicators": ["sad", "fear", "angry"],
                "sleepiness_weight": 0.1,
                "stress_weight": 0.8,
                "fatigue_weight": 0.2
            }
        }
        
        # Emotion to PHQ-9 score weights
        self.emotion_weights = {
            "happy": -0.5,
            "neutral": 0.0,
            "sad": 2.0,
            "angry": 1.5,
            "fear": 1.8,
            "surprise": 0.2,
            "disgust": 1.2
        }
    
    def generate_phq9_auto_fill(self, facial_analysis: ComprehensiveFacialAnalysis) -> Dict[str, Any]:
        """
        Generate auto-filled PHQ-9 responses based on facial analysis.
        
        Args:
            facial_analysis: Comprehensive facial analysis result
            
        Returns:
            Dictionary with auto-filled PHQ-9 responses and confidence scores
        """
        try:
            auto_fill_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "source": "facial_analysis",
                "confidence": facial_analysis.phq9_estimation.confidence,
                "estimated_total_score": facial_analysis.phq9_estimation.estimated_score,
                "severity_level": facial_analysis.phq9_estimation.severity_level,
                "responses": {},
                "reasoning": {},
                "recommendations": []
            }
            
            # Calculate individual question scores
            for question_id, mapping in self.question_mappings.items():
                score, reasoning = self._calculate_question_score(
                    facial_analysis, question_id, mapping
                )
                
                auto_fill_data["responses"][question_id] = {
                    "score": score,
                    "confidence": self._calculate_question_confidence(
                        facial_analysis, question_id, mapping
                    )
                }
                
                auto_fill_data["reasoning"][question_id] = reasoning
            
            # Generate recommendations based on analysis
            auto_fill_data["recommendations"] = self._generate_recommendations(facial_analysis)
            
            return auto_fill_data
            
        except Exception as e:
            logger.error(f"Error generating PHQ-9 auto-fill: {e}")
            return self._create_fallback_response()
    
    def _calculate_question_score(self, analysis: ComprehensiveFacialAnalysis, 
                                question_id: str, mapping: Dict) -> tuple[int, str]:
        """Calculate PHQ-9 score for a specific question based on facial analysis."""
        try:
            base_score = 0
            reasoning_parts = []
            
            # Check emotion indicators
            emotion_dist = analysis.emotion_distribution
            for emotion in mapping["facial_indicators"]:
                emotion_value = getattr(emotion_dist, emotion, 0)
                weight = self.emotion_weights.get(emotion, 0)
                contribution = emotion_value * weight
                base_score += contribution
                
                if emotion_value > 0.3:
                    reasoning_parts.append(f"High {emotion} expression ({emotion_value:.2f})")
            
            # Check sleepiness indicators
            sleepiness_weight = mapping["sleepiness_weight"]
            if analysis.sleepiness.level == "Very tired":
                base_score += sleepiness_weight * 2
                reasoning_parts.append("Severe sleepiness detected")
            elif analysis.sleepiness.level == "Slightly tired":
                base_score += sleepiness_weight * 1
                reasoning_parts.append("Mild sleepiness detected")
            
            # Check stress indicators
            stress_weight = mapping["stress_weight"]
            if analysis.stress.level == "High":
                base_score += stress_weight * 2
                reasoning_parts.append("High stress levels detected")
            elif analysis.stress.level == "Medium":
                base_score += stress_weight * 1
                reasoning_parts.append("Moderate stress detected")
            
            # Check fatigue indicators
            fatigue_weight = mapping["fatigue_weight"]
            if analysis.fatigue.overall_fatigue:
                base_score += fatigue_weight * 1.5
                reasoning_parts.append("Fatigue signs detected")
            
            # Normalize to PHQ-9 scale (0-3)
            normalized_score = max(0, min(3, int(base_score)))
            
            reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No significant indicators"
            
            return normalized_score, reasoning
            
        except Exception as e:
            logger.error(f"Error calculating question score for {question_id}: {e}")
            return 0, "Analysis error"
    
    def _calculate_question_confidence(self, analysis: ComprehensiveFacialAnalysis, 
                                    question_id: str, mapping: Dict) -> float:
        """Calculate confidence score for a specific question."""
        try:
            # Base confidence from analysis quality
            base_confidence = analysis.analysis_quality
            
            # Adjust based on relevant indicators
            confidence_factors = []
            
            # Check if relevant emotions are detected
            emotion_dist = analysis.emotion_distribution
            relevant_emotions = [getattr(emotion_dist, emotion, 0) 
                               for emotion in mapping["facial_indicators"]]
            max_emotion = max(relevant_emotions) if relevant_emotions else 0
            confidence_factors.append(max_emotion)
            
            # Check sleepiness relevance
            if mapping["sleepiness_weight"] > 0.3:
                sleepiness_confidence = 1.0 if analysis.sleepiness.level != "Unknown" else 0.5
                confidence_factors.append(sleepiness_confidence)
            
            # Check stress relevance
            if mapping["stress_weight"] > 0.3:
                stress_confidence = 1.0 if analysis.stress.level != "Unknown" else 0.5
                confidence_factors.append(stress_confidence)
            
            # Check fatigue relevance
            if mapping["fatigue_weight"] > 0.3:
                fatigue_confidence = 1.0 if analysis.fatigue.overall_fatigue else 0.7
                confidence_factors.append(fatigue_confidence)
            
            # Calculate weighted confidence
            if confidence_factors:
                adjusted_confidence = base_confidence * (0.5 + 0.5 * (sum(confidence_factors) / len(confidence_factors)))
            else:
                adjusted_confidence = base_confidence * 0.7
            
            return min(1.0, max(0.0, adjusted_confidence))
            
        except Exception as e:
            logger.error(f"Error calculating confidence for {question_id}: {e}")
            return 0.5
    
    def _generate_recommendations(self, analysis: ComprehensiveFacialAnalysis) -> List[str]:
        """Generate recommendations based on facial analysis."""
        recommendations = []
        
        try:
            # High PHQ-9 score recommendations
            if analysis.phq9_estimation.estimated_score >= 15:
                recommendations.append(
                    "Your analysis suggests significant mental health concerns. "
                    "Please consider speaking with a healthcare provider or mental health professional."
                )
            elif analysis.phq9_estimation.estimated_score >= 10:
                recommendations.append(
                    "Consider self-care practices and monitoring your mental health regularly. "
                    "You may benefit from speaking with a counselor or therapist."
                )
            
            # Sleep-related recommendations
            if analysis.sleepiness.level == "Very tired":
                recommendations.append(
                    "Severe fatigue detected. Consider improving sleep hygiene, "
                    "reducing screen time before bed, and establishing a consistent sleep schedule."
                )
            elif analysis.sleepiness.level == "Slightly tired":
                recommendations.append(
                    "Mild fatigue detected. Ensure you're getting adequate rest and taking breaks."
                )
            
            # Stress-related recommendations
            if analysis.stress.level == "High":
                recommendations.append(
                    "High stress levels detected. Try relaxation techniques like deep breathing, "
                    "meditation, or gentle exercise. Consider stress management strategies."
                )
            elif analysis.stress.level == "Medium":
                recommendations.append(
                    "Moderate stress detected. Consider incorporating stress-reduction activities "
                    "into your daily routine."
                )
            
            # Mood-related recommendations
            if analysis.primary_emotion in ["sad", "angry", "fear"]:
                recommendations.append(
                    "Negative emotions detected. Consider engaging in activities that bring you joy, "
                    "connecting with supportive people, or practicing mindfulness."
                )
            
            # General wellness recommendations
            if analysis.fatigue.overall_fatigue:
                recommendations.append(
                    "Fatigue signs detected. Focus on adequate rest, nutrition, and gentle physical activity."
                )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return ["Consider consulting with a healthcare provider for personalized guidance."]
    
    def _create_fallback_response(self) -> Dict[str, Any]:
        """Create fallback response when analysis fails."""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "source": "facial_analysis",
            "confidence": 0.0,
            "estimated_total_score": 0,
            "severity_level": "Unknown",
            "responses": {},
            "reasoning": {},
            "recommendations": [
                "Unable to analyze facial expressions. Please ensure good lighting and camera positioning.",
                "Consider manual completion of PHQ-9 questionnaire for accurate assessment."
            ]
        }
    
    def validate_phq9_responses(self, responses: Dict[str, int]) -> Dict[str, Any]:
        """Validate and analyze PHQ-9 responses."""
        try:
            total_score = sum(responses.values())
            
            # Determine severity
            if total_score <= 4:
                severity = "Minimal"
            elif total_score <= 9:
                severity = "Mild"
            elif total_score <= 14:
                severity = "Moderate"
            elif total_score <= 19:
                severity = "Moderately severe"
            else:
                severity = "Severe"
            
            # Identify concerning patterns
            concerns = []
            if responses.get("q9_suicidal", 0) > 0:
                concerns.append("Suicidal ideation detected - immediate professional help recommended")
            
            if total_score >= 15:
                concerns.append("High depression risk - professional evaluation recommended")
            
            if responses.get("q3_sleep", 0) >= 2:
                concerns.append("Significant sleep disturbance - consider sleep hygiene evaluation")
            
            return {
                "total_score": total_score,
                "severity_level": severity,
                "concerns": concerns,
                "recommendations": self._get_severity_recommendations(severity, total_score)
            }
            
        except Exception as e:
            logger.error(f"Error validating PHQ-9 responses: {e}")
            return {
                "total_score": 0,
                "severity_level": "Unknown",
                "concerns": ["Validation error"],
                "recommendations": ["Please review responses and try again"]
            }
    
    def _get_severity_recommendations(self, severity: str, score: int) -> List[str]:
        """Get recommendations based on PHQ-9 severity level."""
        recommendations = []
        
        if severity == "Severe":
            recommendations.extend([
                "Immediate professional mental health evaluation recommended",
                "Consider crisis intervention services if needed",
                "Regular monitoring and support essential"
            ])
        elif severity == "Moderately severe":
            recommendations.extend([
                "Professional mental health evaluation recommended",
                "Consider therapy or counseling",
                "Monitor symptoms closely"
            ])
        elif severity == "Moderate":
            recommendations.extend([
                "Consider speaking with a healthcare provider",
                "Self-care strategies and monitoring recommended",
                "Consider therapy if symptoms persist"
            ])
        elif severity == "Mild":
            recommendations.extend([
                "Monitor mood and symptoms",
                "Self-care strategies may be helpful",
                "Consider professional help if symptoms worsen"
            ])
        else:  # Minimal
            recommendations.extend([
                "Continue current self-care practices",
                "Regular mental health check-ins recommended"
            ])
        
        return recommendations
