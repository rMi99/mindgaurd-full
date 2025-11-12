"""
Enhanced Facial Analysis Service with Design Patterns
Integrates Observer, Factory, Strategy patterns with adaptive accuracy tuning
"""

import cv2
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import io
import base64
from PIL import Image

from app.core.patterns import (
    ModelManager, ModelType, AccuracyMonitor, 
    WebSocketManager, Observer, Subject
)
from app.services.real_facial_analyzer import RealFacialAnalyzer

logger = logging.getLogger(__name__)

class AdaptiveFacialAnalysisService(Subject):
    """
    Enhanced facial analysis service with adaptive accuracy tuning
    Implements Observer pattern for real-time updates
    """
    
    def __init__(self):
        super().__init__()
        self.model_manager = ModelManager()
        self.real_analyzer = RealFacialAnalyzer()
        self.accuracy_monitor = AccuracyMonitor()
        self.session_data: Dict[str, Any] = {}
        self.analysis_history: List[Dict[str, Any]] = []
        
        # Initialize with default model
        self._initialize_default_model()
        
        # Subscribe to accuracy monitoring
        self.accuracy_monitor.attach(self)
    
    def _initialize_default_model(self):
        """Initialize with default CNN model"""
        try:
            self.current_model = self.model_manager.create_model(ModelType.CNN)
            logger.info("Initialized with CNN model")
        except Exception as e:
            logger.error(f"Failed to initialize default model: {e}")
            self.current_model = None
    
    def analyze_facial_expression(self, image_file) -> Dict[str, Any]:
        """
        Analyze facial expression with adaptive accuracy tuning
        
        Args:
            image_file: File-like object containing image data
            
        Returns:
            Comprehensive analysis results
        """
        try:
            # Get base analysis from real analyzer
            base_result = self.real_analyzer.analyze_facial_expression(image_file)
            
            # Enhance with adaptive model if available
            if self.current_model:
                enhanced_result = self._enhance_with_adaptive_model(image_file, base_result)
            else:
                enhanced_result = base_result
            
            # Update accuracy monitoring
            self._update_accuracy_monitoring(enhanced_result)
            
            # Store analysis history
            self.analysis_history.append({
                'timestamp': datetime.now(),
                'result': enhanced_result,
                'model_type': enhanced_result.get('model_type', 'unknown')
            })
            
            # Notify observers
            self.notify({
                'type': 'facial_analysis',
                'result': enhanced_result,
                'session_id': enhanced_result.get('session_id')
            })
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error in adaptive facial analysis: {e}")
            return self._create_error_result(str(e))
    
    def _enhance_with_adaptive_model(self, image_file, base_result: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance analysis with adaptive model"""
        try:
            # Read image for model prediction
            image_data = image_file.file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return base_result
            
            # Get model prediction
            model_prediction = self.current_model.predict(image)
            
            # Combine results
            enhanced_result = base_result.copy()
            enhanced_result.update({
                'model_prediction': model_prediction,
                'model_type': model_prediction.get('model_type', 'unknown'),
                'model_confidence': model_prediction.get('confidence', 0.0),
                'adaptive_analysis': True
            })
            
            # Use model prediction if confidence is higher
            if model_prediction.get('confidence', 0) > base_result.get('confidence', 0):
                enhanced_result.update({
                    'emotion': model_prediction.get('emotion'),
                    'confidence': model_prediction.get('confidence'),
                    'probabilities': model_prediction.get('probabilities')
                })
            
            return enhanced_result
            
        except Exception as e:
            logger.error(f"Error enhancing with adaptive model: {e}")
            return base_result
    
    def _update_accuracy_monitoring(self, result: Dict[str, Any]):
        """Update accuracy monitoring with new result"""
        try:
            confidence = result.get('confidence', 0.0)
            model_info = {
                'model_type': result.get('model_type', 'unknown'),
                'face_detected': result.get('face_detected', False),
                'analysis_quality': result.get('analysis_quality', 0.0)
            }
            
            # Add accuracy reading
            self.accuracy_monitor.add_accuracy_reading(confidence, model_info)
            
        except Exception as e:
            logger.error(f"Error updating accuracy monitoring: {e}")
    
    def switch_model(self, model_type: str) -> Dict[str, Any]:
        """Switch to different model type"""
        try:
            model_enum = ModelType(model_type.lower())
            new_model = self.model_manager.switch_model(model_enum)
            self.current_model = new_model
            
            logger.info(f"Switched to {model_type} model")
            
            return {
                'status': 'success',
                'message': f'Switched to {model_type} model',
                'model_info': new_model.get_model_info()
            }
            
        except ValueError:
            return {
                'status': 'error',
                'message': f'Unsupported model type: {model_type}',
                'supported_models': self.model_manager.get_supported_models()
            }
        except Exception as e:
            logger.error(f"Error switching model: {e}")
            return {
                'status': 'error',
                'message': f'Failed to switch model: {str(e)}'
            }
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and statistics"""
        try:
            model_status = self.model_manager.get_model_status()
            accuracy_stats = self.accuracy_monitor.get_accuracy_stats()
            
            return {
                'model_status': model_status,
                'accuracy_stats': accuracy_stats,
                'analysis_history_count': len(self.analysis_history),
                'service_status': 'active'
            }
            
        except Exception as e:
            logger.error(f"Error getting model status: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_analysis_trends(self, limit: int = 50) -> Dict[str, Any]:
        """Get analysis trends and statistics"""
        try:
            if not self.analysis_history:
                return {
                    'trends': [],
                    'statistics': {},
                    'message': 'No analysis data available'
                }
            
            # Get recent analyses
            recent_analyses = self.analysis_history[-limit:]
            
            # Calculate statistics
            emotions = [a['result'].get('emotion', 'unknown') for a in recent_analyses]
            confidences = [a['result'].get('confidence', 0.0) for a in recent_analyses]
            
            # Emotion distribution
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            # Confidence statistics
            avg_confidence = np.mean(confidences) if confidences else 0.0
            min_confidence = np.min(confidences) if confidences else 0.0
            max_confidence = np.max(confidences) if confidences else 0.0
            
            # Model usage
            model_usage = {}
            for analysis in recent_analyses:
                model_type = analysis.get('model_type', 'unknown')
                model_usage[model_type] = model_usage.get(model_type, 0) + 1
            
            return {
                'trends': recent_analyses,
                'statistics': {
                    'total_analyses': len(recent_analyses),
                    'emotion_distribution': emotion_counts,
                    'confidence_stats': {
                        'average': avg_confidence,
                        'minimum': min_confidence,
                        'maximum': max_confidence
                    },
                    'model_usage': model_usage,
                    'time_range': {
                        'start': recent_analyses[0]['timestamp'].isoformat() if recent_analyses else None,
                        'end': recent_analyses[-1]['timestamp'].isoformat() if recent_analyses else None
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting analysis trends: {e}")
            return {
                'trends': [],
                'statistics': {},
                'error': str(e)
            }
    
    def start_session(self, user_id: str) -> Dict[str, Any]:
        """Start analysis session"""
        try:
            session_id = f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            self.session_data[session_id] = {
                'user_id': user_id,
                'start_time': datetime.now(),
                'analysis_count': 0,
                'total_confidence': 0.0,
                'emotions': [],
                'status': 'active'
            }
            
            logger.info(f"Started session {session_id} for user {user_id}")
            
            return {
                'status': 'success',
                'session_id': session_id,
                'message': 'Session started successfully'
            }
            
        except Exception as e:
            logger.error(f"Error starting session: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Stop analysis session"""
        try:
            if session_id not in self.session_data:
                return {
                    'status': 'error',
                    'message': 'Session not found'
                }
            
            session = self.session_data[session_id]
            session['status'] = 'stopped'
            session['end_time'] = datetime.now()
            session['duration'] = (session['end_time'] - session['start_time']).total_seconds()
            
            # Calculate session statistics
            if session['analysis_count'] > 0:
                session['average_confidence'] = session['total_confidence'] / session['analysis_count']
            
            logger.info(f"Stopped session {session_id}")
            
            return {
                'status': 'success',
                'session_summary': session,
                'message': 'Session stopped successfully'
            }
            
        except Exception as e:
            logger.error(f"Error stopping session: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get session status"""
        try:
            if session_id not in self.session_data:
                return {
                    'status': 'error',
                    'message': 'Session not found'
                }
            
            session = self.session_data[session_id]
            current_time = datetime.now()
            
            if session['status'] == 'active':
                duration = (current_time - session['start_time']).total_seconds()
            else:
                duration = session.get('duration', 0)
            
            return {
                'status': 'success',
                'session_id': session_id,
                'session_status': session['status'],
                'duration_seconds': duration,
                'analysis_count': session['analysis_count'],
                'average_confidence': session.get('average_confidence', 0.0)
            }
            
        except Exception as e:
            logger.error(f"Error getting session status: {e}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def update(self, subject: Subject, data: Dict[str, Any]) -> None:
        """Observer update method"""
        if 'accuracy_analysis' in data.get('type', ''):
            logger.info(f"Accuracy analysis update: {data.get('analysis', {}).get('status', 'unknown')}")
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result"""
        return {
            'emotion': 'analysis_error',
            'confidence': 0.0,
            'face_detected': False,
            'error': error_message,
            'timestamp': datetime.now().isoformat(),
            'model_type': 'error'
        }

# Global service instance
adaptive_service = AdaptiveFacialAnalysisService()

def get_adaptive_service() -> AdaptiveFacialAnalysisService:
    """Get the global adaptive service instance"""
    return adaptive_service
