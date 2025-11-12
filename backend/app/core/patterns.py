"""
Design Patterns Implementation for MindGuard AI System
Implements Observer, Factory, Strategy, and MVC patterns for scalable architecture
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Callable
from enum import Enum
import logging
from datetime import datetime
import threading
from collections import deque
import numpy as np

logger = logging.getLogger(__name__)

# =============================================================================
# OBSERVER PATTERN - For Real-time Updates
# =============================================================================

class Observer(ABC):
    """Abstract observer interface for real-time updates"""
    
    @abstractmethod
    def update(self, subject: 'Subject', data: Dict[str, Any]) -> None:
        """Update method called when subject state changes"""
        pass

class Subject(ABC):
    """Abstract subject interface for observer pattern"""
    
    def __init__(self):
        self._observers: List[Observer] = []
        self._lock = threading.Lock()
    
    def attach(self, observer: Observer) -> None:
        """Attach an observer"""
        with self._lock:
            if observer not in self._observers:
                self._observers.append(observer)
                logger.debug(f"Observer attached: {type(observer).__name__}")
    
    def detach(self, observer: Observer) -> None:
        """Detach an observer"""
        with self._lock:
            if observer in self._observers:
                self._observers.remove(observer)
                logger.debug(f"Observer detached: {type(observer).__name__}")
    
    def notify(self, data: Dict[str, Any]) -> None:
        """Notify all observers"""
        with self._lock:
            for observer in self._observers:
                try:
                    observer.update(self, data)
                except Exception as e:
                    logger.error(f"Error notifying observer {type(observer).__name__}: {e}")

# =============================================================================
# FACTORY PATTERN - For Dynamic Model Loading
# =============================================================================

class ModelType(Enum):
    """Enumeration of available model types"""
    CNN = "cnn"
    MOBILENET = "mobilenet"
    RESNET = "resnet"
    EFFICIENTNET = "efficientnet"
    VGG = "vgg"
    CUSTOM = "custom"

class ModelFactory(ABC):
    """Abstract factory for creating AI models"""
    
    @abstractmethod
    def create_model(self, model_type: ModelType, **kwargs) -> 'AIModel':
        """Create a model instance"""
        pass
    
    @abstractmethod
    def get_supported_types(self) -> List[ModelType]:
        """Get list of supported model types"""
        pass

class AIModel(ABC):
    """Abstract AI model interface"""
    
    @abstractmethod
    def predict(self, input_data: np.ndarray) -> Dict[str, Any]:
        """Make prediction on input data"""
        pass
    
    @abstractmethod
    def get_accuracy(self) -> float:
        """Get current model accuracy"""
        pass
    
    @abstractmethod
    def update_accuracy(self, new_accuracy: float) -> None:
        """Update model accuracy"""
        pass
    
    @abstractmethod
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        pass

class FacialModelFactory(ModelFactory):
    """Factory for creating facial analysis models"""
    
    def __init__(self):
        self._model_registry: Dict[ModelType, Callable] = {}
        self._register_models()
    
    def _register_models(self):
        """Register available model types"""
        try:
            # Register CNN model
            from app.models.cnn_model import CNNModel
            self._model_registry[ModelType.CNN] = CNNModel
            
            # Register MobileNet model
            from app.models.mobilenet_model import MobileNetModel
            self._model_registry[ModelType.MOBILENET] = MobileNetModel
            
            # Register ResNet model
            from app.models.resnet_model import ResNetModel
            self._model_registry[ModelType.RESNET] = ResNetModel
            
            logger.info("Model factory initialized with registered models")
        except ImportError as e:
            logger.warning(f"Some models not available: {e}")
    
    def create_model(self, model_type: ModelType, **kwargs) -> AIModel:
        """Create a model instance"""
        if model_type not in self._model_registry:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        model_class = self._model_registry[model_type]
        return model_class(**kwargs)
    
    def get_supported_types(self) -> List[ModelType]:
        """Get list of supported model types"""
        return list(self._model_registry.keys())

# =============================================================================
# STRATEGY PATTERN - For Optimization Strategies
# =============================================================================

class OptimizationStrategy(ABC):
    """Abstract strategy for model optimization"""
    
    @abstractmethod
    def optimize(self, model: AIModel, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization strategy"""
        pass
    
    @abstractmethod
    def get_strategy_name(self) -> str:
        """Get strategy name"""
        pass

class DropoutTuningStrategy(OptimizationStrategy):
    """Strategy for dropout tuning"""
    
    def __init__(self, dropout_range: tuple = (0.1, 0.5)):
        self.dropout_range = dropout_range
    
    def optimize(self, model: AIModel, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply dropout tuning based on overfitting detection"""
        current_accuracy = model.get_accuracy()
        
        if current_accuracy > 0.95:  # Overfitting detected
            # Increase dropout to reduce overfitting
            new_dropout = min(self.dropout_range[1], 0.5)
            logger.info(f"Overfitting detected (accuracy: {current_accuracy:.3f}), increasing dropout to {new_dropout}")
            
            return {
                'action': 'increase_dropout',
                'new_dropout': new_dropout,
                'reason': 'overfitting_detected',
                'current_accuracy': current_accuracy
            }
        elif current_accuracy < 0.7:  # Underfitting detected
            # Decrease dropout to improve learning
            new_dropout = max(self.dropout_range[0], 0.1)
            logger.info(f"Underfitting detected (accuracy: {current_accuracy:.3f}), decreasing dropout to {new_dropout}")
            
            return {
                'action': 'decrease_dropout',
                'new_dropout': new_dropout,
                'reason': 'underfitting_detected',
                'current_accuracy': current_accuracy
            }
        
        return {
            'action': 'no_change',
            'reason': 'accuracy_optimal',
            'current_accuracy': current_accuracy
        }
    
    def get_strategy_name(self) -> str:
        return "Dropout Tuning"

class EarlyStoppingStrategy(OptimizationStrategy):
    """Strategy for early stopping"""
    
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_accuracy = 0.0
        self.wait_count = 0
    
    def optimize(self, model: AIModel, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply early stopping strategy"""
        current_accuracy = model.get_accuracy()
        
        if current_accuracy > self.best_accuracy + self.min_delta:
            self.best_accuracy = current_accuracy
            self.wait_count = 0
            return {
                'action': 'continue_training',
                'reason': 'accuracy_improving',
                'current_accuracy': current_accuracy,
                'best_accuracy': self.best_accuracy
            }
        else:
            self.wait_count += 1
            if self.wait_count >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                return {
                    'action': 'stop_training',
                    'reason': 'no_improvement',
                    'current_accuracy': current_accuracy,
                    'best_accuracy': self.best_accuracy,
                    'wait_count': self.wait_count
                }
            else:
                return {
                    'action': 'continue_training',
                    'reason': 'waiting_for_improvement',
                    'current_accuracy': current_accuracy,
                    'best_accuracy': self.best_accuracy,
                    'wait_count': self.wait_count
                }
    
    def get_strategy_name(self) -> str:
        return "Early Stopping"

class AdaptiveLearningRateStrategy(OptimizationStrategy):
    """Strategy for adaptive learning rate"""
    
    def __init__(self, initial_lr: float = 0.001, decay_factor: float = 0.5):
        self.initial_lr = initial_lr
        self.decay_factor = decay_factor
        self.current_lr = initial_lr
        self.accuracy_history = deque(maxlen=10)
    
    def optimize(self, model: AIModel, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply adaptive learning rate strategy"""
        current_accuracy = model.get_accuracy()
        self.accuracy_history.append(current_accuracy)
        
        if len(self.accuracy_history) >= 5:
            recent_improvement = current_accuracy - self.accuracy_history[0]
            
            if recent_improvement < 0.01:  # Minimal improvement
                # Reduce learning rate
                self.current_lr *= self.decay_factor
                logger.info(f"Reducing learning rate to {self.current_lr:.6f}")
                
                return {
                    'action': 'reduce_learning_rate',
                    'new_lr': self.current_lr,
                    'reason': 'minimal_improvement',
                    'current_accuracy': current_accuracy,
                    'improvement': recent_improvement
                }
            elif recent_improvement > 0.05:  # Good improvement
                # Increase learning rate slightly
                self.current_lr *= 1.1
                logger.info(f"Increasing learning rate to {self.current_lr:.6f}")
                
                return {
                    'action': 'increase_learning_rate',
                    'new_lr': self.current_lr,
                    'reason': 'good_improvement',
                    'current_accuracy': current_accuracy,
                    'improvement': recent_improvement
                }
        
        return {
            'action': 'maintain_learning_rate',
            'current_lr': self.current_lr,
            'reason': 'stable_performance',
            'current_accuracy': current_accuracy
        }
    
    def get_strategy_name(self) -> str:
        return "Adaptive Learning Rate"

# =============================================================================
# ACCURACY MONITOR - Core Component for Adaptive Tuning
# =============================================================================

class AccuracyMonitor(Subject):
    """Monitors model accuracy and triggers adaptive adjustments"""
    
    def __init__(self, window_size: int = 50):
        super().__init__()
        self.window_size = window_size
        self.accuracy_history = deque(maxlen=window_size)
        self.variance_threshold = 0.01  # Threshold for detecting overfitting
        self.overfitting_threshold = 0.95  # Accuracy threshold for overfitting
        self.underfitting_threshold = 0.7  # Accuracy threshold for underfitting
        
    def add_accuracy_reading(self, accuracy: float, model_info: Dict[str, Any]) -> None:
        """Add new accuracy reading and check for issues"""
        self.accuracy_history.append({
            'accuracy': accuracy,
            'timestamp': datetime.now(),
            'model_info': model_info
        })
        
        # Analyze accuracy patterns
        analysis = self._analyze_accuracy_patterns()
        
        # Notify observers if issues detected
        if analysis['status'] != 'normal':
            self.notify({
                'type': 'accuracy_analysis',
                'analysis': analysis,
                'current_accuracy': accuracy,
                'history_length': len(self.accuracy_history)
            })
    
    def _analyze_accuracy_patterns(self) -> Dict[str, Any]:
        """Analyze accuracy patterns for overfitting/underfitting"""
        if len(self.accuracy_history) < 10:
            return {'status': 'insufficient_data', 'message': 'Need more data points'}
        
        recent_accuracies = [entry['accuracy'] for entry in list(self.accuracy_history)[-10:]]
        current_accuracy = recent_accuracies[-1]
        
        # Calculate variance
        variance = np.var(recent_accuracies)
        
        # Check for overfitting (high accuracy with low variance)
        if current_accuracy > self.overfitting_threshold and variance < self.variance_threshold:
            return {
                'status': 'overfitting',
                'message': 'Model showing signs of overfitting',
                'current_accuracy': current_accuracy,
                'variance': variance,
                'recommendation': 'increase_regularization'
            }
        
        # Check for underfitting (low accuracy)
        if current_accuracy < self.underfitting_threshold:
            return {
                'status': 'underfitting',
                'message': 'Model showing signs of underfitting',
                'current_accuracy': current_accuracy,
                'variance': variance,
                'recommendation': 'reduce_regularization'
            }
        
        # Check for instability (high variance)
        if variance > self.variance_threshold * 3:
            return {
                'status': 'unstable',
                'message': 'Model showing unstable performance',
                'current_accuracy': current_accuracy,
                'variance': variance,
                'recommendation': 'stabilize_training'
            }
        
        return {
            'status': 'normal',
            'message': 'Model performance is normal',
            'current_accuracy': current_accuracy,
            'variance': variance
        }
    
    def get_accuracy_stats(self) -> Dict[str, Any]:
        """Get accuracy statistics"""
        if not self.accuracy_history:
            return {'status': 'no_data'}
        
        accuracies = [entry['accuracy'] for entry in self.accuracy_history]
        
        return {
            'current_accuracy': accuracies[-1],
            'average_accuracy': np.mean(accuracies),
            'min_accuracy': np.min(accuracies),
            'max_accuracy': np.max(accuracies),
            'variance': np.var(accuracies),
            'trend': self._calculate_trend(accuracies),
            'data_points': len(accuracies)
        }
    
    def _calculate_trend(self, accuracies: List[float]) -> str:
        """Calculate accuracy trend"""
        if len(accuracies) < 5:
            return 'insufficient_data'
        
        recent = accuracies[-5:]
        older = accuracies[-10:-5] if len(accuracies) >= 10 else accuracies[:-5]
        
        recent_avg = np.mean(recent)
        older_avg = np.mean(older)
        
        if recent_avg > older_avg + 0.01:
            return 'improving'
        elif recent_avg < older_avg - 0.01:
            return 'declining'
        else:
            return 'stable'

# =============================================================================
# MODEL MANAGER - Coordinates All Components
# =============================================================================

class ModelManager(Observer):
    """Manages AI models with adaptive accuracy tuning"""
    
    def __init__(self):
        self.model_factory = FacialModelFactory()
        self.current_model: Optional[AIModel] = None
        self.accuracy_monitor = AccuracyMonitor()
        self.optimization_strategies: List[OptimizationStrategy] = []
        self.model_history: List[Dict[str, Any]] = []
        
        # Initialize strategies
        self._initialize_strategies()
        
        # Subscribe to accuracy monitoring
        self.accuracy_monitor.attach(self)
    
    def _initialize_strategies(self):
        """Initialize optimization strategies"""
        self.optimization_strategies = [
            DropoutTuningStrategy(),
            EarlyStoppingStrategy(),
            AdaptiveLearningRateStrategy()
        ]
        logger.info(f"Initialized {len(self.optimization_strategies)} optimization strategies")
    
    def create_model(self, model_type: ModelType, **kwargs) -> AIModel:
        """Create a new model"""
        try:
            model = self.model_factory.create_model(model_type, **kwargs)
            self.current_model = model
            
            # Record model creation
            model_info = {
                'model_type': model_type.value,
                'created_at': datetime.now(),
                'model_info': model.get_model_info()
            }
            self.model_history.append(model_info)
            
            logger.info(f"Created new model: {model_type.value}")
            return model
            
        except Exception as e:
            logger.error(f"Failed to create model {model_type.value}: {e}")
            raise
    
    def switch_model(self, new_model_type: ModelType, **kwargs) -> AIModel:
        """Switch to a different model type"""
        if self.current_model:
            # Save current model state
            current_info = {
                'model_type': 'previous',
                'accuracy': self.current_model.get_accuracy(),
                'switched_at': datetime.now()
            }
            self.model_history.append(current_info)
        
        # Create new model
        new_model = self.create_model(new_model_type, **kwargs)
        logger.info(f"Switched to model: {new_model_type.value}")
        
        return new_model
    
    def update(self, subject: Subject, data: Dict[str, Any]) -> None:
        """Observer update method"""
        if isinstance(subject, AccuracyMonitor):
            self._handle_accuracy_update(data)
    
    def _handle_accuracy_update(self, data: Dict[str, Any]) -> None:
        """Handle accuracy monitoring updates"""
        analysis = data.get('analysis', {})
        
        if analysis.get('status') in ['overfitting', 'underfitting', 'unstable']:
            logger.warning(f"Model issue detected: {analysis['status']}")
            
            # Apply optimization strategies
            for strategy in self.optimization_strategies:
                try:
                    result = strategy.optimize(self.current_model, data)
                    logger.info(f"Applied {strategy.get_strategy_name()}: {result['action']}")
                    
                    # Record optimization action
                    optimization_record = {
                        'strategy': strategy.get_strategy_name(),
                        'action': result['action'],
                        'reason': result.get('reason', 'unknown'),
                        'timestamp': datetime.now(),
                        'model_accuracy': self.current_model.get_accuracy() if self.current_model else 0.0
                    }
                    self.model_history.append(optimization_record)
                    
                except Exception as e:
                    logger.error(f"Error applying strategy {strategy.get_strategy_name()}: {e}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status"""
        if not self.current_model:
            return {'status': 'no_model', 'message': 'No model currently loaded'}
        
        accuracy_stats = self.accuracy_monitor.get_accuracy_stats()
        
        return {
            'status': 'active',
            'model_info': self.current_model.get_model_info(),
            'current_accuracy': self.current_model.get_accuracy(),
            'accuracy_stats': accuracy_stats,
            'optimization_strategies': [s.get_strategy_name() for s in self.optimization_strategies],
            'model_history_count': len(self.model_history)
        }
    
    def get_supported_models(self) -> List[str]:
        """Get list of supported model types"""
        return [model_type.value for model_type in self.model_factory.get_supported_types()]

# =============================================================================
# WEBSOCKET MANAGER - For Real-time Communication
# =============================================================================

class WebSocketManager(Observer):
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.connections: List[Any] = []  # Will store WebSocket connections
        self.message_queue = deque(maxlen=1000)
    
    def add_connection(self, websocket: Any) -> None:
        """Add a new WebSocket connection"""
        self.connections.append(websocket)
        logger.info(f"WebSocket connection added. Total connections: {len(self.connections)}")
    
    def remove_connection(self, websocket: Any) -> None:
        """Remove a WebSocket connection"""
        if websocket in self.connections:
            self.connections.remove(websocket)
            logger.info(f"WebSocket connection removed. Total connections: {len(self.connections)}")
    
    def update(self, subject: Subject, data: Dict[str, Any]) -> None:
        """Observer update method for real-time notifications"""
        # Queue message for broadcasting
        message = {
            'timestamp': datetime.now().isoformat(),
            'type': data.get('type', 'update'),
            'data': data
        }
        
        self.message_queue.append(message)
        self._broadcast_message(message)
    
    async def _broadcast_message(self, message: Dict[str, Any]) -> None:
        """Broadcast message to all connected clients"""
        if not self.connections:
            return
        
        # This would be implemented with actual WebSocket broadcasting
        # For now, we'll just log the message
        logger.debug(f"Broadcasting message to {len(self.connections)} connections: {message['type']}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics"""
        return {
            'active_connections': len(self.connections),
            'queued_messages': len(self.message_queue),
            'status': 'active' if self.connections else 'inactive'
        }

# Global instances
model_manager = ModelManager()
websocket_manager = WebSocketManager()

# Subscribe WebSocket manager to accuracy monitoring
model_manager.accuracy_monitor.attach(websocket_manager)
