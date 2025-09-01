"""
Enhanced facial analysis service providing comprehensive real-time assessment
including mood, sleepiness, fatigue, stress indicators, and PHQ-9 estimation.
"""

import cv2
import numpy as np
import logging
import math
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import uuid

from app.models.facial_metrics import (
    ComprehensiveFacialAnalysis, EyeMetrics, HeadPoseMetrics,
    MicroExpressionMetrics, EmotionDistribution, SleepinessLevel,
    FatigueIndicators, StressLevel, PHQ9Estimation
)

logger = logging.getLogger(__name__)

class EnhancedFacialAnalyzer:
    """Enhanced facial analyzer with comprehensive metrics extraction."""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.emotion_detector = None
        self.landmark_detector = None
        self._initialize_detectors()
        
        # Analysis parameters
        self.ear_threshold = 0.3  # Eye Aspect Ratio threshold for blink detection
        self.fatigue_head_angle_threshold = 15  # degrees
        self.stress_micro_movement_threshold = 0.5
        
        # PHQ-9 mapping weights
        self.phq9_emotion_weights = {
            'sad': 3.5,
            'angry': 2.0,
            'fear': 2.5,
            'neutral': 1.0,
            'happy': -1.5,
            'surprise': 0.5,
            'disgust': 1.5
        }
    
    def _initialize_detectors(self):
        """Initialize all detection models."""
        try:
            # Initialize OpenCV cascades
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
            
            # Try to initialize advanced detectors
            try:
                import dlib
                self.landmark_detector = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
            except:
                logger.warning("Dlib landmark detector not available - using fallback methods")
                
            # Initialize emotion detector
            try:
                from fer import FER
                self.emotion_detector = FER(mtcnn=True)
                logger.info("FER emotion detector initialized")
            except ImportError:
                logger.warning("FER library not available - using mock emotion detection")
                
        except Exception as e:
            logger.error(f"Error initializing detectors: {e}")
    
    def analyze_comprehensive(self, image: np.ndarray, session_id: str = None, user_id: str = None) -> ComprehensiveFacialAnalysis:
        """
        Perform comprehensive facial analysis on the input image.
        
        Args:
            image: Input image as numpy array
            session_id: Optional session identifier
            user_id: Optional user identifier
            
        Returns:
            ComprehensiveFacialAnalysis with all metrics
        """
        try:
            # Check image quality
            frame_quality = self._assess_frame_quality(image)
            
            # Detect faces
            faces = self._detect_faces(image)
            
            if len(faces) == 0:  # Use len() instead of not faces to avoid array ambiguity
                return self._create_no_face_result(frame_quality)
            
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_roi = image[y:y+h, x:x+w]
            
            # Extract all metrics
            emotion_data = self._analyze_emotions(face_roi)
            eye_metrics = self._analyze_eyes(face_roi)
            head_pose = self._analyze_head_pose(face_roi)
            micro_expressions = self._analyze_micro_expressions(face_roi)
            
            # Calculate derived insights
            sleepiness = self._assess_sleepiness(eye_metrics)
            fatigue = self._assess_fatigue(eye_metrics, head_pose)
            stress = self._assess_stress(micro_expressions, emotion_data)
            phq9_estimation = self._estimate_phq9(emotion_data, stress, sleepiness)
            
            # Calculate overall analysis quality
            analysis_quality = self._calculate_analysis_quality(
                frame_quality, len(faces), eye_metrics, head_pose
            )
            
            return ComprehensiveFacialAnalysis(
                face_detected=True,
                primary_emotion=emotion_data['primary_emotion'],
                emotion_confidence=emotion_data['confidence'],
                emotion_distribution=EmotionDistribution(**emotion_data['distribution']),
                eye_metrics=eye_metrics,
                head_pose=head_pose,
                micro_expressions=micro_expressions,
                mood_assessment=self._map_emotion_to_mood(emotion_data['primary_emotion']),
                sleepiness=sleepiness,
                fatigue=fatigue,
                stress=stress,
                phq9_estimation=phq9_estimation,
                analysis_quality=analysis_quality,
                frame_quality=frame_quality
            )
            
        except Exception as e:
            logger.error(f"Error in comprehensive analysis: {e}")
            return self._create_error_result(frame_quality)
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in the image."""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            # Convert numpy array to list of tuples to avoid boolean ambiguity
            return [tuple(face) for face in faces] if len(faces) > 0 else []
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _analyze_emotions(self, face_roi: np.ndarray) -> Dict:
        """Analyze emotions using FER or fallback method."""
        try:
            if self.emotion_detector:
                result = self.emotion_detector.detect_emotions(face_roi)
                if result and len(result) > 0:  # Check if result exists and has content
                    emotions = result[0]['emotions']
                    primary_emotion = max(emotions, key=emotions.get)
                    confidence = emotions[primary_emotion]
                    
                    return {
                        'primary_emotion': primary_emotion,
                        'confidence': confidence,
                        'distribution': emotions
                    }
            
            # Fallback to mock emotions
            return self._mock_emotion_analysis()
            
        except Exception as e:
            logger.error(f"Error in emotion analysis: {e}")
            return self._mock_emotion_analysis()
    
    def _analyze_eyes(self, face_roi: np.ndarray) -> EyeMetrics:
        """Analyze eye metrics for sleepiness detection."""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray)
            
            if len(eyes) >= 2:  # Use len() to avoid array boolean ambiguity
                # Calculate EAR for detected eyes
                left_ear = self._calculate_ear(eyes[0], gray)
                right_ear = self._calculate_ear(eyes[1], gray)
                avg_ear = (left_ear + right_ear) / 2
                
                # Estimate blink metrics (simplified)
                blink_rate = self._estimate_blink_rate(avg_ear)
                blink_duration = self._estimate_blink_duration(avg_ear)
                
                return EyeMetrics(
                    left_ear=left_ear,
                    right_ear=right_ear,
                    avg_ear=avg_ear,
                    blink_rate=blink_rate,
                    blink_duration=blink_duration
                )
            else:
                # Fallback values when eyes not detected properly
                return EyeMetrics(
                    left_ear=0.35,
                    right_ear=0.35,
                    avg_ear=0.35,
                    blink_rate=15.0,
                    blink_duration=150.0
                )
                
        except Exception as e:
            logger.error(f"Error in eye analysis: {e}")
            return EyeMetrics(
                left_ear=0.35,
                right_ear=0.35,
                avg_ear=0.35,
                blink_rate=15.0,
                blink_duration=150.0
            )
    
    def _analyze_head_pose(self, face_roi: np.ndarray) -> HeadPoseMetrics:
        """Analyze head pose for fatigue detection."""
        try:
            # Simplified head pose estimation using face contours
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Calculate rough head angles based on face position and shape
            pitch = self._estimate_pitch(gray)
            yaw = self._estimate_yaw(gray, w)
            roll = self._estimate_roll(gray)
            stability = self._calculate_head_stability([pitch, yaw, roll])
            
            return HeadPoseMetrics(
                pitch=pitch,
                yaw=yaw,
                roll=roll,
                stability=stability
            )
            
        except Exception as e:
            logger.error(f"Error in head pose analysis: {e}")
            return HeadPoseMetrics(
                pitch=0.0,
                yaw=0.0,
                roll=0.0,
                stability=0.8
            )
    
    def _analyze_micro_expressions(self, face_roi: np.ndarray) -> MicroExpressionMetrics:
        """Analyze micro-expressions for stress detection."""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Calculate metrics based on facial features
            muscle_tension = self._calculate_muscle_tension(gray)
            asymmetry = self._calculate_facial_asymmetry(gray)
            micro_movement_freq = self._estimate_micro_movements(gray)
            expression_variability = self._calculate_expression_variability(gray)
            
            return MicroExpressionMetrics(
                muscle_tension=muscle_tension,
                asymmetry=asymmetry,
                micro_movement_frequency=micro_movement_freq,
                expression_variability=expression_variability
            )
            
        except Exception as e:
            logger.error(f"Error in micro-expression analysis: {e}")
            return MicroExpressionMetrics(
                muscle_tension=0.3,
                asymmetry=0.1,
                micro_movement_frequency=0.2,
                expression_variability=0.4
            )
    
    def _assess_sleepiness(self, eye_metrics: EyeMetrics) -> SleepinessLevel:
        """Assess sleepiness level based on eye metrics."""
        factors = []
        
        if eye_metrics.avg_ear < 0.25:
            level = "Very tired"
            confidence = 0.8
            factors.extend(["Low eye aspect ratio", "Frequent blinking"])
        elif eye_metrics.avg_ear < 0.3:
            level = "Slightly tired"
            confidence = 0.6
            factors.append("Reduced eye openness")
        else:
            level = "Alert"
            confidence = 0.7
            factors.append("Normal eye activity")
        
        if eye_metrics.blink_rate > 20:
            factors.append("High blink frequency")
            if level == "Alert":
                level = "Slightly tired"
                confidence = 0.6
        
        return SleepinessLevel(
            level=level,
            confidence=confidence,
            contributing_factors=factors
        )
    
    def _assess_fatigue(self, eye_metrics: EyeMetrics, head_pose: HeadPoseMetrics) -> FatigueIndicators:
        """Assess fatigue indicators."""
        yawning_detected = eye_metrics.avg_ear < 0.2  # Simplified yawn detection
        head_droop_detected = abs(head_pose.pitch) > self.fatigue_head_angle_threshold
        
        overall_fatigue = yawning_detected or head_droop_detected or eye_metrics.avg_ear < 0.25
        
        confidence = 0.7 if (yawning_detected or head_droop_detected) else 0.5
        
        return FatigueIndicators(
            yawning_detected=yawning_detected,
            head_droop_detected=head_droop_detected,
            overall_fatigue=overall_fatigue,
            confidence=confidence
        )
    
    def _assess_stress(self, micro_expressions: MicroExpressionMetrics, emotion_data: Dict) -> StressLevel:
        """Assess stress level based on micro-expressions and emotions."""
        indicators = []
        stress_score = 0
        
        # Check micro-expression indicators
        if micro_expressions.muscle_tension > 0.6:
            stress_score += 2
            indicators.append("High muscle tension")
        
        if micro_expressions.asymmetry > 0.3:
            stress_score += 1
            indicators.append("Facial asymmetry")
        
        if micro_expressions.micro_movement_frequency > self.stress_micro_movement_threshold:
            stress_score += 1
            indicators.append("Frequent micro-movements")
        
        # Check emotional indicators
        stress_emotions = ['angry', 'fear', 'disgust']
        for emotion in stress_emotions:
            if emotion_data['distribution'].get(emotion, 0) > 0.3:
                stress_score += 1
                indicators.append(f"High {emotion} expression")
        
        # Determine stress level
        if stress_score >= 3:
            level = "High"
            confidence = 0.8
        elif stress_score >= 1:
            level = "Medium" 
            confidence = 0.6
        else:
            level = "Low"
            confidence = 0.7
            indicators = ["No significant stress indicators"]
        
        return StressLevel(
            level=level,
            confidence=confidence,
            indicators=indicators
        )
    
    def _estimate_phq9(self, emotion_data: Dict, stress: StressLevel, sleepiness: SleepinessLevel) -> PHQ9Estimation:
        """Estimate PHQ-9 score based on facial analysis."""
        base_score = 0
        contributing_expressions = []
        
        # Calculate score based on emotion distribution
        for emotion, weight in self.phq9_emotion_weights.items():
            emotion_strength = emotion_data['distribution'].get(emotion, 0)
            contribution = emotion_strength * weight
            base_score += contribution
            
            if emotion_strength > 0.3 and weight > 0:
                contributing_expressions.append(f"High {emotion} expression")
        
        # Adjust for stress and sleepiness
        if stress.level == "High":
            base_score += 3
            contributing_expressions.append("High stress indicators")
        elif stress.level == "Medium":
            base_score += 1
        
        if sleepiness.level == "Very tired":
            base_score += 2
            contributing_expressions.append("Severe fatigue signs")
        elif sleepiness.level == "Slightly tired":
            base_score += 1
        
        # Normalize to PHQ-9 scale (0-27)
        estimated_score = max(0, min(27, int(base_score * 3)))
        
        # Determine severity level
        if estimated_score <= 4:
            severity = "Minimal"
        elif estimated_score <= 9:
            severity = "Mild"
        elif estimated_score <= 14:
            severity = "Moderate"
        elif estimated_score <= 19:
            severity = "Moderately severe"
        else:
            severity = "Severe"
        
        # Calculate confidence based on analysis quality
        confidence = min(0.8, emotion_data['confidence'] * 0.7 + 0.1)
        
        return PHQ9Estimation(
            estimated_score=estimated_score,
            confidence=confidence,
            severity_level=severity,
            contributing_expressions=contributing_expressions
        )
    
    # Helper methods for calculations
    def _calculate_ear(self, eye_region: Tuple[int, int, int, int], gray: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio."""
        x, y, w, h = eye_region
        roi = gray[y:y+h, x:x+w]
        if roi.size > 0:
            # Simplified EAR calculation based on region variance
            variance = np.var(roi)
            return min(0.5, variance / 1000.0)
        return 0.35
    
    def _estimate_blink_rate(self, avg_ear: float) -> float:
        """Estimate blink rate based on EAR."""
        if avg_ear < 0.25:
            return 25.0  # High blink rate
        elif avg_ear < 0.3:
            return 18.0  # Moderate blink rate
        else:
            return 12.0  # Normal blink rate
    
    def _estimate_blink_duration(self, avg_ear: float) -> float:
        """Estimate average blink duration."""
        return 120.0 + (0.35 - avg_ear) * 200.0
    
    def _estimate_pitch(self, gray: np.ndarray) -> float:
        """Estimate head pitch angle."""
        h, w = gray.shape
        upper_half = gray[:h//2, :]
        lower_half = gray[h//2:, :]
        
        upper_intensity = np.mean(upper_half)
        lower_intensity = np.mean(lower_half)
        
        # Simplified pitch estimation
        pitch = (lower_intensity - upper_intensity) / 10.0
        return np.clip(pitch, -30, 30)
    
    def _estimate_yaw(self, gray: np.ndarray, width: int) -> float:
        """Estimate head yaw angle."""
        left_half = gray[:, :width//2]
        right_half = gray[:, width//2:]
        
        left_intensity = np.mean(left_half)
        right_intensity = np.mean(right_half)
        
        # Simplified yaw estimation
        yaw = (right_intensity - left_intensity) / 5.0
        return np.clip(yaw, -45, 45)
    
    def _estimate_roll(self, gray: np.ndarray) -> float:
        """Estimate head roll angle."""
        # Simplified roll estimation using edge detection
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
        
        if lines is not None:
            angles = []
            for line in lines[:5]:  # Take first 5 lines
                rho, theta = line[0]
                angle = theta * 180 / np.pi - 90
                angles.append(angle)
            
            if angles:
                return np.clip(np.mean(angles), -30, 30)
        
        return 0.0
    
    def _calculate_head_stability(self, angles: List[float]) -> float:
        """Calculate head stability score."""
        angle_variance = np.var(angles)
        stability = max(0.0, 1.0 - angle_variance / 100.0)
        return min(1.0, stability)
    
    def _calculate_muscle_tension(self, gray: np.ndarray) -> float:
        """Calculate facial muscle tension."""
        # Use edge density as a proxy for muscle tension
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / edges.size
        return min(1.0, edge_density * 3.0)
    
    def _calculate_facial_asymmetry(self, gray: np.ndarray) -> float:
        """Calculate facial asymmetry."""
        h, w = gray.shape
        left_half = gray[:, :w//2]
        right_half = cv2.flip(gray[:, w//2:], 1)
        
        # Resize to match dimensions
        min_width = min(left_half.shape[1], right_half.shape[1])
        left_half = left_half[:, :min_width]
        right_half = right_half[:, :min_width]
        
        # Calculate difference
        diff = np.abs(left_half.astype(float) - right_half.astype(float))
        asymmetry = np.mean(diff) / 255.0
        return min(1.0, asymmetry)
    
    def _estimate_micro_movements(self, gray: np.ndarray) -> float:
        """Estimate micro-movement frequency."""
        # Simplified using image gradients
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        movement_score = np.mean(gradient_magnitude) / 255.0
        return min(1.0, movement_score)
    
    def _calculate_expression_variability(self, gray: np.ndarray) -> float:
        """Calculate expression variability."""
        # Use local standard deviation as variability measure
        kernel = np.ones((5, 5), np.float32) / 25
        smoothed = cv2.filter2D(gray, -1, kernel)
        variability = np.std(gray - smoothed) / 255.0
        return min(1.0, variability * 2.0)
    
    def _assess_frame_quality(self, image: np.ndarray) -> float:
        """Assess the quality of the input frame."""
        try:
            # Check image size
            h, w = image.shape[:2]
            size_score = min(1.0, (h * w) / (480 * 640))
            
            # Check brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Check sharpness using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            return (size_score + brightness_score + sharpness_score) / 3.0
            
        except Exception:
            return 0.5
    
    def _calculate_analysis_quality(self, frame_quality: float, num_faces: int, 
                                  eye_metrics: EyeMetrics, head_pose: HeadPoseMetrics) -> float:
        """Calculate overall analysis quality score."""
        face_score = 1.0 if num_faces == 1 else (0.7 if num_faces > 1 else 0.3)
        eye_score = 1.0 if eye_metrics.avg_ear > 0.2 else 0.5
        pose_score = 1.0 - (abs(head_pose.yaw) + abs(head_pose.pitch)) / 100.0
        
        return (frame_quality + face_score + eye_score + pose_score) / 4.0
    
    def _map_emotion_to_mood(self, emotion: str) -> str:
        """Map detected emotion to mood categories."""
        mood_mapping = {
            'happy': 'Happy',
            'sad': 'Sad',
            'angry': 'Angry',
            'neutral': 'Neutral',
            'surprise': 'Surprised',
            'fear': 'Sad',  # Map fear to sad for simplicity
            'disgust': 'Angry'  # Map disgust to angry
        }
        return mood_mapping.get(emotion, 'Neutral')
    
    def _mock_emotion_analysis(self) -> Dict:
        """Fallback emotion analysis when FER is not available."""
        import random
        
        emotions = {
            'neutral': 0.4,
            'happy': 0.2,
            'sad': 0.15,
            'angry': 0.1,
            'surprise': 0.05,
            'fear': 0.05,
            'disgust': 0.05
        }
        
        primary_emotion = max(emotions, key=emotions.get)
        
        return {
            'primary_emotion': primary_emotion,
            'confidence': emotions[primary_emotion],
            'distribution': emotions
        }
    
    def _create_no_face_result(self, frame_quality: float) -> ComprehensiveFacialAnalysis:
        """Create result when no face is detected."""
        return ComprehensiveFacialAnalysis(
            face_detected=False,
            primary_emotion="no_face_detected",
            emotion_confidence=0.0,
            emotion_distribution=EmotionDistribution(
                happy=0, sad=0, angry=0, fear=0, surprise=0, disgust=0, neutral=0
            ),
            mood_assessment="Unknown",
            sleepiness=SleepinessLevel(level="Unknown", confidence=0.0, contributing_factors=["No face detected"]),
            fatigue=FatigueIndicators(yawning_detected=False, head_droop_detected=False, overall_fatigue=False, confidence=0.0),
            stress=StressLevel(level="Unknown", confidence=0.0, indicators=["No face detected"]),
            phq9_estimation=PHQ9Estimation(
                estimated_score=0, confidence=0.0, severity_level="Unknown", contributing_expressions=[]
            ),
            analysis_quality=0.0,
            frame_quality=frame_quality
        )
    
    def _create_error_result(self, frame_quality: float) -> ComprehensiveFacialAnalysis:
        """Create result when analysis fails."""
        return ComprehensiveFacialAnalysis(
            face_detected=False,
            primary_emotion="analysis_error",
            emotion_confidence=0.0,
            emotion_distribution=EmotionDistribution(
                happy=0, sad=0, angry=0, fear=0, surprise=0, disgust=0, neutral=0
            ),
            mood_assessment="Error",
            sleepiness=SleepinessLevel(level="Error", confidence=0.0, contributing_factors=["Analysis failed"]),
            fatigue=FatigueIndicators(yawning_detected=False, head_droop_detected=False, overall_fatigue=False, confidence=0.0),
            stress=StressLevel(level="Error", confidence=0.0, indicators=["Analysis failed"]),
            phq9_estimation=PHQ9Estimation(
                estimated_score=0, confidence=0.0, severity_level="Error", contributing_expressions=[]
            ),
            analysis_quality=0.0,
            frame_quality=frame_quality
        )
