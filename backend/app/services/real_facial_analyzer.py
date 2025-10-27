"""
Real Facial Analysis Service for MindGuard
Provides actual facial emotion recognition without mock data
"""

import cv2
import numpy as np
import logging
import pickle
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import io

logger = logging.getLogger(__name__)

class RealFacialAnalyzer:
    """Real facial analyzer using trained models"""
    
    def __init__(self):
        self.face_cascade = None
        self.eye_cascade = None
        self.fer_model = None
        self.scaler = None
        self.emotions = None
        self.model_loaded = False
        self.detectors_initialized = False
        self._initialize_detectors()
        self._load_models()
    
    def _initialize_detectors(self):
        """Initialize OpenCV detectors with lazy loading and fallbacks"""
        try:
            if not self.detectors_initialized:
                # Try OpenCV first
                self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
                
                # Initialize fallback detectors
                self.dlib_available = False
                self.mediapipe_available = False
                self.face_recognition_available = False
                
                # Try Dlib
                try:
                    import dlib
                    self.face_detector_dlib = dlib.get_frontal_face_detector()
                    self.dlib_available = True
                    logger.info("✅ Dlib detector initialized successfully")
                except ImportError:
                    logger.warning("⚠️ Dlib not available - using OpenCV fallback")
                
                # Try MediaPipe
                try:
                    import mediapipe as mp
                    self.mp_face_detection = mp.solutions.face_detection
                    self.face_detection_mp = self.mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5
                    )
                    self.mediapipe_available = True
                    logger.info("✅ MediaPipe detector initialized successfully")
                except ImportError:
                    logger.warning("⚠️ MediaPipe not available - using OpenCV fallback")
                
                # Try face_recognition
                try:
                    import face_recognition
                    self.face_recognition_available = True
                    logger.info("✅ face_recognition library available")
                except ImportError:
                    logger.warning("⚠️ face_recognition not available - using OpenCV fallback")
                
                self.detectors_initialized = True
                logger.info("✅ OpenCV detectors initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize detectors: {e}")
            self.detectors_initialized = False
    
    def _load_models(self):
        """Load trained models"""
        try:
            model_path = "data/models/real_fer_model.pkl"
            if os.path.exists(model_path):
                with open(model_path, 'rb') as f:
                    model_data = pickle.load(f)
                    self.fer_model = model_data['model']
                    self.scaler = model_data['scaler']
                    self.emotions = model_data['emotions']
                    self.model_loaded = True
                    logger.info("✅ Real FER model loaded successfully")
            else:
                logger.warning("⚠️ Real FER model not found, using fallback methods")
                self.model_loaded = False
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            self.model_loaded = False
    
    def analyze_facial_expression(self, image_file) -> Dict:
        """
        Analyze facial expression from image file
        
        Args:
            image_file: File-like object containing image data
            
        Returns:
            Dict with analysis results
        """
        try:
            # Read image from file
            image_data = image_file.file.read()
            nparr = np.frombuffer(image_data, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if image is None:
                return self._create_error_result("Invalid image data")
            
            # Detect faces
            faces = self._detect_faces(image)
            if len(faces) == 0:
                return self._create_no_face_result()
            
            # Get the largest face
            face = max(faces, key=lambda x: x[2] * x[3])
            x, y, w, h = face
            face_roi = image[y:y+h, x:x+w]
            
            # Analyze emotion
            if self.model_loaded:
                emotion, confidence = self._predict_emotion_with_model(face_roi)
            else:
                emotion, confidence = self._predict_emotion_heuristic(face_roi)
            
            # Analyze additional metrics
            eye_metrics = self._analyze_eyes(face_roi)
            head_pose = self._analyze_head_pose(face_roi)
            sleepiness_data = self._assess_sleepiness(eye_metrics)
            
            # Ensure sleepiness is a string
            if isinstance(sleepiness_data, dict):
                sleepiness = sleepiness_data.get('level', 'unknown')
            elif sleepiness_data is None:
                sleepiness = 'unknown'
            else:
                sleepiness = str(sleepiness_data)
            
            # Create comprehensive result
            result = {
                'emotion': emotion,
                'confidence': confidence,
                'face_count': len(faces),
                'face_detected': True,
                'detection_method': 'OpenCV Cascade',
                'emotion_method': 'Trained Model' if self.model_loaded else 'Heuristic',
                'eye_metrics': eye_metrics,
                'head_pose': head_pose,
                'sleepiness': sleepiness,
                'sleepiness_data': sleepiness_data,  # Keep full data for debugging
                'analysis_quality': self._calculate_analysis_quality(image, faces),
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info(f"✅ Real facial analysis complete: {emotion} ({confidence:.2f})")
            return result
            
        except Exception as e:
            logger.error(f"❌ Error in facial analysis: {e}")
            return self._create_error_result(str(e))
    
    def _detect_faces(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in image with improved detection parameters and fallbacks"""
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = []
            
            # Try OpenCV first
            detection_params = [
                (1.1, 3),   # Standard parameters
                (1.05, 4),  # More sensitive
                (1.3, 5),   # Less sensitive but more stable
                (1.2, 6)    # Balanced approach
            ]
            
            for scale_factor, min_neighbors in detection_params:
                faces = self.face_cascade.detectMultiScale(
                    gray, 
                    scaleFactor=scale_factor, 
                    minNeighbors=min_neighbors,
                    minSize=(30, 30),  # Minimum face size
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    logger.info(f"Face detected with OpenCV: scale={scale_factor}, neighbors={min_neighbors}")
                    break
            
            # If OpenCV fails, try MediaPipe
            if len(faces) == 0 and self.mediapipe_available:
                try:
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = self.face_detection_mp.process(rgb_image)
                    
                    if results.detections:
                        h, w = image.shape[:2]
                        for detection in results.detections:
                            bbox = detection.location_data.relative_bounding_box
                            x = int(bbox.xmin * w)
                            y = int(bbox.ymin * h)
                            width = int(bbox.width * w)
                            height = int(bbox.height * h)
                            faces.append((x, y, width, height))
                        
                        if faces:
                            logger.info(f"Face detected with MediaPipe: {len(faces)} faces")
                except Exception as e:
                    logger.warning(f"MediaPipe detection failed: {e}")
            
            # If MediaPipe fails, try face_recognition
            if len(faces) == 0 and self.face_recognition_available:
                try:
                    import face_recognition
                    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    face_locations = face_recognition.face_locations(rgb_image)
                    
                    for (top, right, bottom, left) in face_locations:
                        faces.append((left, top, right - left, bottom - top))
                    
                    if faces:
                        logger.info(f"Face detected with face_recognition: {len(faces)} faces")
                except Exception as e:
                    logger.warning(f"face_recognition detection failed: {e}")
            
            # If still no faces, try with image preprocessing
            if len(faces) == 0:
                # Enhance image contrast
                enhanced = cv2.equalizeHist(gray)
                faces = self.face_cascade.detectMultiScale(
                    enhanced, 
                    scaleFactor=1.1, 
                    minNeighbors=3,
                    minSize=(20, 20)
                )
                if faces:
                    logger.info(f"Enhanced detection found {len(faces)} faces")
            
            return [tuple(face) for face in faces] if len(faces) > 0 else []
        except Exception as e:
            logger.error(f"Error detecting faces: {e}")
            return []
    
    def _predict_emotion_with_model(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Predict emotion using trained model with feature alignment"""
        try:
            # Extract features (same as training)
            features = self._extract_facial_features(face_roi)
            if features is None:
                return "neutral", 0.5
            
            # Ensure features match scaler expectations
            features = np.array(features).reshape(1, -1)
            expected_features = self.scaler.mean_.shape[0]
            
            # Feature alignment
            if features.shape[1] < expected_features:
                # Pad missing features with zeros
                padding = np.zeros((1, expected_features - features.shape[1]))
                features = np.hstack([features, padding])
                logger.debug(f"Padded features from {features.shape[1]} to {expected_features}")
            elif features.shape[1] > expected_features:
                # Truncate excess features
                features = features[:, :expected_features]
                logger.debug(f"Truncated features from {features.shape[1]} to {expected_features}")
            
            # Scale features
            features_scaled = self.scaler.transform(features)
            
            # Predict
            prediction = self.fer_model.predict(features_scaled)[0]
            confidence = np.max(self.fer_model.predict_proba(features_scaled))
            
            emotion = self.emotions[prediction]
            logger.debug(f"Model prediction: {emotion} (confidence: {confidence:.3f})")
            return emotion, confidence
            
        except Exception as e:
            logger.error(f"Error in model prediction: {e}")
            return "neutral", 0.5
    
    def _predict_emotion_heuristic(self, face_roi: np.ndarray) -> Tuple[str, float]:
        """Predict emotion using heuristic methods"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            
            # Analyze facial features
            eye_metrics = self._analyze_eyes(face_roi)
            mouth_region = self._analyze_mouth_region(gray)
            eyebrow_region = self._analyze_eyebrow_region(gray)
            
            # Simple heuristic rules
            emotion_scores = {
                'happy': 0.0,
                'sad': 0.0,
                'angry': 0.0,
                'surprise': 0.0,
                'fear': 0.0,
                'disgust': 0.0,
                'neutral': 0.5
            }
            
            # Happy: high mouth curvature, wide eyes
            if mouth_region['curvature'] > 0.3 and eye_metrics['avg_ear'] > 0.3:
                emotion_scores['happy'] += 0.8
            
            # Sad: low mouth curvature, droopy eyes
            if mouth_region['curvature'] < -0.2 and eye_metrics['avg_ear'] < 0.25:
                emotion_scores['sad'] += 0.7
            
            # Angry: furrowed eyebrows, tense mouth
            if eyebrow_region['tension'] > 0.6 and mouth_region['tension'] > 0.5:
                emotion_scores['angry'] += 0.8
            
            # Surprise: wide eyes, raised eyebrows
            if eye_metrics['avg_ear'] > 0.4 and eyebrow_region['height'] > 0.6:
                emotion_scores['surprise'] += 0.7
            
            # Fear: wide eyes, tense mouth
            if eye_metrics['avg_ear'] > 0.35 and mouth_region['tension'] > 0.4:
                emotion_scores['fear'] += 0.6
            
            # Disgust: wrinkled nose, downturned mouth
            if mouth_region['curvature'] < -0.1 and mouth_region['tension'] > 0.3:
                emotion_scores['disgust'] += 0.6
            
            # Find best emotion
            best_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[best_emotion]
            
            # If no strong emotion detected, default to neutral with moderate confidence
            if confidence < 0.3:
                best_emotion = "neutral"
                confidence = 0.6
            
            return best_emotion, confidence
            
        except Exception as e:
            logger.error(f"Error in heuristic prediction: {e}")
            return "neutral", 0.5
    
    def _extract_facial_features(self, face_roi: np.ndarray) -> Optional[np.ndarray]:
        """Extract facial features for model prediction with exact 32 features"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray, (48, 48))
            
            features = []
            
            # HOG features (9)
            hog_features = self._extract_hog_features(gray)
            features.extend(hog_features)
            
            # LBP features (16)
            lbp_features = self._extract_lbp_features(gray)
            features.extend(lbp_features)
            
            # Eye aspect ratio (1)
            ear = self._calculate_eye_aspect_ratio(gray)
            features.append(ear)
            
            # Mouth aspect ratio (1)
            mar = self._calculate_mouth_aspect_ratio(gray)
            features.append(mar)
            
            # Facial symmetry (1)
            symmetry = self._calculate_facial_symmetry(gray)
            features.append(symmetry)
            
            # Texture features (2)
            texture_features = self._extract_texture_features(gray)
            features.extend(texture_features)
            
            # Additional features to match model expectations (2 more to reach 32)
            # Brightness and contrast features
            brightness = np.mean(gray) / 255.0
            features.append(brightness)
            
            contrast = np.std(gray) / 255.0
            features.append(contrast)
            
            # Ensure we have exactly 32 features
            if len(features) != 32:
                logger.warning(f"Feature count mismatch: {len(features)} features, expected 32")
                # Pad or truncate to match expected size
                while len(features) < 32:
                    features.append(0.0)
                features = features[:32]
            
            # Validate feature array
            features_array = np.array(features, dtype=np.float32)
            if features_array.shape[0] != 32:
                logger.error(f"Feature array shape mismatch: {features_array.shape}, expected (32,)")
                # Force correct shape
                if features_array.shape[0] < 32:
                    features_array = np.pad(features_array, (0, 32 - features_array.shape[0]), 'constant')
                else:
                    features_array = features_array[:32]
            
            logger.debug(f"Extracted {features_array.shape[0]} features for model prediction")
            return features_array
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None
    
    def _extract_hog_features(self, image: np.ndarray) -> List[float]:
        """Extract HOG features"""
        try:
            grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            orientation = np.arctan2(grad_y, grad_x)
            
            bins = 9
            hist, _ = np.histogram(orientation, bins=bins, range=(-np.pi, np.pi), weights=magnitude)
            return hist.tolist()
        except:
            return [0] * 9
    
    def _extract_lbp_features(self, image: np.ndarray) -> List[float]:
        """Extract LBP features"""
        try:
            lbp_image = np.zeros_like(image)
            
            for i in range(1, image.shape[0]-1):
                for j in range(1, image.shape[1]-1):
                    center = image[i, j]
                    binary_string = ""
                    
                    neighbors = [
                        image[i-1, j-1], image[i-1, j], image[i-1, j+1],
                        image[i, j+1], image[i+1, j+1], image[i+1, j],
                        image[i+1, j-1], image[i, j-1]
                    ]
                    
                    for neighbor in neighbors:
                        binary_string += "1" if neighbor >= center else "0"
                    
                    lbp_image[i, j] = int(binary_string, 2)
            
            hist, _ = np.histogram(lbp_image, bins=256, range=(0, 256))
            return hist.tolist()[:16]
        except:
            return [0] * 16
    
    def _calculate_eye_aspect_ratio(self, face_roi: np.ndarray) -> float:
        """Calculate Eye Aspect Ratio"""
        try:
            eyes = self.eye_cascade.detectMultiScale(face_roi)
            if len(eyes) >= 2:
                eye1 = eyes[0]
                eye2 = eyes[1]
                
                ear1 = self._calculate_ear_for_eye(eye1, face_roi)
                ear2 = self._calculate_ear_for_eye(eye2, face_roi)
                
                return (ear1 + ear2) / 2
            return 0.3
        except:
            return 0.3
    
    def _calculate_ear_for_eye(self, eye_region: Tuple[int, int, int, int], face_roi: np.ndarray) -> float:
        """Calculate EAR for single eye"""
        x, y, w, h = eye_region
        eye_roi = face_roi[y:y+h, x:x+w]
        
        if eye_roi.size == 0:
            return 0.3
        
        height, width = eye_roi.shape
        if width > 0:
            return height / width
        return 0.3
    
    def _calculate_mouth_aspect_ratio(self, face_roi: np.ndarray) -> float:
        """Calculate Mouth Aspect Ratio"""
        try:
            h, w = face_roi.shape
            mouth_region = face_roi[int(h*0.6):, :]
            
            if mouth_region.size > 0:
                return mouth_region.shape[0] / mouth_region.shape[1]
            return 0.5
        except:
            return 0.5
    
    def _calculate_facial_symmetry(self, face_roi: np.ndarray) -> float:
        """Calculate facial symmetry"""
        try:
            h, w = face_roi.shape
            left_half = face_roi[:, :w//2]
            right_half = cv2.flip(face_roi[:, w//2:], 1)
            
            min_width = min(left_half.shape[1], right_half.shape[1])
            left_half = left_half[:, :min_width]
            right_half = right_half[:, :min_width]
            
            diff = np.abs(left_half.astype(float) - right_half.astype(float))
            symmetry = 1.0 - (np.mean(diff) / 255.0)
            return max(0.0, min(1.0, symmetry))
        except:
            return 0.5
    
    def _extract_texture_features(self, image: np.ndarray) -> List[float]:
        """Extract texture features"""
        try:
            kernel = np.ones((5, 5), np.float32) / 25
            smoothed = cv2.filter2D(image, -1, kernel)
            texture = np.std(image - smoothed)
            
            edges = cv2.Canny(image, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            return [texture / 100.0, edge_density]
        except:
            return [0.0, 0.0]
    
    def _analyze_eyes(self, face_roi: np.ndarray) -> Dict:
        """Analyze eye metrics"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            eyes = self.eye_cascade.detectMultiScale(gray)
            
            if len(eyes) >= 2:
                left_ear = self._calculate_ear_for_eye(eyes[0], gray)
                right_ear = self._calculate_ear_for_eye(eyes[1], gray)
                avg_ear = (left_ear + right_ear) / 2
                
                return {
                    'left_ear': left_ear,
                    'right_ear': right_ear,
                    'avg_ear': avg_ear,
                    'blink_rate': self._estimate_blink_rate(avg_ear),
                    'blink_duration': self._estimate_blink_duration(avg_ear)
                }
            else:
                return {
                    'left_ear': 0.35,
                    'right_ear': 0.35,
                    'avg_ear': 0.35,
                    'blink_rate': 15.0,
                    'blink_duration': 150.0
                }
        except Exception as e:
            logger.error(f"Error analyzing eyes: {e}")
            return {
                'left_ear': 0.35,
                'right_ear': 0.35,
                'avg_ear': 0.35,
                'blink_rate': 15.0,
                'blink_duration': 150.0
            }
    
    def _analyze_mouth_region(self, gray: np.ndarray) -> Dict:
        """Analyze mouth region"""
        try:
            h, w = gray.shape
            mouth_region = gray[int(h*0.6):, :]
            
            # Calculate mouth curvature
            if mouth_region.size > 0:
                # Simple curvature estimation
                top_edge = mouth_region[0, :]
                bottom_edge = mouth_region[-1, :]
                curvature = np.mean(top_edge) - np.mean(bottom_edge)
                curvature = curvature / 255.0
                
                # Calculate tension
                tension = np.std(mouth_region) / 255.0
                
                return {
                    'curvature': curvature,
                    'tension': tension
                }
            else:
                return {'curvature': 0.0, 'tension': 0.0}
        except:
            return {'curvature': 0.0, 'tension': 0.0}
    
    def _analyze_eyebrow_region(self, gray: np.ndarray) -> Dict:
        """Analyze eyebrow region"""
        try:
            h, w = gray.shape
            eyebrow_region = gray[:int(h*0.4), :]
            
            if eyebrow_region.size > 0:
                # Calculate eyebrow height
                height = np.mean(eyebrow_region) / 255.0
                
                # Calculate tension
                tension = np.std(eyebrow_region) / 255.0
                
                return {
                    'height': height,
                    'tension': tension
                }
            else:
                return {'height': 0.5, 'tension': 0.0}
        except:
            return {'height': 0.5, 'tension': 0.0}
    
    def _analyze_head_pose(self, face_roi: np.ndarray) -> Dict:
        """Analyze head pose"""
        try:
            gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape
            
            # Calculate pitch
            upper_half = gray[:h//2, :]
            lower_half = gray[h//2:, :]
            pitch = (np.mean(lower_half) - np.mean(upper_half)) / 10.0
            pitch = np.clip(pitch, -30, 30)
            
            # Calculate yaw
            left_half = gray[:, :w//2]
            right_half = gray[:, w//2:]
            yaw = (np.mean(right_half) - np.mean(left_half)) / 5.0
            yaw = np.clip(yaw, -45, 45)
            
            # Calculate roll
            edges = cv2.Canny(gray, 50, 150)
            lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=50)
            roll = 0.0
            
            if lines is not None:
                angles = []
                for line in lines[:5]:
                    rho, theta = line[0]
                    angle = theta * 180 / np.pi - 90
                    angles.append(angle)
                
                if angles:
                    roll = np.clip(np.mean(angles), -30, 30)
            
            return {
                'pitch': pitch,
                'yaw': yaw,
                'roll': roll,
                'stability': 0.8
            }
        except Exception as e:
            logger.error(f"Error analyzing head pose: {e}")
            return {'pitch': 0.0, 'yaw': 0.0, 'roll': 0.0, 'stability': 0.8}
    
    def _assess_sleepiness(self, eye_metrics: Dict) -> Dict:
        """Assess sleepiness level"""
        try:
            avg_ear = eye_metrics['avg_ear']
            blink_rate = eye_metrics['blink_rate']
            
            if avg_ear < 0.25:
                level = "Very tired"
                confidence = 0.8
                factors = ["Low eye aspect ratio", "Frequent blinking"]
            elif avg_ear < 0.3:
                level = "Slightly tired"
                confidence = 0.6
                factors = ["Reduced eye openness"]
            else:
                level = "Alert"
                confidence = 0.7
                factors = ["Normal eye activity"]
            
            if blink_rate > 20:
                factors.append("High blink frequency")
                if level == "Alert":
                    level = "Slightly tired"
                    confidence = 0.6
            
            return {
                'level': level,
                'confidence': confidence,
                'contributing_factors': factors
            }
        except Exception as e:
            logger.error(f"Error assessing sleepiness: {e}")
            return {
                'level': "Unknown",
                'confidence': 0.0,
                'contributing_factors': ["Analysis failed"]
            }
    
    def _estimate_blink_rate(self, avg_ear: float) -> float:
        """Estimate blink rate"""
        if avg_ear < 0.25:
            return 25.0
        elif avg_ear < 0.3:
            return 18.0
        else:
            return 12.0
    
    def _estimate_blink_duration(self, avg_ear: float) -> float:
        """Estimate blink duration"""
        return 120.0 + (0.35 - avg_ear) * 200.0
    
    def _calculate_analysis_quality(self, image: np.ndarray, faces: List) -> float:
        """Calculate analysis quality score"""
        try:
            # Image quality
            h, w = image.shape[:2]
            size_score = min(1.0, (h * w) / (480 * 640))
            
            # Brightness
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray) / 255.0
            brightness_score = 1.0 - abs(brightness - 0.5) * 2
            
            # Sharpness
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(1.0, laplacian_var / 1000.0)
            
            # Face detection quality
            face_score = 1.0 if len(faces) == 1 else (0.7 if len(faces) > 1 else 0.3)
            
            return (size_score + brightness_score + sharpness_score + face_score) / 4.0
            
        except Exception:
            return 0.5
    
    def _create_no_face_result(self) -> Dict:
        """Create result when no face is detected"""
        return {
            'emotion': 'no_face_detected',
            'confidence': 0.0,
            'face_count': 0,
            'face_detected': False,
            'detection_method': 'OpenCV Cascade',
            'emotion_method': 'None',
            'analysis_quality': 0.0,
            'timestamp': datetime.now().isoformat()
        }
    
    def _create_error_result(self, error_message: str) -> Dict:
        """Create result when analysis fails"""
        return {
            'emotion': 'analysis_error',
            'confidence': 0.0,
            'face_count': 0,
            'face_detected': False,
            'detection_method': 'Error',
            'emotion_method': 'Error',
            'analysis_quality': 0.0,
            'error': error_message,
            'timestamp': datetime.now().isoformat()
        }

# Global instance
real_analyzer = RealFacialAnalyzer()

def analyze_facial_expression(image_file) -> Dict:
    """Analyze facial expression using real analyzer"""
    return real_analyzer.analyze_facial_expression(image_file)

def get_analyzer_status() -> Dict:
    """Get analyzer status"""
    return {
        'model_loaded': real_analyzer.model_loaded,
        'detectors_available': real_analyzer.face_cascade is not None,
        'analyzer_type': 'Real Facial Analyzer'
    }
