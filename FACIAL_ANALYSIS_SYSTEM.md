# 🧠 MindGuard Facial Analysis & Mood Recognition System

## Overview

The MindGuard Facial Analysis System is a comprehensive AI-powered solution that provides real-time facial expression analysis, mood recognition, sleepiness detection, fatigue assessment, stress level analysis, and PHQ-9 depression screening estimation. The system integrates seamlessly with the existing MindGuard mental health platform.

## 🎯 Key Features

### 1. Real-Time Facial Expression Analysis
- **Emotion Detection**: Identifies 7 primary emotions (happy, sad, angry, fear, surprise, disgust, neutral)
- **Confidence Scoring**: Provides confidence levels for each emotion detection
- **Multi-modal Analysis**: Combines facial landmarks, micro-expressions, and head pose

### 2. Sleepiness & Fatigue Detection
- **Eye Aspect Ratio (EAR)**: Calculates eye openness to detect drowsiness
- **Blink Rate Analysis**: Monitors blink frequency and duration
- **Head Pose Tracking**: Detects head droop and nodding patterns
- **Fatigue Indicators**: Identifies yawning and micro-sleep patterns

### 3. Stress Level Assessment
- **Micro-expression Analysis**: Detects subtle facial muscle movements
- **Facial Asymmetry**: Measures stress-related facial tension
- **Muscle Tension Scoring**: Quantifies facial muscle stress indicators
- **Stress Classification**: Low, Medium, High stress levels

### 4. PHQ-9 Integration
- **Auto-fill Functionality**: Automatically estimates PHQ-9 questionnaire responses
- **Question Mapping**: Maps facial indicators to specific PHQ-9 questions
- **Confidence Scoring**: Provides reasoning and confidence for each response
- **Severity Assessment**: Estimates depression severity levels

## 🏗️ System Architecture

### Backend Components (Python FastAPI)

#### Core Services
- **`EnhancedFacialAnalyzer`**: Main analysis engine with comprehensive metrics
- **`PHQ9IntegrationService`**: PHQ-9 auto-fill and validation service
- **`HealthRecommendationService`**: Generates personalized recommendations

#### API Endpoints
```
/api/facial-analysis/analyze          # Basic emotion analysis
/api/facial-dashboard/analyze         # Comprehensive analysis
/api/facial-dashboard/analyze-enhanced # Enhanced analysis with recommendations
/api/phq9-integration/auto-fill       # PHQ-9 auto-fill
/api/phq9-integration/validate        # PHQ-9 response validation
```

#### Data Models
- **`ComprehensiveFacialAnalysis`**: Complete analysis result
- **`EyeMetrics`**: Eye aspect ratio and blink analysis
- **`HeadPoseMetrics`**: Head orientation and stability
- **`MicroExpressionMetrics`**: Stress and tension indicators
- **`PHQ9Estimation`**: Depression screening scores

### Frontend Components (Next.js/React)

#### Core Components
- **`RealTimeFacialDashboard`**: Live analysis dashboard with webcam integration
- **`PHQ9FacialIntegration`**: PHQ-9 auto-fill interface
- **`FacialDashboardPage`**: Main dashboard page with comprehensive features

#### Features
- **Real-time Webcam Integration**: Live facial capture and analysis
- **Interactive Dashboards**: Real-time metrics and trend visualization
- **Session Management**: Track analysis sessions and progress
- **Responsive Design**: Mobile-friendly interface

## 🔧 Technical Implementation

### AI/ML Models

#### Emotion Recognition
- **Primary**: FER (Facial Expression Recognition) library
- **Fallback**: Custom OpenCV-based emotion detection
- **Enhancement**: DeepFace integration for improved accuracy

#### Sleepiness Detection
- **Eye Aspect Ratio (EAR)**: Mathematical calculation of eye openness
- **Blink Detection**: Frequency and duration analysis
- **Head Pose Estimation**: Pitch, yaw, and roll angle calculation

#### Stress Analysis
- **Micro-expression Detection**: Subtle facial movement analysis
- **Muscle Tension**: Edge detection and facial asymmetry
- **Stress Indicators**: Multi-factor stress level assessment

### Data Processing Pipeline

1. **Image Capture**: Webcam stream capture at 3-second intervals
2. **Preprocessing**: Image normalization and quality assessment
3. **Face Detection**: OpenCV Haar cascades for face localization
4. **Feature Extraction**: Facial landmarks and micro-expressions
5. **AI Analysis**: Multi-task neural network processing
6. **Result Integration**: Combine all metrics into comprehensive analysis
7. **Recommendation Generation**: Personalized insights and suggestions

## 📊 Analysis Metrics

### Facial Expression Metrics
- **Primary Emotion**: Dominant emotional state
- **Emotion Distribution**: Probability scores for all emotions
- **Confidence Level**: Analysis reliability score
- **Face Detection**: Boolean face presence indicator

### Physiological Indicators
- **Eye Aspect Ratio**: 0.0-1.0 scale (normal: 0.25-0.35)
- **Blink Rate**: Blinks per minute
- **Head Angles**: Pitch, yaw, roll in degrees
- **Muscle Tension**: 0.0-1.0 stress indicator

### Health Assessments
- **Sleepiness Level**: Alert, Slightly tired, Very tired
- **Fatigue Detection**: Yawning, head droop, overall fatigue
- **Stress Level**: Low, Medium, High
- **PHQ-9 Score**: 0-27 depression screening scale

## 🚀 Getting Started

### Backend Setup

1. **Install Dependencies**:
```bash
cd backend
pip install -r requirements.txt
```

2. **Start the Server**:
```bash
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

3. **Train Enhanced Models** (Optional):
```bash
python scripts/train_facial_model.py --epochs 50 --batch_size 32
```

### Frontend Setup

1. **Install Dependencies**:
```bash
cd frontend
npm install
```

2. **Start Development Server**:
```bash
npm run dev
```

3. **Access the Dashboard**:
Navigate to `http://localhost:3000/facial-dashboard`

## 📱 Usage Guide

### Real-Time Analysis

1. **Enable Camera Access**: Grant permission when prompted
2. **Start Analysis**: Click "Start Analysis" to begin real-time monitoring
3. **View Results**: Monitor live metrics in the dashboard
4. **Session Tracking**: View session duration and analysis quality

### PHQ-9 Auto-Fill

1. **Navigate to PHQ-9 Integration**: Access the PHQ-9 facial analysis tool
2. **Position Face**: Ensure good lighting and clear face visibility
3. **Analyze**: Click "Analyze & Auto-Fill" to generate responses
4. **Review Results**: Check estimated scores and reasoning
5. **Validate**: Use the validation endpoint to verify responses

### Dashboard Features

- **Live Metrics**: Real-time emotion, sleepiness, and stress monitoring
- **Trend Analysis**: Historical data visualization
- **Session Management**: Track analysis sessions and progress
- **Insights & Recommendations**: Personalized health suggestions

## 🔒 Privacy & Security

### Data Protection
- **No Video Recording**: Analysis only, no video storage
- **Local Processing**: Maximum processing done locally
- **Encrypted Transmission**: Secure data transfer
- **Optional Storage**: User-controlled data persistence

### Privacy Features
- **Temporary Analysis**: Results can be session-only
- **User Control**: Full control over data storage
- **Secure APIs**: Authentication and authorization
- **GDPR Compliance**: Data protection compliance

## 🎯 Use Cases

### Mental Health Monitoring
- **Daily Mood Tracking**: Regular emotional state assessment
- **Stress Management**: Real-time stress level monitoring
- **Sleep Quality**: Sleepiness and fatigue detection
- **Depression Screening**: PHQ-9 score estimation

### Healthcare Applications
- **Telemedicine**: Remote mental health assessment
- **Clinical Screening**: Pre-consultation health indicators
- **Patient Monitoring**: Continuous health status tracking
- **Research**: Mental health data collection and analysis

### Wellness & Self-Care
- **Personal Insights**: Understanding emotional patterns
- **Lifestyle Optimization**: Data-driven health recommendations
- **Progress Tracking**: Mental health improvement monitoring
- **Early Intervention**: Early warning system for health issues

## 🔧 Configuration

### Environment Variables
```bash
# Database
MONGODB_URL=mongodb://localhost:27017/mindguard

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Model Configuration
- **Analysis Interval**: 3 seconds (configurable)
- **Image Quality Threshold**: 0.5 minimum
- **Confidence Threshold**: 0.6 minimum for reliable results
- **Session Timeout**: 30 minutes default

## 📈 Performance Metrics

### Analysis Speed
- **Processing Time**: < 2 seconds per analysis
- **Real-time Capability**: 3-second intervals
- **Concurrent Users**: 100+ simultaneous analyses
- **Accuracy**: 85%+ emotion recognition accuracy

### System Requirements
- **CPU**: 2+ cores recommended
- **RAM**: 4GB+ for optimal performance
- **GPU**: Optional, for enhanced model training
- **Camera**: 720p+ webcam for best results

## 🛠️ Troubleshooting

### Common Issues

1. **Camera Permission Denied**:
   - Ensure browser permissions are granted
   - Check camera availability
   - Try different browsers

2. **Low Analysis Quality**:
   - Improve lighting conditions
   - Position face clearly in frame
   - Ensure stable internet connection

3. **Model Loading Errors**:
   - Check dependencies installation
   - Verify model files exist
   - Restart the application

### Performance Optimization

1. **Reduce Analysis Frequency**: Increase interval between analyses
2. **Lower Image Quality**: Reduce camera resolution
3. **Disable Advanced Features**: Turn off micro-expression analysis
4. **Use GPU Acceleration**: Enable CUDA if available

## 🔮 Future Enhancements

### Planned Features
- **Multi-language Support**: International language support
- **Advanced Analytics**: Machine learning insights
- **Integration APIs**: Third-party health platform integration
- **Mobile Apps**: Native mobile applications

### Research Areas
- **Improved Accuracy**: Enhanced emotion recognition models
- **New Metrics**: Additional health indicators
- **Real-time Processing**: Faster analysis algorithms
- **Privacy Enhancement**: On-device processing capabilities

## 📚 API Documentation

### Comprehensive Analysis Endpoint
```http
POST /api/facial-dashboard/analyze
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "session_id": "optional_session_id",
  "user_id": "optional_user_id",
  "include_raw_metrics": true,
  "include_phq9_estimation": true
}
```

### PHQ-9 Auto-Fill Endpoint
```http
POST /api/phq9-integration/auto-fill
Content-Type: application/json

{
  "image": "base64_encoded_image",
  "user_id": "user_id",
  "session_id": "optional_session_id"
}
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

### Code Standards
- **Python**: PEP 8 compliance
- **TypeScript**: ESLint configuration
- **Documentation**: Comprehensive docstrings
- **Testing**: Unit and integration tests

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For technical support or questions:
- **Documentation**: Check this file and inline comments
- **Issues**: Create GitHub issues for bugs
- **Discussions**: Use GitHub discussions for questions
- **Email**: Contact the development team

---

**Note**: This facial analysis system is designed for wellness monitoring and should not replace professional medical diagnosis or treatment. Always consult with healthcare providers for accurate medical assessment.
