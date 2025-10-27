# MindGuard - AI-Powered Mental Health Assessment Platform

## 🧠 Project Overview

MindGuard is a comprehensive AI-powered mental health risk prediction system that combines multiple machine learning models to provide personalized health insights and recommendations. The platform uses advanced facial expression analysis, audio emotion detection, and traditional health assessment questionnaires to deliver comprehensive mental health monitoring.

## 🚀 Key Features

### Core Functionality
- **Multi-Modal Health Assessment**: Combines PHQ-9 questionnaires, facial expression analysis, and audio emotion detection
- **Real-Time Facial Analysis**: Advanced computer vision for emotion, sleepiness, fatigue, and stress detection
- **AI-Powered Risk Prediction**: GradientBoostingClassifier for accurate health risk assessment
- **Personalized Recommendations**: Intelligent recommendation engine for health activities and interventions
- **Comprehensive Dashboard**: Interactive health charts and trend analysis
- **Multi-Language Support**: English, Spanish, and French localization
- **Secure Authentication**: JWT-based user management with bcrypt password hashing

### Advanced AI Capabilities
- **Facial Expression Analysis**: Real-time emotion detection with micro-expression analysis
- **PHQ-9 Estimation**: Automated depression screening score estimation from facial features
- **Audio Emotion Analysis**: Voice-based emotion and stress level detection
- **Biometric Monitoring**: Eye tracking, head pose analysis, and fatigue detection
- **Sleepiness Assessment**: Advanced algorithms for detecting sleep deprivation indicators

## 🤖 AI Models and Technologies

### 1. Health Risk Prediction Model
- **Algorithm**: GradientBoostingClassifier (scikit-learn)
- **Features**: 14 comprehensive health indicators including age, gender, sleep hours, exercise frequency, stress levels, diet quality, social connections, work-life balance, mental/physical health history, substance use, family history, financial stress, and relationship status
- **Output**: Risk levels (Low, Normal, High) with confidence scores
- **Accuracy**: Optimized for high accuracy with feature importance analysis

### 2. Facial Expression Analysis
- **Computer Vision**: OpenCV with MediaPipe integration
- **Emotion Detection**: 7 primary emotions (happy, sad, angry, fear, surprise, disgust, neutral)
- **Advanced Metrics**:
  - Eye Aspect Ratio (EAR) for blink detection
  - Micro-expression analysis for stress detection
  - Head pose estimation for fatigue assessment
  - Sleepiness level calculation
  - PHQ-9 score estimation from facial features

### 3. Audio Emotion Analysis
- **Speech Processing**: Librosa and Whisper integration
- **Emotion Recognition**: Voice-based emotion detection
- **Stress Detection**: Audio stress level analysis
- **Real-time Processing**: Live audio chunk analysis

### 4. Deep Learning Models
- **PyTorch Neural Networks**: Custom RiskModel architecture
- **TensorFlow Integration**: Advanced deep learning capabilities
- **Multi-Modal Fusion**: Combining facial, audio, and questionnaire data

## 🛠️ Technical Stack

### Backend (FastAPI + Python)
```
Core Framework: FastAPI with Uvicorn ASGI server
Database: MongoDB with Motor async driver
Authentication: JWT with passlib/bcrypt
Machine Learning: scikit-learn, PyTorch, TensorFlow
Computer Vision: OpenCV, MediaPipe, FER
Audio Processing: Librosa, Whisper
Data Processing: pandas, numpy, joblib
```

### Frontend (Next.js + TypeScript)
```
Framework: Next.js 14 with App Router
Language: TypeScript with strict type checking
Styling: TailwindCSS with Radix UI components
State Management: Zustand for global state
Charts: Recharts for interactive visualizations
Internationalization: next-intl for multi-language support
```

## 📊 API Endpoints

### Authentication & User Management
```
POST /api/auth/register          - User registration
POST /api/auth/login             - User authentication
POST /api/auth/logout            - User logout
GET  /api/auth/me                - Get current user info
POST /api/auth/refresh           - Refresh access token
```

### Health Assessment
```
POST /api/assessment/submit      - Submit health assessment
GET  /api/assessment/history     - Get assessment history
GET  /api/assessment/{id}        - Get specific assessment
```

### AI Prediction & Analysis
```
POST /api/predictions            - ML-based health risk prediction
GET  /api/predictions/history    - Prediction history
POST /api/predict                - Simple risk prediction
```

### Facial Analysis
```
POST /api/facial-analysis/analyze           - Real-time facial analysis
GET  /api/facial-analysis/session/{id}      - Get analysis session
POST /api/facial-dashboard/analyze          - Comprehensive facial assessment
GET  /api/facial-dashboard/sessions         - Get all analysis sessions
```

### Audio Analysis
```
POST /api/audio-analysis/analyze            - Audio emotion analysis
GET  /api/audio-analysis/health             - Audio analysis health check
```

### Dashboard & Analytics
```
GET  /api/dashboard/data                    - Dashboard data
GET  /api/dashboard/trends                  - Health trends
GET  /api/dashboard/insights                - Personalized insights
```

### Recommendations
```
GET  /api/recommendations                   - Personalized recommendations
GET  /api/recommendations/brain-heal        - Brain healing activities
```

### Global Statistics
```
GET  /api/global-stats                      - Global health statistics
GET  /api/global-stats/demographics         - Demographic breakdowns
```

### Admin & Research
```
GET  /api/admin/users                       - User management
GET  /api/research/insights                 - Research insights
GET  /api/health                           - System health check
```

## 🏃‍♂️ How to Run the Application

### Prerequisites
- Python 3.8+
- Node.js 18+
- MongoDB (local or cloud)
- Git

### Quick Start

1. **Clone the Repository**
```bash
git clone <repository-url>
cd mindgaurd-full
```

2. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. **Frontend Setup**
```bash
cd frontend
npm install
```

4. **Environment Configuration**
Create `.env` file in backend directory:
```env
# Database
MONGO_URI=mongodb://localhost:27017
MONGO_DB=mindguard_db

# Security
SECRET_KEY=your_secret_key_here
JWT_SECRET_KEY=your_jwt_secret_here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true
```

Create `.env.local` file in frontend directory:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

5. **Initialize Database**
```bash
cd backend
python scripts/init_db.py
```

6. **Train AI Models**
```bash
python scripts/train_model.py
python scripts/train_facial_model.py
python scripts/train_audio_model.py
```

7. **Start the Application**
```bash
# From project root
make up
# OR manually:
# Backend: cd backend && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
# Frontend: cd frontend && npm run dev
```

### Access Points
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📱 Application Pages and Features

### Main Pages
1. **Home Page** (`/`) - Landing page with assessment and dashboard access
2. **Authentication** (`/auth`) - Login and registration
3. **Assessment** (`/assessment`) - Multi-step health questionnaire with facial analysis
4. **Dashboard** (`/dashboard`) - Personal health dashboard with charts and insights
5. **Facial Dashboard** (`/facial-dashboard`) - Real-time facial expression analysis
6. **Health Dashboard** (`/health-dashboard`) - Comprehensive health monitoring
7. **History** (`/history`) - Assessment and prediction history
8. **Global Stats** (`/global-stats`) - Global health statistics and demographics
9. **Profile** (`/profile`) - User profile management
10. **Settings** (`/settings`) - Application settings and preferences

### Key Components
- **RealTimeFacialDashboard**: Live facial analysis with emotion detection
- **AssessmentForm**: Multi-step health assessment wizard
- **InteractiveHealthChart**: Dynamic health trend visualization
- **FacialExpressionAnalysis**: Camera-based emotion detection
- **LanguageSelector**: Multi-language support
- **Header/Footer**: Navigation and branding

## 🔬 Research and Development

### Datasets Used
1. **OSMI Mental Health in Tech Survey** (Kaggle)
2. **PHQ-9 Depression Screening Dataset** (Kaggle)
3. **Sleep Health Dataset** (Kaggle)
4. **Custom 14-day Depression Symptoms Dataset**

### Model Performance
- **Health Risk Prediction**: GradientBoostingClassifier with optimized hyperparameters
- **Facial Emotion Detection**: Multi-modal analysis with 7 emotion categories
- **PHQ-9 Estimation**: Automated scoring from facial features
- **Real-time Processing**: Optimized for low-latency analysis

### Advanced Features
- **Micro-expression Analysis**: Detection of subtle facial movements
- **Biometric Monitoring**: Eye tracking and head pose estimation
- **Multi-modal Fusion**: Combining questionnaire, facial, and audio data
- **Personalized Recommendations**: AI-driven health intervention suggestions

## 🔒 Security and Privacy

- **JWT Authentication**: Secure token-based authentication
- **Password Hashing**: bcrypt with salt for secure password storage
- **CORS Configuration**: Proper cross-origin resource sharing setup
- **Input Validation**: Comprehensive data validation and sanitization
- **Privacy-First Design**: Anonymous assessment options available

## 🌐 Internationalization

Supported Languages:
- **English** (en) - Default
- **Spanish** (es) - Español  
- **French** (fr) - Français

Language switching is handled through the `useTranslations` hook and comprehensive translation files.

## 📈 Future Roadmap

- [ ] Advanced deep learning model integration
- [ ] Mobile application (React Native)
- [ ] Real-time notifications and alerts
- [ ] Integration with health devices and wearables
- [ ] Advanced analytics and research dashboard
- [ ] Telemedicine integration
- [ ] Multi-user collaborative features
- [ ] Advanced privacy controls and data encryption

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For issues and support:
1. Check the API documentation at `/docs` endpoint
2. Review backend logs for error details
3. Check browser console for frontend errors
4. Create an issue in the repository

---

**Built with ❤️ for better mental health awareness and support**

*MindGuard combines cutting-edge AI technology with user-friendly design to provide comprehensive mental health monitoring and personalized recommendations.*
