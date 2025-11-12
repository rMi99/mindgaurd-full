# MindGuard Enhanced AI System

A Next.js + Python-based adaptive facial analysis system with AI model accuracy tuning, automatic overfitting detection, and dynamic model switching for optimal performance.

## 🎯 Key Features

### ✅ AI Model Accuracy Tuning
- **Adaptive Accuracy Monitoring**: Automatically detects and adjusts accuracy thresholds when overfitting (100% accuracy) is detected using live data variance analysis
- **Real-time Performance Optimization**: Continuous monitoring of model performance with automatic adjustments
- **Overfitting Prevention**: Intelligent detection of overfitting patterns and automatic countermeasures

### 🧍‍♀️ Real-Time Face Analysis
- **Live Camera Feed**: Real-time facial expression analysis using MediaPipe, OpenCV, and TensorFlow.js
- **Multi-Modal Analysis**: Comprehensive analysis of emotions, focus, sleepiness, fatigue, and stress levels
- **PHQ-9 Integration**: AI-powered estimation of PHQ-9 depression screening scores

### 🧩 AI Model Mechanism & Design Patterns

#### Backend Architecture (Python FastAPI)
- **Model-View-Controller (MVC)**: Clean separation of concerns for scalable architecture
- **Observer Pattern**: Real-time updates between AI inference layer and UI components
- **Factory Pattern**: Dynamic loading and switching of ML models (CNN, MobileNet, ResNet, EfficientNet)
- **Strategy Pattern**: Multiple optimization strategies (dropout tuning, early stopping, adaptive learning rates)

#### Frontend Architecture (Next.js)
- **Component-based Architecture**: React Hooks for state management
- **WebSocket Integration**: Real-time communication with backend for live updates
- **Atomic Design Pattern**: Reusable UI components (face preview, accuracy meter, logs)

### 🖥️ Technology Stack

#### AI Model Technologies
- **TensorFlow / PyTorch**: Model training and inference
- **OpenCV**: Real-time video stream handling
- **Scikit-learn**: Accuracy monitoring and adaptive correction
- **NumPy / Pandas**: Analytics and data processing

#### Backend Technologies
- **FastAPI**: High-performance API framework with async support
- **Pydantic**: Data validation and serialization
- **WebSockets**: Real-time bidirectional communication
- **MongoDB**: Document-based database for user data and model stats
- **Redis**: Caching and session management

#### Frontend Technologies
- **Next.js 14**: React framework with App Router
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **React Webcam**: Camera integration
- **WebSocket Client**: Real-time communication

#### Deployment Technologies
- **Docker**: Containerization for all services
- **NGINX**: Reverse proxy with WebSocket support
- **GitHub Actions**: CI/CD pipeline
- **AWS**: Cloud hosting and model storage

## 🚀 Quick Start

### Prerequisites
- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.12+ (for local development)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/mindguard-enhanced.git
cd mindguard-enhanced
```

2. **Start with Docker Compose**
```bash
# Start all services
docker-compose -f docker-compose.enhanced.yml up -d

# Or start with specific profiles
docker-compose -f docker-compose.enhanced.yml --profile training up -d
docker-compose -f docker-compose.enhanced.yml --profile monitoring up -d
```

3. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Enhanced Dashboard: http://localhost:3000/enhanced-facial-dashboard

### Local Development

1. **Backend Setup**
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

2. **Frontend Setup**
```bash
cd frontend
npm install
npm run dev
```

## 🏗️ Architecture Overview

### System Components

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   AI Models     │
│   (Next.js)     │◄──►│   (FastAPI)     │◄──►│   (TensorFlow)  │
│                 │    │                 │    │                 │
│ • WebSocket     │    │ • Observer      │    │ • CNN           │
│ • Real-time UI  │    │ • Factory       │    │ • MobileNet     │
│ • Camera Feed   │    │ • Strategy      │    │ • ResNet        │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   NGINX         │    │   MongoDB       │    │   Redis         │
│   (Proxy)       │    │   (Database)    │    │   (Cache)       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Design Patterns Implementation

#### 1. Observer Pattern
```python
# Real-time updates between components
class AccuracyMonitor(Subject):
    def add_accuracy_reading(self, accuracy: float):
        # Analyze patterns and notify observers
        analysis = self._analyze_accuracy_patterns()
        if analysis['status'] != 'normal':
            self.notify({'type': 'accuracy_analysis', 'analysis': analysis})
```

#### 2. Factory Pattern
```python
# Dynamic model creation and switching
class FacialModelFactory(ModelFactory):
    def create_model(self, model_type: ModelType) -> AIModel:
        if model_type == ModelType.CNN:
            return CNNModel()
        elif model_type == ModelType.MOBILENET:
            return MobileNetModel()
        # ... other models
```

#### 3. Strategy Pattern
```python
# Multiple optimization strategies
class DropoutTuningStrategy(OptimizationStrategy):
    def optimize(self, model: AIModel, data: Dict) -> Dict:
        if model.get_accuracy() > 0.95:  # Overfitting detected
            return {'action': 'increase_dropout', 'new_dropout': 0.5}
```

## 📊 API Endpoints

### Core Analysis Endpoints
- `POST /api/facial-analysis/analyze` - Analyze facial expression with adaptive tuning
- `POST /api/facial-analysis/analyze-file` - Analyze uploaded image file
- `GET /api/facial-analysis/health` - Service health check

### Model Management
- `POST /api/facial-analysis/model/switch` - Switch to different AI model
- `GET /api/facial-analysis/model/status` - Get current model status
- `GET /api/facial-analysis/model/supported` - Get supported model types

### Session Management
- `POST /api/facial-analysis/session/start` - Start analysis session
- `POST /api/facial-analysis/session/{id}/stop` - Stop analysis session
- `GET /api/facial-analysis/session/{id}/status` - Get session status

### WebSocket Endpoints
- `WS /api/facial-analysis/ws/{client_id}` - Real-time WebSocket connection

## 🔧 Configuration

### Environment Variables

#### Backend
```bash
PYTHONPATH=/app
ENVIRONMENT=production
LOG_LEVEL=INFO
MONGODB_URL=mongodb://mongo:27017/mindguard
REDIS_URL=redis://redis:6379
MODEL_CACHE_SIZE=1000
ACCURACY_MONITORING_ENABLED=true
WEBSOCKET_ENABLED=true
```

#### Frontend
```bash
NODE_ENV=production
NEXT_PUBLIC_API_URL=http://localhost:8000
NEXT_PUBLIC_WS_URL=ws://localhost:8000
NEXT_PUBLIC_ADAPTIVE_MODE=true
```

### Model Configuration
```python
# Accuracy monitoring settings
ACCURACY_WINDOW_SIZE = 50
OVERFITTING_THRESHOLD = 0.95
UNDERFITTING_THRESHOLD = 0.7
VARIANCE_THRESHOLD = 0.01

# Optimization strategy settings
DROPOUT_RANGE = (0.1, 0.5)
EARLY_STOPPING_PATIENCE = 5
LEARNING_RATE_DECAY = 0.5
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v
pytest tests/ --cov=app --cov-report=html
```

### Frontend Tests
```bash
cd frontend
npm test
npm run test:coverage
```

### Integration Tests
```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/ -v
```

## 📈 Monitoring and Observability

### Health Checks
- Backend: `GET /health` - Returns service status and metrics
- Frontend: `GET /` - Returns application status
- Database: MongoDB connection health
- Cache: Redis connection health

### Metrics Collection
- Model accuracy trends
- Response times
- Error rates
- WebSocket connection counts
- Session statistics

### Logging
- Structured logging with JSON format
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Request/response logging
- WebSocket connection logging

## 🚀 Deployment

### Docker Deployment
```bash
# Build and start all services
docker-compose -f docker-compose.enhanced.yml up -d

# Scale services
docker-compose -f docker-compose.enhanced.yml up -d --scale backend=3
```

### AWS Deployment
```bash
# Deploy to ECS
aws ecs create-cluster --cluster-name mindguard-cluster
aws ecs register-task-definition --cli-input-json file://task-definition.json
aws ecs create-service --cluster mindguard-cluster --service-name mindguard-service --task-definition mindguard-task
```

### Kubernetes Deployment
```bash
# Apply Kubernetes manifests
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/ingress.yaml
```

## 🔒 Security

### Authentication & Authorization
- JWT-based authentication
- Role-based access control
- Session management with Redis

### Data Protection
- HTTPS/TLS encryption
- Input validation and sanitization
- Rate limiting and DDoS protection
- Secure WebSocket connections

### Privacy
- No video recording - analysis only
- Local processing where possible
- Encrypted data transmission
- Optional data storage for trend analysis

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines
- Follow PEP 8 for Python code
- Use TypeScript for frontend development
- Write comprehensive tests
- Update documentation
- Follow conventional commit messages

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- TensorFlow team for the ML framework
- FastAPI team for the excellent web framework
- Next.js team for the React framework
- OpenCV contributors for computer vision tools
- All open-source contributors who made this project possible

## 📞 Support

For support and questions:
- Create an issue on GitHub
- Email: support@mindguard.ai
- Documentation: https://docs.mindguard.ai

---

**Built with ❤️ for better mental health monitoring through AI**

