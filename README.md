# MindGuard - AI-Powered Health Risk Prediction System

MindGuard is a comprehensive health risk assessment platform that uses machine learning to provide personalized health insights and recommendations. Built with FastAPI backend and Next.js frontend, it offers a modern, secure, and user-friendly experience.

## 🚀 Features

### Backend (FastAPI)
- **Enhanced Prediction Model**: GradientBoostingClassifier for accurate health risk assessment
- **Recommendation Engine**: Personalized recommendations and "Brain Heal" activities
- **JWT Authentication**: Secure user management with passlib password hashing
- **RESTful API**: Clean, documented endpoints for all functionality
- **CORS Support**: Cross-origin resource sharing for frontend integration

### Frontend (Next.js + TypeScript)
- **Dynamic Dashboard**: Interactive health charts using Recharts
- **Multi-step Assessment**: Seamless wizard with Zustand state management
- **Responsive Design**: Modern UI/UX with TailwindCSS
- **Internationalization**: Multi-language support (English, Spanish, French)
- **Dark Mode**: Theme switching with persistent preferences

## 🛠️ Tech Stack

### Backend
- **Python 3.8+**
- **FastAPI** - Modern, fast web framework
- **scikit-learn** - Machine learning library
- **passlib** - Password hashing
- **PyJWT** - JSON Web Token authentication
- **uvicorn** - ASGI server

### Frontend
- **Next.js 14** - React framework with App Router
- **TypeScript** - Type-safe JavaScript
- **TailwindCSS** - Utility-first CSS framework
- **Zustand** - State management
- **Recharts** - Chart library
- **next-intl** - Internationalization

## 📋 Prerequisites

- Python 3.8 or higher
- Node.js 18 or higher
- npm or yarn package manager
- Git

## 🚀 Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd mindgaurd-full
```

### 2. Backend Setup

```bash
# Navigate to backend directory
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run setup script (optional)
python setup.py

# Start the server
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`
- API Documentation: `http://localhost:8000/docs`
- Health Check: `http://localhost:8000/health`

### 3. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
# or
yarn install

# Start development server
npm run dev
# or
yarn dev
```

The frontend will be available at `http://localhost:3000`

## 📁 Project Structure

```
mindgaurd-full/
├── backend/
│   ├── app/
│   │   ├── models/
│   │   │   └── enhanced_model.py          # ML model implementation
│   │   ├── routes/
│   │   │   └── auth.py                    # Authentication endpoints
│   │   ├── services/
│   │   │   └── recommendation_service.py  # Recommendation logic
│   │   └── main.py                        # FastAPI application
│   ├── requirements.txt                    # Python dependencies
│   └── setup.py                           # Setup script
├── frontend/
│   ├── app/
│   │   ├── components/
│   │   │   ├── dashboard/
│   │   │   │   └── InteractiveHealthChart.tsx  # Health charts
│   │   │   └── Header.tsx                        # Navigation header
│   │   ├── assessment/
│   │   │   └── page.tsx                          # Assessment wizard
│   │   └── layout.tsx                           # App layout
│   ├── lib/
│   │   ├── api.ts                              # API client
│   │   ├── stores/
│   │   │   ├── assessmentStore.ts              # Assessment state
│   │   │   └── authStore.ts                    # Authentication state
│   │   └── translations.ts                     # Internationalization
│   └── package.json                            # Node.js dependencies
└── README.md                                   # This file
```

## 🔧 Configuration

### Backend Environment Variables

Create a `.env` file in the backend directory:

```env
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Security
SECRET_KEY=mindguard_secret_key_2024
JWT_SECRET_KEY=mindguard_secret_key_2024
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=30

# CORS
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:3001
```

### Frontend Environment Variables

Create a `.env.local` file in the frontend directory:

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## 📊 API Endpoints

### Authentication
- `POST /api/auth/register` - User registration
- `POST /api/auth/login` - User login
- `POST /api/auth/logout` - User logout
- `GET /api/auth/me` - Get current user info
- `POST /api/auth/refresh` - Refresh access token

### Assessment
- `POST /api/assessment/submit` - Submit health assessment
- `GET /api/assessment/history` - Get assessment history
- `GET /api/assessment/{id}` - Get specific assessment

### Dashboard
- `GET /api/dashboard/data` - Get dashboard data
- `GET /api/dashboard/trends` - Get health trends

### Recommendations
- `GET /api/recommendations` - Get personalized recommendations
- `GET /api/recommendations/brain-heal` - Get Brain Heal activities

## 🧪 Testing

### Backend Testing

```bash
cd backend
python -m pytest
```

### Frontend Testing

```bash
cd frontend
npm run test
```

## 🚀 Deployment

### Backend Deployment

1. **Production Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Variables**
   - Set `DEBUG=false`
   - Use strong, unique secret keys
   - Configure production database

3. **Server**
   ```bash
   uvicorn app.main:app --host 0.0.0.0 --port 8000
   ```

### Frontend Deployment

1. **Build**
   ```bash
   npm run build
   ```

2. **Start Production Server**
   ```bash
   npm start
   ```

## 🔒 Security Features

- JWT-based authentication
- Password hashing with bcrypt
- CORS configuration
- Input validation and sanitization
- Secure HTTP headers

## 🌐 Internationalization

The application supports multiple languages:
- **English** (en) - Default
- **Spanish** (es) - Español
- **French** (fr) - Français

Language switching is handled through the `useTranslations` hook and translation files.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -am 'Add feature'`
4. Push to branch: `git push origin feature-name`
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues:

1. Check the documentation at `/docs` endpoint
2. Review the logs in the backend
3. Check browser console for frontend errors
4. Create an issue in the repository

## 🎯 Roadmap

- [ ] Database integration (PostgreSQL/MySQL)
- [ ] Real-time notifications
- [ ] Mobile app (React Native)
- [ ] Advanced analytics dashboard
- [ ] Integration with health devices
- [ ] Machine learning model training interface

## 🙏 Acknowledgments

- FastAPI community for the excellent framework
- Next.js team for the amazing React framework
- scikit-learn contributors for ML tools
- All contributors and users of this project

---

**Built with ❤️ for better mental health awareness and support** 