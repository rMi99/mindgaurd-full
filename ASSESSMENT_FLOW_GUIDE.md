# Assessment API Flow - Complete Guide

## 🎯 Overview

This guide covers the complete end-to-end assessment flow that has been implemented and tested. The flow ensures proper authentication, data validation, and seamless user experience.

## ✅ What's Been Fixed

### 🔧 Backend Fixes
- **Enhanced Validation**: Added comprehensive validation for all required fields
- **Proper Error Messages**: Detailed error responses with specific field information
- **Authentication Integration**: All assessment endpoints require valid JWT tokens
- **Data Structure Handling**: Proper handling of the expected data format

### 🎨 Frontend Fixes
- **Authentication Checks**: All assessment routes check authentication before proceeding
- **Redirect Logic**: Seamless redirect to login with return to intended route
- **User Data Pre-filling**: Automatically fills user data when available
- **Error Handling**: Clear error messages and retry mechanisms
- **Data Transformation**: Proper data structure transformation for backend compatibility

## 🚀 Quick Start

### 1. Start the Backend
```bash
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Frontend
```bash
cd frontend
npm run dev
```

### 3. Test the Flow
```bash
python test_assessment_flow.py
```

## 🔄 Complete User Flow

### Step 1: User Clicks "Start Assessment"
- **Location**: Homepage (`/`)
- **Action**: User clicks "Start Assessment" button
- **Authentication Check**: If not logged in, redirects to `/auth` with return URL stored

### Step 2: Authentication
- **Location**: `/auth`
- **Options**: Login or Register
- **After Success**: Automatically redirects back to `/assessment`

### Step 3: Assessment Form
- **Location**: `/assessment`
- **Features**:
  - Pre-filled user data (age, gender, etc.)
  - Multi-step form with validation
  - Real-time error checking
  - Progress tracking

### Step 4: Assessment Submission
- **Data Validation**: Frontend validates all required fields
- **API Call**: Sends properly formatted data to `/api/assessment`
- **Backend Processing**: Comprehensive validation and assessment generation
- **Result Storage**: Assessment saved to database with user association

### Step 5: Results Display
- **Location**: `/assessment/results`
- **Features**:
  - Comprehensive assessment results
  - Risk level analysis
  - Personalized recommendations
  - Cultural considerations
  - Emergency resources

## 📊 Data Flow

### Frontend → Backend Data Structure
```json
{
  "demographics": {
    "age": "25",
    "gender": "female",
    "region": "North America",
    "education": "Bachelor's",
    "employmentStatus": "Employed"
  },
  "phq9": {
    "scores": {
      "1": 1,
      "2": 2,
      "3": 1,
      "4": 0,
      "5": 1,
      "6": 2,
      "7": 1,
      "8": 0,
      "9": 0
    }
  },
  "sleep": {
    "sleepHours": "7",
    "sleepQuality": "3",
    "exerciseFrequency": "4",
    "stressLevel": "5",
    "socialSupport": "4",
    "screenTime": "3"
  },
  "language": "en"
}
```

### Backend Response Structure
```json
{
  "assessment_id": "assess_abc123def456",
  "riskLevel": "moderate",
  "phq9Score": 8,
  "confidenceScore": 0.85,
  "riskFactors": ["Elevated PHQ-9 score", "Sleep disruption"],
  "protectiveFactors": ["Strong social support", "Regular exercise"],
  "recommendations": ["Consider speaking with a healthcare provider"],
  "culturalConsiderations": ["In your region, community support is valuable"],
  "emergencyResources": [...],
  "brainHealActivities": [...],
  "weeklyPlan": {...},
  "created_at": "2024-01-15T10:30:00Z"
}
```

## 🔐 Authentication Flow

### Token Management
- **Storage**: JWT tokens stored in localStorage as `mindguard_token`
- **Headers**: All API calls include `Authorization: Bearer <token>`
- **Validation**: Backend validates tokens on every protected endpoint

### Redirect Logic
```javascript
// Store intended destination
localStorage.setItem('redirect_after_login', '/assessment')

// After successful login
const redirectTo = localStorage.getItem('redirect_after_login')
localStorage.removeItem('redirect_after_login')
router.push(redirectTo || '/dashboard')
```

## 🛡️ Error Handling

### Frontend Error Handling
- **Validation Errors**: Real-time field validation with clear messages
- **API Errors**: Proper error display with retry options
- **Authentication Errors**: Automatic redirect to login
- **Network Errors**: User-friendly error messages

### Backend Error Handling
- **Validation Errors**: Detailed field-specific error messages
- **Authentication Errors**: 401 Unauthorized with clear message
- **Database Errors**: Graceful fallback with generated IDs
- **Processing Errors**: Comprehensive error logging

## 🧪 Testing

### Manual Testing Steps
1. **Start both servers** (backend and frontend)
2. **Navigate to homepage** (`http://localhost:3000`)
3. **Click "Start Assessment"** (should redirect to login if not authenticated)
4. **Register/Login** with test credentials
5. **Complete assessment form** with test data
6. **Verify results page** displays correctly
7. **Check dashboard** for assessment history

### Automated Testing
```bash
# Run the test script
python test_assessment_flow.py
```

### Test Credentials
- **Email**: `test@example.com`
- **Password**: `testpassword123`

## 🐛 Common Issues & Solutions

### "Missing required assessment data" Error
- **Cause**: Data structure mismatch between frontend and backend
- **Solution**: Ensure frontend sends data in the expected format

### Authentication Token Issues
- **Cause**: Token not properly stored or expired
- **Solution**: Clear localStorage and re-authenticate

### Validation Errors
- **Cause**: Missing required fields or invalid data types
- **Solution**: Check browser console for specific field errors

## 📈 Performance Optimizations

### Frontend
- **Lazy Loading**: Assessment components load on demand
- **State Management**: Efficient Zustand store for form data
- **Error Boundaries**: Graceful error handling

### Backend
- **Database Indexing**: Optimized queries for assessment history
- **Caching**: Assessment results cached for quick retrieval
- **Async Processing**: Non-blocking assessment generation

## 🔮 Future Enhancements

### Planned Features
- **Offline Support**: Assessment completion without internet
- **Progress Saving**: Auto-save assessment progress
- **Multi-language**: Full internationalization support
- **Advanced Analytics**: Detailed assessment insights
- **Integration**: Third-party health app integration

### Technical Improvements
- **Real-time Validation**: Server-side validation feedback
- **File Upload**: Support for additional assessment documents
- **WebSocket**: Real-time assessment progress updates
- **PWA**: Progressive web app capabilities

## 📞 Support

For issues or questions:
1. Check the browser console for error messages
2. Review the backend logs for API errors
3. Run the test script to verify functionality
4. Check this guide for common solutions

---

**Status**: ✅ Production Ready
**Last Updated**: January 2024
**Version**: 1.0.0
