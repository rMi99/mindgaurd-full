#!/bin/bash

# MindGuard Startup Script
# This script starts both the backend and frontend services

echo "🚀 Starting MindGuard..."
echo "========================"

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo "❌ Port $1 is already in use"
        return 1
    else
        echo "✅ Port $1 is available"
        return 0
    fi
}

# Function to start backend
start_backend() {
    echo "🔧 Starting Backend..."
    cd backend
    
    # Check if virtual environment exists
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    echo "🔌 Activating virtual environment..."
    source venv/bin/activate
    
    # Install dependencies if requirements.txt exists
    if [ -f "requirements.txt" ]; then
        echo "📥 Installing dependencies..."
        pip install -r requirements.txt
    fi
    
    # Check if port 8000 is available
    if check_port 8000; then
        echo "🚀 Starting FastAPI server on port 8000..."
        python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
        BACKEND_PID=$!
        echo "✅ Backend started with PID: $BACKEND_PID"
    else
        echo "❌ Backend startup failed"
        return 1
    fi
    
    cd ..
}

# Function to start frontend
start_frontend() {
    echo "🎨 Starting Frontend..."
    cd frontend
    
    # Check if node_modules exists
    if [ ! -d "node_modules" ]; then
        echo "📦 Installing dependencies..."
        npm install
    fi
    
    # Check if port 3000 is available
    if check_port 3000; then
        echo "🚀 Starting Next.js server on port 3000..."
        npm run dev &
        FRONTEND_PID=$!
        echo "✅ Frontend started with PID: $FRONTEND_PID"
    else
        echo "❌ Frontend startup failed"
        return 1
    fi
    
    cd ..
}

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down services..."
    
    if [ ! -z "$BACKEND_PID" ]; then
        echo "🛑 Stopping backend (PID: $BACKEND_PID)..."
        kill $BACKEND_PID 2>/dev/null
    fi
    
    if [ ! -z "$FRONTEND_PID" ]; then
        echo "🛑 Stopping frontend (PID: $FRONTEND_PID)..."
        kill $FRONTEND_PID 2>/dev/null
    fi
    
    echo "✅ All services stopped"
    exit 0
}

# Set trap to cleanup on script exit
trap cleanup SIGINT SIGTERM EXIT

# Start services
start_backend
if [ $? -eq 0 ]; then
    start_frontend
    if [ $? -eq 0 ]; then
        echo ""
        echo "🎉 MindGuard is starting up!"
        echo "========================"
        echo "📱 Frontend: http://localhost:3000"
        echo "🔧 Backend:  http://localhost:8000"
        echo "📚 API Docs: http://localhost:8000/docs"
        echo ""
        echo "Press Ctrl+C to stop all services"
        echo ""
        
        # Wait for user to stop
        wait
    else
        echo "❌ Frontend failed to start"
        exit 1
    fi
else
    echo "❌ Backend failed to start"
    exit 1
fi 