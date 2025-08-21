#!/bin/bash

echo "🚀 Starting MindGuard Assessment Flow Test"
echo "=========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "❌ Node.js is not installed. Please install Node.js first."
    exit 1
fi

# Check if npm is installed
if ! command -v npm &> /dev/null; then
    echo "❌ npm is not installed. Please install npm first."
    exit 1
fi

echo "✅ Prerequisites check passed"

# Function to check if a port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Start backend if not running
if ! check_port 8000; then
    echo "🔧 Starting backend server..."
    cd backend
    if [ ! -d "venv" ]; then
        echo "📦 Creating virtual environment..."
        python3 -m venv venv
    fi
    
    echo "📦 Installing backend dependencies..."
    source venv/bin/activate
    pip install -r requirements.txt
    
    echo "🚀 Starting backend on http://localhost:8000"
    python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
    BACKEND_PID=$!
    cd ..
    
    # Wait for backend to start
    echo "⏳ Waiting for backend to start..."
    sleep 5
else
    echo "✅ Backend already running on port 8000"
fi

# Start frontend if not running
if ! check_port 3000; then
    echo "🎨 Starting frontend server..."
    cd frontend
    
    echo "📦 Installing frontend dependencies..."
    npm install
    
    echo "🚀 Starting frontend on http://localhost:3000"
    npm run dev &
    FRONTEND_PID=$!
    cd ..
    
    # Wait for frontend to start
    echo "⏳ Waiting for frontend to start..."
    sleep 10
else
    echo "✅ Frontend already running on port 3000"
fi

echo ""
echo "🎉 Both servers are starting up!"
echo ""
echo "📋 Next steps:"
echo "1. Wait for both servers to fully start (check the terminal output above)"
echo "2. Open http://localhost:3000 in your browser"
echo "3. Click 'Start Assessment' to test the flow"
echo "4. Register/login with test credentials:"
echo "   - Email: test@example.com"
echo "   - Password: testpassword123"
echo ""
echo "🔧 To stop the servers, press Ctrl+C in this terminal"
echo ""

# Wait for user to stop
wait
