# Makefile for MindGuard project - Unified Environment Setup

# Phony targets (not files)
.PHONY: help build clean-build setup backend frontend up stop status train infer init-db

PYTHONPATH := $(shell pwd)
BACKEND_DIR := /home/rmi/Desktop/mindgaurd-full/backend
FRONTEND_DIR := /home/rmi/Desktop/mindgaurd-full/frontend
ROOT_VENV := /home/rmi/Desktop/mindgaurd-full/venv
PYTHON := /usr/bin/python3
PIP := /usr/bin/python3 -m pip

# Setup virtual environment and install all packages
setup-venv:
	@echo "🐍 Creating virtual environment..."
	@python3 -m venv venv
	@echo "✅ Virtual environment created at ./venv"
	@echo "📦 Installing packages in virtual environment..."
	@./venv/bin/pip install --upgrade pip
	@./venv/bin/pip install -r requirements.txt
	@echo "✅ Virtual environment setup complete!"
	@echo "🚀 To activate: source venv/bin/activate"
	@echo "🚀 To start server: make backend-venv"

# Build command - install all packages (system-wide)
build:
	@echo "🏗️ Building MindGuard backend environment..."
	@echo "⬆️ Upgrading pip..."
	@$(PIP) install --upgrade pip

	@echo "📦 Installing Python packages from requirements.txt..."
	@$(PIP) install -r requirements.txt
	@echo "✅ Build complete! All packages installed."
	@echo "🚀 You can now run: make backend (to start server) or make setup (to initialize database)"

# Clean build - reinstall packages
clean-build:
	@echo "🧹 Cleaning previous build..."
	@$(PIP) uninstall -y -r requirements.txt || true
	@make build

# Install additional packages (usage: make install PACKAGE=package_name)
install:
	@echo "📦 Installing $(PACKAGE)..."
	@$(PIP) install $(PACKAGE)
	@$(PIP) freeze > requirements.txt
	@echo "✅ $(PACKAGE) installed and requirements.txt updated"

# Train model
train:
	@echo "🚀 Training model..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_model.py

# Run inference
infer:
	@echo "🔎 Running inference..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/infer_model.py

# Start backend server (dev) - using virtual environment
backend-venv:
	@echo "🚀 Starting backend with virtual environment (Uvicorn)..."
	@nohup ./venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
	@echo "✅ Backend started in background (PID: $$!)"
	@echo "📝 Logs: tail -f backend.log"

# Start backend server (dev) - system-wide
backend:
	@echo "🚀 Starting backend (Uvicorn)..."
	@nohup $(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload > backend.log 2>&1 &
	@echo "✅ Backend started in background (PID: $$!)"
	@echo "📝 Logs: tail -f backend.log"

# Start frontend server (Next.js)
frontend:
	@echo "💻 Starting frontend (Next.js)..."
	@cd $(FRONTEND_DIR) && nohup npm run dev > ../frontend.log 2>&1 &
	@echo "✅ Frontend started in background (PID: $$!)"
	@echo "📝 Logs: tail -f frontend.log"

# Start both backend & frontend together
up:
	@echo "🚀 Starting backend & frontend..."
	@make -j2 backend frontend

# Stop all running services
stop:
	@echo "🛑 Stopping all services..."
	@pkill -f "uvicorn app.main:app" || echo "Backend not running"
	@pkill -f "next dev" || echo "Frontend not running"
	@pkill -f "node.*next" || echo "Next.js not running"
	@echo "✅ All services stopped"

# Check status of running services
status:
	@echo "📊 Service Status:"
	@echo "Backend:"
	@pgrep -f "uvicorn app.main:app" && echo "  ✅ Running (PID: $$(pgrep -f 'uvicorn app.main:app'))" || echo "  ❌ Not running"
	@echo "Frontend:"
	@pgrep -f "next dev" && echo "  ✅ Running (PID: $$(pgrep -f 'next dev'))" || echo "  ❌ Not running"
	@echo ""
	@echo "🌐 URLs:"
	@echo "  Backend:  http://localhost:8000"
	@echo "  Frontend: http://localhost:3000"

# Initialize database (run init_db.py)
init-db:
	@echo "🗃️ Initializing database..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/init_db.py

# Seed initial data
seed:
	@echo "🌱 Seeding initial data..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) app/seeders/db_seeder.py

# Force refresh seeded data (clears existing data)
seed-force:
	@echo "🔄 Force refreshing seeded data..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) app/seeders/db_seeder.py --force

# Test seeder (without database - just data generation)
seed-test:
	@echo "🧪 Testing data seeder..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) app/seeders/seed_initial_data.py

# Setup project (init db + seed data)
setup:
	@echo "⚙️ Setting up project..."
	@make init-db
	@make seed

# Show available commands
help:
	@echo "🎯 MindGuard Backend - Available Commands:"
	@echo ""
	@echo "🏗️  Setup Commands:"
	@echo "   make setup-venv   - Create virtual environment and install all packages"
	@echo "   make build        - Install packages system-wide (no venv)"
	@echo "   make clean-build  - Remove venv and rebuild from scratch"
	@echo "   make setup        - Initialize database and seed data"
	@echo ""
	@echo "🚀 Development Commands:"
	@echo "   make backend      - Start backend server (uvicorn) - system-wide"
	@echo "   make backend-venv - Start backend server (uvicorn) - using virtual environment"
	@echo "   make frontend     - Start frontend server (Next.js)"
	@echo "   make up           - Start both backend and frontend"
	@echo "   make stop         - Stop all running services"
	@echo "   make status       - Check status of running services"
	@echo ""
	@echo "🤖 ML Commands:"
	@echo "   make train        - Train the ML model"
	@echo "   make infer        - Run model inference"
	@echo ""
	@echo "🎭 Facial Analysis Commands:"
	@echo "   make download-data     - Download facial models and datasets"
	@echo "   make download-real-data - Download real facial datasets and models"
	@echo "   make install-ml-packages - Install required ML packages (fer, deepface, scikit-learn)"
	@echo "   make train-real-fer    - Train real FER model with actual data"
	@echo "   make train-all         - Train FER and DeepFace models"
	@echo "   make run-app           - Start backend server (simple_server.py)"
	@echo "   make setup-and-run     - Complete automation: download → install → train → run"
	@echo "   make setup-real-facial - Setup real facial analysis system"
	@echo "   make setup-real-complete - Complete real facial analysis setup and run"
	@echo ""
	@echo "🗄️  Database Commands:"
	@echo "   make init-db      - Initialize database"
	@echo "   make seed         - Seed initial data"
	@echo "   make seed-force   - Force refresh seeded data"
	@echo "   make seed-test    - Test data seeder"
	@echo ""
	@echo "📦 Package Management:"
	@echo "   make install PACKAGE=name  - Install a new package and update requirements"
	@echo ""
	@echo "💡 Quick start (with venv): make setup-venv && make setup && make backend-venv"
	@echo "💡 Quick start (system-wide): make build && make setup && make backend"
	@echo "🎭 Facial analysis: make setup-and-run"



# train:
# 	export PYTHONPATH=$(pwd)
# 	python scripts/train_model.py

# train:
# 	export PYTHONPATH=$(pwd)
# 	python scripts/infer_model.py


# up:
# 	export PYTHONPATH=$(pwd)
# 	python scripts/train_model.py

# uvicorn app.main:app --reload

# cd /home/rmi/Desktop/mindgaurd-full/backend && source venv/bin/activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# cd /home/rmi/Desktop/mindgaurd-full/frontend && npm run dev

# cd /home/rmi/Desktop/mindgaurd-full/backend && source venv/bin/activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# cd /home/rmi/Desktop/mindgaurd-full/backend && source venv/bin/activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

#< proper name>
#/home/rmi/Desktop/mindgaurd-full/venv/bin/python -c "from app.services.health_recommendation_service import HealthRecommendationService; print('HealthRecommendationService imported successfully')"
 


# =============================
# MindGuard Facial Analysis Commands
# =============================

# --- Download Models & Data ---
download-data:
	@echo "⬇️ Downloading facial models and data..."
	@$(PYTHON) scripts/download_facial_data.py
	@echo "✅ Data and models downloaded successfully."

# --- Download Real Facial Data ---
download-real-data:
	@echo "⬇️ Downloading real facial datasets and models..."
	@$(PYTHON) scripts/download_real_facial_data.py
	@echo "✅ Real facial data downloaded successfully."

# --- Install Missing Packages ---
install-ml-packages:
	@echo "📦 Installing ML packages..."
	@$(PIP) install --break-system-packages fer deepface scikit-learn
	@echo "✅ ML packages installed successfully."

# --- Train Real FER Model ---
train-real-fer:
	@echo "🚀 Training real FER model..."
	@$(PYTHON) scripts/train_real_fer_model.py
	@echo "✅ Real FER model trained successfully."

# --- Train All Models ---
train-all:
	@echo "🚀 Starting full training..."
	@echo "📦 Using real training scripts..."
	@$(PYTHON) scripts/train_real_fer_model.py || echo "❌ Real FER training failed"
	@$(PYTHON) scripts/train_fer_model_simple.py || echo "❌ FER training failed"
	@$(PYTHON) scripts/train_deepface_model_simple.py || echo "❌ DeepFace training failed"
	@echo "✅ All models trained successfully."

# --- Run Backend App ---
run-app:
	@echo "🔥 Starting backend server..."
	@$(PYTHON) simple_server.py

# --- One-Command Automation ---
setup-and-run:
	@make download-data
	@make train-all
	@make run-app

# --- Real Facial Analysis Setup ---
setup-real-facial:
	@echo "🎯 Setting up real facial analysis system..."
	@make download-real-data
	@make install-ml-packages
	@make train-real-fer
	@echo "✅ Real facial analysis system ready!"

# --- Complete Real Setup ---
setup-real-complete:
	@echo "🚀 Complete real facial analysis setup..."
	@make download-real-data
	@make install-ml-packages
	@make train-real-fer
	@make backend
	@echo "🎉 Real facial analysis system running!"

# Train all emotion detection models (legacy)
train-all-legacy:
	@echo "🚀 Training all emotion models..."
	@echo "🧠 Training FER model..."
	@$(PYTHON) scripts/train_fer_model_simple.py || echo "❌ FER training failed"
	@echo "🤖 Training DeepFace model..."
	@$(PYTHON) scripts/train_deepface_model_simple.py || echo "❌ DeepFace training failed"
	@echo "🎯 Training facial model..."
	@$(PYTHON) scripts/train_facial_model.py || echo "❌ Facial model training failed"
	@echo "✅ Training complete."

# Train specific models
train-fer:
	@echo "🧠 Training FER model..."
	@$(PYTHON) scripts/train_fer_model.py
	@echo "✅ FER training complete."

train-deepface:
	@echo "🤖 Training DeepFace model..."
	@$(PYTHON) scripts/train_deepface_model.py
	@echo "✅ DeepFace training complete."
 