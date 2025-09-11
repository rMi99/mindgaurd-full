# Makefile for MindGuard project

PYTHONPATH := $(shell pwd)
BACKEND_DIR := /home/rmi/Desktop/mindgaurd-full/backend
FRONTEND_DIR := /home/rmi/Desktop/mindgaurd-full/frontend
VENV := $(BACKEND_DIR)/venv/bin/activate
PYTHON := $(BACKEND_DIR)/venv/bin/python
PIP := $(BACKEND_DIR)/venv/bin/pip

# Build command - setup virtual environment and install all packages
build:
	@echo "🏗️ Building MindGuard backend environment..."
	@echo "📂 Creating virtual environment..."
	@python3 -m venv $(BACKEND_DIR)/venv
	@echo "⬆️ Upgrading pip..."
	@$(PIP) install --upgrade pip
	@echo "📦 Installing Python packages from requirements.txt..."
	@$(PIP) install -r $(BACKEND_DIR)/requirements.txt
	@echo "✅ Build complete! Virtual environment created and packages installed."
	@echo "🚀 You can now run: make backend (to start server) or make setup (to initialize database)"

# Clean build - remove virtual environment and rebuild
clean-build:
	@echo "🧹 Cleaning previous build..."
	@rm -rf $(BACKEND_DIR)/venv
	@make build

# Install additional packages (usage: make install PACKAGE=package_name)
install:
	@echo "📦 Installing $(PACKAGE)..."
	@$(PIP) install $(PACKAGE)
	@$(PIP) freeze > $(BACKEND_DIR)/requirements.txt
	@echo "✅ $(PACKAGE) installed and requirements.txt updated"

# Train model
train:
	@echo "🚀 Training model..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/train_model.py

# Run inference
infer:
	@echo "🔎 Running inference..."
	@PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/infer_model.py

# Start backend server (dev)souc
backend:
	@echo "🌐 Starting backend (Uvicorn)..."
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# Start frontend server (Next.js)
frontend:
	@echo "💻 Starting frontend (Next.js)..."
	cd $(FRONTEND_DIR) && npm run dev

# Start both backend & frontend together
up:
	@echo "🚀 Starting backend & frontend..."
	@make -j2 backend frontend

# Initialize database (run init_db.py)
init-db:
	@echo "🗃️ Initializing database..."
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) $(PYTHON) scripts/init_db.py

# Seed initial data
seed:
	@echo "🌱 Seeding initial data..."
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) $(PYTHON) app/seeders/db_seeder.py

# Force refresh seeded data (clears existing data)
seed-force:
	@echo "🔄 Force refreshing seeded data..."
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) $(PYTHON) app/seeders/db_seeder.py --force

# Test seeder (without database - just data generation)
seed-test:
	@echo "🧪 Testing data seeder..."
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) $(PYTHON) app/seeders/seed_initial_data.py

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
	@echo "   make build        - Create virtual environment and install all packages"
	@echo "   make clean-build  - Remove venv and rebuild from scratch"
	@echo "   make setup        - Initialize database and seed data"
	@echo ""
	@echo "🚀 Development Commands:"
	@echo "   make backend      - Start backend server (uvicorn)"
	@echo "   make frontend     - Start frontend server (Next.js)"
	@echo "   make up           - Start both backend and frontend"
	@echo ""
	@echo "🤖 ML Commands:"
	@echo "   make train        - Train the ML model"
	@echo "   make infer        - Run model inference"
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
	@echo "💡 Quick start: make build && make setup && make backend"



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
 

#< proper name>
#@cd /home/rmi/Desktop/mindgaurd-full/backend && /home/rmi/Desktop/mindgaurd-full/venv/bin/python -c "
#from app.services.health_recommendation_service import HealthRecommendationService
#try:
 #   service = HealthRecommendationService()
  #  print('HealthRecommendationService created successfully')
    #print(f'Exercise database: {len(service.exercise_database)} exercises')
   # print(f'Game database: {len(service.game_database)} games')
#except Exception as e:
 #   print(f'Error: {e}')"