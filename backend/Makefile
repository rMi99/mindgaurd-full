# Makefile for MindGuard project

PYTHONPATH := $(shell pwd)
BACKEND_DIR := /home/rmi/Desktop/mindgaurd-full/backend
FRONTEND_DIR := /home/rmi/Desktop/mindgaurd-full/frontend
VENV := $(BACKEND_DIR)/venv/bin/activate

# Train model
train:
	@echo "🚀 Training model..."
	@PYTHONPATH=$(PYTHONPATH) python scripts/train_model.py

# Run inference
infer:
	@echo "🔎 Running inference..."
	@PYTHONPATH=$(PYTHONPATH) python scripts/infer_model.py

# Start backend server (dev)
backend:
	@echo "🌐 Starting backend (Uvicorn)..."
	$(BACKEND_DIR)/venv/bin/python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

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
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) python scripts/init_db.py

# Seed initial data
seed:
	@echo "🌱 Seeding initial data..."
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) python app/seeders/db_seeder.py

# Force refresh seeded data (clears existing data)
seed-force:
	@echo "🔄 Force refreshing seeded data..."
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) python app/seeders/db_seeder.py --force

# Test seeder (without database - just data generation)
seed-test:
	@echo "🧪 Testing data seeder..."
	@cd $(BACKEND_DIR) && PYTHONPATH=$(PYTHONPATH) python app/seeders/seed_initial_data.py

# Setup project (init db + seed data)
setup:
	@echo "⚙️ Setting up project..."
	@make init-db
	@make seed



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