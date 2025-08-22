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