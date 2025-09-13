# External Makefile for MindGuard Project
# Handles Docker, training scenarios, and deployment

.PHONY: help dev prod training clean logs test

# Default target
help:
	@echo "MindGuard Docker & Training Commands"
	@echo "=================================="
	@echo ""
	@echo "Development:"
	@echo "  dev          - Start development environment"
	@echo "  dev-build    - Build development images"
	@echo "  dev-stop     - Stop development environment"
	@echo ""
	@echo "Production:"
	@echo "  prod         - Start production environment"
	@echo "  prod-build   - Build production images"
	@echo "  prod-stop    - Stop production environment"
	@echo ""
	@echo "Training:"
	@echo "  training     - Start training environment"
	@echo "  train-base   - Train base model"
	@echo "  train-audio  - Train audio analysis model"
	@echo "  train-facial - Train facial analysis model"
	@echo "  train-multi  - Train multimodal model"
	@echo "  jupyter      - Start Jupyter training notebook"
	@echo ""
	@echo "Utilities:"
	@echo "  logs         - Show logs from all services"
	@echo "  clean        - Clean up containers and volumes"
	@echo "  test         - Run API tests"
	@echo "  health       - Check service health"

# Development Environment
dev:
	@echo "🚀 Starting development environment..."
	docker-compose -f docker-compose.dev.yml up -d
	@echo "✅ Development environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"
	@echo "MongoDB: localhost:27017"

dev-build:
	@echo "🔨 Building development images..."
	docker-compose -f docker-compose.dev.yml build

dev-stop:
	@echo "🛑 Stopping development environment..."
	docker-compose -f docker-compose.dev.yml down

dev-logs:
	@echo "📋 Development logs..."
	docker-compose -f docker-compose.dev.yml logs -f

# Production Environment
prod:
	@echo "🚀 Starting production environment..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "✅ Production environment started!"
	@echo "Frontend: http://localhost:3000"
	@echo "Backend: http://localhost:8000"

prod-build:
	@echo "🔨 Building production images..."
	docker-compose -f docker-compose.prod.yml build

prod-stop:
	@echo "🛑 Stopping production environment..."
	docker-compose -f docker-compose.prod.yml down

prod-logs:
	@echo "📋 Production logs..."
	docker-compose -f docker-compose.prod.yml logs -f

# Training Environment
training:
	@echo "🧠 Starting training environment..."
	docker-compose -f docker-compose.training.yml up -d training-base
	@echo "✅ Training environment ready!"
	@echo "Run: make train-base, make train-audio, make train-facial, or make train-multi"

train-base:
	@echo "🎯 Training base model..."
	docker-compose -f docker-compose.training.yml run --rm training-base python scripts/train_model.py

train-audio:
	@echo "🎵 Training audio analysis model..."
	docker-compose -f docker-compose.training.yml run --rm training-audio

train-facial:
	@echo "👁️ Training facial analysis model..."
	docker-compose -f docker-compose.training.yml run --rm training-facial

train-multi:
	@echo "🔮 Training multimodal model..."
	docker-compose -f docker-compose.training.yml run --rm training-multimodal

jupyter:
	@echo "📓 Starting Jupyter training notebook..."
	docker-compose -f docker-compose.training.yml up -d jupyter
	@echo "✅ Jupyter started at http://localhost:8888"

# Utility Commands
logs:
	@echo "📋 All service logs..."
	@echo "Choose: dev-logs, prod-logs, or training-logs"

training-logs:
	@echo "📋 Training logs..."
	docker-compose -f docker-compose.training.yml logs -f

clean:
	@echo "🧹 Cleaning up containers and volumes..."
	docker-compose -f docker-compose.dev.yml down -v --remove-orphans
	docker-compose -f docker-compose.prod.yml down -v --remove-orphans
	docker-compose -f docker-compose.training.yml down -v --remove-orphans
	docker system prune -f
	@echo "✅ Cleanup complete!"

test:
	@echo "🧪 Running API tests..."
	docker-compose -f docker-compose.dev.yml exec backend-dev python -m pytest tests/ -v

health:
	@echo "🏥 Checking service health..."
	@echo "Backend Health:"
	@curl -s http://localhost:8000/health | jq . || echo "Backend not responding"
	@echo "Frontend Health:"
	@curl -s http://localhost:3000 | head -n 1 || echo "Frontend not responding"
	@echo "Facial Analysis Health:"
	@curl -s http://localhost:8000/api/facial-analysis/health | jq . || echo "Facial analysis not responding"
	@echo "Audio Analysis Health:"
	@curl -s http://localhost:8000/api/audio-analysis/health | jq . || echo "Audio analysis not responding"

# Quick development setup
setup:
	@echo "⚡ Quick development setup..."
	make dev-build
	make dev
	@echo "⏳ Waiting for services to start..."
	sleep 10
	make health

# Training scenarios
scenario-quick:
	@echo "⚡ Quick training scenario..."
	make training
	make train-base

scenario-full:
	@echo "🎯 Full training scenario..."
	make training
	make train-base
	make train-audio
	make train-facial
	make train-multi

# Database operations
db-backup:
	@echo "💾 Backing up database..."
	docker-compose -f docker-compose.prod.yml exec mongo mongodump --db mindguard_prod --out /backup
	docker cp $$(docker-compose -f docker-compose.prod.yml ps -q mongo):/backup ./backup-$$(date +%Y%m%d_%H%M%S)

db-restore:
	@echo "📥 Restoring database..."
	@echo "Usage: make db-restore BACKUP_FILE=backup-20240101_120000"
	@if [ -z "$(BACKUP_FILE)" ]; then echo "Please specify BACKUP_FILE"; exit 1; fi
	docker cp $(BACKUP_FILE) $$(docker-compose -f docker-compose.prod.yml ps -q mongo):/restore
	docker-compose -f docker-compose.prod.yml exec mongo mongorestore --db mindguard_prod /restore

# Monitoring
monitor:
	@echo "📊 Service monitoring..."
	watch -n 5 'docker-compose -f docker-compose.dev.yml ps'

# Security scan
security-scan:
	@echo "🔒 Running security scan..."
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image mindguard-full_backend-dev:latest
	docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
		aquasec/trivy image mindguard-full_frontend-dev:latest

