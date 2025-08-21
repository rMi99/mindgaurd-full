up:
	export PYTHONPATH=$(pwd)
	python scripts/train_model.py

down:
	export PYTHONPATH=$(pwd)
	python app/main.py

uvicorn app.main:app --reload

cd /home/rmi/Desktop/mindgaurd-full/backend && source venv/bin/activate && python -m uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

cd /home/rmi/Desktop/mindgaurd-full/frontend && npm run dev