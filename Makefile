# Makefile — Shortcuts for common development tasks
# Usage: make <target>

.PHONY: install train api ui docker-up docker-down mlflow clean

# Install all dependencies
install:
	pip install -r requirements.txt

# Train the model (generates data if not present)
train:
	PYTHONPATH=. python app/training/train.py

# Train with hyperparameter tuning (slower)
train-tune:
	PYTHONPATH=. python app/training/train.py --tune

# Start the FastAPI backend
api:
	PYTHONPATH=. uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --reload

# Start the Streamlit UI
ui:
	PYTHONPATH=. streamlit run app/ui/streamlit_app.py

# Start MLflow tracking server
mlflow:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

# Docker commands
docker-up:
	cd docker && docker-compose up --build

docker-down:
	cd docker && docker-compose down

docker-logs:
	cd docker && docker-compose logs -f

# Clean generated files
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -f predictions_log.jsonl mlflow.db
