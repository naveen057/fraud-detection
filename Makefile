.PHONY: setup data eda compare tune train api test docker-build docker-run benchmark all

setup:
	pip install -r requirements-dev.txt

data:
	python scripts/01_generate_data.py
eda:
	python scripts/02_eda.py
compare:
	python scripts/03_compare_models.py
tune:
	python scripts/04_tune_and_evaluate.py
train: tune                       # alias: produces models/fraud_model.joblib
all: data eda compare tune

api:
	uvicorn app.main:app --reload --port 8000
test:
	pytest -q
benchmark:                         # needs the API running on :8000
	python scripts/05_benchmark_api.py --n 500

docker-build:
	docker build -t fraud-detection-api .
docker-run:
	docker run --rm -p 8000:8000 fraud-detection-api
