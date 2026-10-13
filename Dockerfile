FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000

# libgomp is required by XGBoost / scikit-learn (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY fraud/ fraud/
COPY app/ app/
COPY models/ models/
# Fail the build early (with a clear message) if the trained model is missing.
RUN test -f models/fraud_model.joblib || (echo "ERROR: models/fraud_model.joblib missing - run 'make train' first" && exit 1)

RUN useradd -m appuser && chown -R appuser /app
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fs http://localhost:${PORT}/health || exit 1

# Render injects $PORT; 1 worker per container keeps latency predictable (scale by replicas).
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT} --workers 1"]
