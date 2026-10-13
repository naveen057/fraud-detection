"""Real-time fraud scoring API.

    uvicorn app.main:app --host 0.0.0.0 --port 8000

Env vars: MODEL_PATH (default models/fraud_model.joblib), LATENCY_SLA_MS (default 50)
"""
import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.templating import Jinja2Templates

from fraud.config import MODEL_PATH
from fraud.predictor import FraudPredictor

from .latency import LatencyTracker
from .schemas import BatchRequest, BatchResponse, Prediction, PredictionResponse, Transaction

templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

log = logging.getLogger("fraud-api")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")


@asynccontextmanager
async def lifespan(app: FastAPI):
    path = os.getenv("MODEL_PATH", str(MODEL_PATH))
    app.state.predictor = FraudPredictor.load(path)
    app.state.latency = LatencyTracker(sla_ms=float(os.getenv("LATENCY_SLA_MS", "50")))
    app.state.started_at = time.time()
    # warm-up so the first real request does not pay one-off initialisation costs
    warm = Transaction(amount=10, hour=12, merchant_category="grocery", channel="pos", is_foreign=0,
                       device_new=0, distance_from_home_km=1, account_age_days=100, txn_count_1h=0,
                       txn_count_24h=1, avg_amount_30d=20, time_since_last_txn_min=600)
    app.state.predictor.predict([warm.model_dump(exclude={"transaction_id"})])
    log.info("Model loaded from %s (threshold=%.4f)", path, app.state.predictor.threshold)
    yield


app = FastAPI(
    title="Real-Time Fraud Detection API",
    description="Scores card transactions with a tuned gradient-boosting / random-forest model.",
    version="1.0.0", lifespan=lifespan)


@app.get("/", include_in_schema=False, name="home")
def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={})


@app.middleware("http")
async def latency_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-ms"] = f"{ms:.2f}"
    if request.url.path.startswith("/predict"):
        tracker = getattr(request.app.state, "latency", None)
        if tracker is not None:
            tracker.record(ms)
            if ms > tracker.sla_ms:
                log.warning("SLA breach: %s took %.1f ms (sla %.0f ms)", request.url.path, ms, tracker.sla_ms)
    return response


def _records(txns: list[Transaction]) -> list[dict]:
    return [t.model_dump(exclude={"transaction_id"}) for t in txns]


@app.get("/health", tags=["ops"])
def health(request: Request):
    return {"status": "ok", "uptime_s": round(time.time() - request.app.state.started_at, 1)}


@app.get("/model-info", tags=["ops"])
def model_info(request: Request):
    info = request.app.state.predictor.info
    return {k: info.get(k) for k in ("model_name", "strategy", "threshold", "trained_at",
                                     "boosting_backend", "metrics", "raw_features")}


@app.get("/metrics", tags=["ops"])
def metrics(request: Request):
    """Rolling prediction-latency statistics (p50/p95/p99) and SLA violations."""
    return request.app.state.latency.snapshot()


@app.post("/predict", response_model=PredictionResponse, tags=["scoring"])
def predict(txn: Transaction, request: Request):
    t0 = time.perf_counter()
    try:
        out = request.app.state.predictor.predict(_records([txn]))[0]
    except Exception as exc:  # pragma: no cover
        log.exception("prediction failed")
        raise HTTPException(status_code=500, detail="prediction failed") from exc
    return PredictionResponse(transaction_id=txn.transaction_id, **out,
                              latency_ms=round((time.perf_counter() - t0) * 1000, 3))


@app.post("/predict/batch", response_model=BatchResponse, tags=["scoring"])
def predict_batch(req: BatchRequest, request: Request):
    t0 = time.perf_counter()
    outs = request.app.state.predictor.predict(_records(req.transactions))
    preds = [Prediction(transaction_id=t.transaction_id, **o) for t, o in zip(req.transactions, outs)]
    return BatchResponse(predictions=preds, count=len(preds), flagged=sum(p.is_fraud for p in preds),
                         latency_ms=round((time.perf_counter() - t0) * 1000, 3))
