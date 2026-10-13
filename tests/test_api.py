import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client(model_path, monkeypatch):
    monkeypatch.setenv("MODEL_PATH", str(model_path))
    from app.main import app
    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Transaction details" in response.text
    assert 'data-endpoint="' in response.text and "/predict" in response.text


def test_predict(client, sample_txn):
    r = client.post("/predict", json=sample_txn)
    assert r.status_code == 200
    body = r.json()
    assert body["transaction_id"] == "t1" and 0 <= body["fraud_probability"] <= 1
    assert "X-Process-Time-ms" in r.headers


def test_validation_errors(client, sample_txn):
    assert client.post("/predict", json={**sample_txn, "amount": -5}).status_code == 422
    assert client.post("/predict", json={**sample_txn, "merchant_category": "yachts"}).status_code == 422
    bad = dict(sample_txn); bad.pop("hour")
    assert client.post("/predict", json=bad).status_code == 422


def test_batch_and_metrics(client, sample_txn):
    r = client.post("/predict/batch", json={"transactions": [sample_txn] * 5})
    assert r.status_code == 200 and r.json()["count"] == 5
    client.post("/predict", json=sample_txn)
    m = client.get("/metrics").json()
    assert m["total_requests"] >= 2 and "p95_ms" in m


def test_model_info(client):
    assert "threshold" in client.get("/model-info").json()
