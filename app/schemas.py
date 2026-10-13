from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

Category = Literal["grocery", "restaurant", "fuel", "retail", "travel",
                   "entertainment", "electronics", "jewelry", "gift_cards", "crypto"]
Channel = Literal["online", "pos", "atm"]


class Transaction(BaseModel):
    model_config = ConfigDict(json_schema_extra={"example": {
        "transaction_id": "txn_123456", "amount": 899.99, "hour": 2,
        "merchant_category": "electronics", "channel": "online",
        "is_foreign": 1, "device_new": 1, "distance_from_home_km": 340.0,
        "account_age_days": 420, "txn_count_1h": 1, "txn_count_24h": 2,
        "avg_amount_30d": 48.5, "time_since_last_txn_min": 35.0}})

    transaction_id: Optional[str] = Field(None, description="Echoed back in the response")
    amount: float = Field(..., gt=0, le=1_000_000)
    hour: int = Field(..., ge=0, le=23, description="Local hour of day")
    merchant_category: Category
    channel: Channel
    is_foreign: int = Field(..., ge=0, le=1)
    device_new: int = Field(..., ge=0, le=1, description="1 if first time this device is seen")
    distance_from_home_km: float = Field(..., ge=0)
    account_age_days: int = Field(..., ge=0)
    txn_count_1h: int = Field(..., ge=0, description="User's transactions in the previous hour")
    txn_count_24h: int = Field(..., ge=0, description="User's transactions in the previous 24h")
    avg_amount_30d: float = Field(..., ge=0, description="User's mean amount over the previous 30 days")
    time_since_last_txn_min: float = Field(..., ge=0, description="Minutes since user's previous transaction")


class Prediction(BaseModel):
    transaction_id: Optional[str] = None
    fraud_probability: float
    is_fraud: bool
    risk_level: Literal["low", "medium", "high"]
    threshold: float


class PredictionResponse(Prediction):
    latency_ms: float = Field(..., description="Model inference time for this request")


class BatchRequest(BaseModel):
    transactions: list[Transaction] = Field(..., min_length=1, max_length=1000)


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    count: int
    flagged: int
    latency_ms: float
