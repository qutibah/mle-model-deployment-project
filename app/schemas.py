"""Request/response models. Field names match the raw TLC columns that src.features expects."""

from datetime import datetime

from pydantic import BaseModel, Field


class TripRequest(BaseModel):
    PULocationID: int = Field(ge=1, le=265, description="TLC pickup zone ID", examples=[161])
    DOLocationID: int = Field(ge=1, le=265, description="TLC dropoff zone ID", examples=[236])
    trip_distance: float = Field(gt=0, le=100, description="Trip distance in miles", examples=[2.5])
    tpep_pickup_datetime: datetime = Field(description="Pickup time", examples=["2025-01-15T08:30:00"])


class PredictionResponse(BaseModel):
    predicted_duration: float = Field(description="Predicted trip duration")
    unit: str = "minutes"


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
