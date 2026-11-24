"""Prediction API.

    uv run uvicorn app.main:app --reload --port 8000
"""

import os
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, HTTPException, Request

from app.schemas import HealthResponse, PredictionResponse, TripRequest
from src.features import prepare_features
from src.model import DEFAULT_MODEL_PATH, load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # load the model once at startup, not on every request
    app.state.model = load_model(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    yield
    app.state.model = None


app = FastAPI(title="NYC Taxi Trip Duration", version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    loaded = getattr(request.app.state, "model", None) is not None
    return HealthResponse(status="ok" if loaded else "model not loaded", model_loaded=loaded)


@app.post("/predict", response_model=PredictionResponse)
def predict(trip: TripRequest, request: Request) -> PredictionResponse:
    model = getattr(request.app.state, "model", None)
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    # same feature code as training -> no training/serving skew
    features = prepare_features(pd.DataFrame([trip.model_dump()]))
    duration = float(model.predict(features)[0])
    return PredictionResponse(predicted_duration=round(duration, 2))
