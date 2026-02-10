import os
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    features: list[float] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Iris features: sepal_length, sepal_width, petal_length, petal_width",
    )


class PredictResponse(BaseModel):
    prediction: int
    probabilities: list[float]


app = FastAPI(title="ML Docker Versioning API")


@app.on_event("startup")
def load_model_on_startup() -> None:
    model_path = Path(os.getenv("MODEL_PATH", "model/model.joblib"))
    app.state.model_path = model_path
    app.state.model = None
    app.state.model_loaded = False
    app.state.model_error = None

    try:
        app.state.model = joblib.load(model_path)
        app.state.model_loaded = True
    except Exception as exc:  # pragma: no cover
        app.state.model_error = str(exc)


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if app.state.model_loaded else "degraded",
        "model_loaded": app.state.model_loaded,
        "model_path": str(app.state.model_path),
        "error": app.state.model_error,
    }


@app.get("/version")
def version() -> dict[str, str]:
    return {"version": os.getenv("APP_VERSION", "0.1.0")}


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    if not app.state.model_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")

    features = [payload.features]
    prediction = int(app.state.model.predict(features)[0])
    probabilities = app.state.model.predict_proba(features)[0].tolist()

    return PredictResponse(prediction=prediction, probabilities=probabilities)
