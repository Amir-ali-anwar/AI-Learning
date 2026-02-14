"""
FastAPI REST API for Used Car Price Prediction.

Provides:
- POST /predict - Predict car price
- GET /health - Health check
- GET /model-info - Model metadata

Usage:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
import os
import json
import logging
import time
from datetime import datetime
from typing import Optional

# Add src to path before any imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Models
# ============================================================================

class CarFeatures(BaseModel):
    """Input schema for car features."""
    year: int = Field(..., ge=1990, le=2027, description="Model year")
    odometer: int = Field(..., ge=0, le=500000, description="Mileage reading")
    manufacturer: str = Field(..., description="Car manufacturer, e.g. 'ford'")
    model: str = Field(..., description="Car model, e.g. 'f-150'")
    fuel: str = Field(..., description="Fuel type: gas, diesel, electric, hybrid, other")
    
    # Optional fields (defaults will be used if not provided)
    condition: Optional[str] = Field(None, description="Vehicle condition")
    cylinders: Optional[str] = Field(None, description="Number of cylinders")
    title_status: Optional[str] = Field(None, description="Title status")
    transmission: Optional[str] = Field(None, description="Transmission type")
    drive: Optional[str] = Field(None, description="Drive type: fwd, rwd, 4wd")
    type: Optional[str] = Field(None, description="Vehicle type: sedan, SUV, truck, etc.")
    paint_color: Optional[str] = Field(None, description="Paint color")
    state: Optional[str] = Field(None, description="US state abbreviation")
    
    @validator('manufacturer', 'model', 'fuel', pre=True)
    def lowercase_strings(cls, v):
        if isinstance(v, str):
            return v.lower().strip()
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "year": 2020,
                "odometer": 4500,
                "manufacturer": "ford",
                "model": "f-150",
                "fuel": "gas"
            }
        }


class PredictionResponse(BaseModel):
    """Response schema for predictions."""
    predicted_price: float
    model_used: str
    prediction_time_ms: float
    warnings: list = []


class HealthResponse(BaseModel):
    """Response schema for health check."""
    status: str
    model_loaded: bool
    model_name: str = ""
    uptime_seconds: float
    timestamp: str


class ModelInfoResponse(BaseModel):
    """Response schema for model info."""
    model_name: str
    pipeline_version: str
    trained_at: str
    metrics: dict
    n_features: int
    model_size_mb: float


# ============================================================================
# App Setup
# ============================================================================

_start_time = time.time()
_pipeline_loaded = False
_model_name = ""


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load pipeline components on startup."""
    global _pipeline_loaded, _model_name
    
    try:
        from predict import load_pipeline
        _, _, _, _, model_name = load_pipeline()
        _pipeline_loaded = True
        _model_name = model_name
        logger.info(f"Pipeline loaded successfully. Model: {model_name}")
    except Exception as e:
        logger.error(f"Failed to load pipeline on startup: {e}")
        _pipeline_loaded = False
    
    yield  # App is running
    
    # Cleanup on shutdown
    logger.info("Shutting down API server")


app = FastAPI(
    title="Used Car Price Prediction API",
    description="Predict the price of a used car based on its features.",
    version="2.0.0",
    lifespan=lifespan,
)


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy" if _pipeline_loaded else "degraded",
        model_loaded=_pipeline_loaded,
        model_name=_model_name,
        uptime_seconds=round(time.time() - _start_time, 2),
        timestamp=datetime.utcnow().isoformat() + "Z",
    )


@app.get("/model-info", response_model=ModelInfoResponse)
async def model_info():
    """Get current model metadata."""
    metadata_path = os.path.join(os.path.dirname(__file__), 'models', 'best_model_metadata.json')
    
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=404, detail="Model metadata not found")
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return ModelInfoResponse(
        model_name=metadata.get('model_name', 'unknown'),
        pipeline_version=metadata.get('pipeline_version', 'unknown'),
        trained_at=metadata.get('trained_at', 'unknown'),
        metrics=metadata.get('metrics', {}),
        n_features=metadata.get('n_features', 0),
        model_size_mb=metadata.get('model_size_mb', 0),
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(car: CarFeatures):
    """Predict the price of a used car."""
    if not _pipeline_loaded:
        raise HTTPException(
            status_code=503, 
            detail="Model not loaded. Service is starting up or encountered an error."
        )
    
    # Build input dict (only include non-None optional fields)
    input_data = {
        'year': car.year,
        'odometer': car.odometer,
        'manufacturer': car.manufacturer,
        'model': car.model,
        'fuel': car.fuel,
    }
    
    # Add optional fields if provided
    for field in ['condition', 'cylinders', 'title_status', 'transmission',
                  'drive', 'type', 'paint_color', 'state']:
        value = getattr(car, field, None)
        if value is not None:
            input_data[field] = value
    
    start_time = time.time()
    
    try:
        from predict import predict_price
        result = predict_price(input_data, validate=True)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    prediction_time = (time.time() - start_time) * 1000
    
    return PredictionResponse(
        predicted_price=round(result['predicted_price'], 2),
        model_used=result['model_used'],
        prediction_time_ms=round(prediction_time, 2),
        warnings=result.get('input_warnings', []),
    )


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
