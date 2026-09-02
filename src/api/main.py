"""FastAPI REST API Server for SB-RiskIntel Platform."""

import json
import logging
from contextlib import asynccontextmanager
from typing import Any, Dict, List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR, settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global model state
models_cache: Dict[str, Any] = {}


def load_models():
    """Load serialized models and metadata into memory."""
    try:
        ews_model_path = MODELS_DIR / "ews_lightgbm_model.joblib"
        ews_meta_path = MODELS_DIR / "ews_metadata.json"
        claims_model_path = MODELS_DIR / "claims_forecaster_model.joblib"
        restitution_model_path = MODELS_DIR / "restitution_forecaster_model.joblib"

        if ews_model_path.exists():
            models_cache["ews_model"] = joblib.load(ews_model_path)
            with open(ews_meta_path, "r", encoding="utf-8") as f:
                models_cache["ews_meta"] = json.load(f)

        if claims_model_path.exists():
            models_cache["claims_model"] = joblib.load(claims_model_path)
        if restitution_model_path.exists():
            models_cache["restitution_model"] = joblib.load(restitution_model_path)

        logger.info("Models loaded successfully into memory.")
    except Exception as e:
        logger.error(f"Error loading models: {e}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


app = FastAPI(
    title="SB-RiskIntel SupTech API",
    description="Early Warning System & Regulatory Conduct Risk Intelligence API - Superintendencia de Bancos de la República Dominicana",
    version=settings.VERSION,
    lifespan=lifespan,
    contact={
        "name": "Guillén Concepción (Senior Data Scientist & MLOps Engineer)",
        "email": "guillenconcepcion@gmail.com",
        "url": "https://www.linkedin.com/in/guillen-concepcion-25266b127",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas
class RiskScoreRequest(BaseModel):
    total_infracciones_imputadas: float = Field(..., ge=0, description="Quarterly infractions count")
    total_sanciones_impuestas: float = Field(..., ge=0, description="Quarterly sanctions count")
    monto_sanciones_dop: float = Field(..., ge=0, description="Quarterly sanctions amount (DOP)")
    total_procesos_sancionadores: float = Field(..., ge=0, description="Active sanctioning processes")
    total_solicitudes_aml: float = Field(..., ge=0, description="AML/CFT Ley 155-17 requests")
    total_inspecciones_eif: float = Field(..., ge=0, description="Quarterly inspections performed")


class RiskScoreResponse(BaseModel):
    high_risk_probability: float
    risk_level: str  # LOW, MODERATE, HIGH, CRITICAL
    supervisory_alert: bool
    top_contributing_factors: List[Dict[str, Any]]


class ForecastRequest(BaseModel):
    horizon_months: int = Field(default=6, ge=1, le=24, description="Forecast horizon in months")
    scenario_claim_multiplier: float = Field(default=1.0, ge=0.5, le=3.0, description="Stress test multiplier")


class ForecastResponse(BaseModel):
    horizon_months: int
    forecast_periods: List[str]
    predicted_claims: List[float]
    predicted_restitution_dop: List[float]
    total_projected_restitution_dop: float


@app.get("/health", tags=["System"])
def health_check():
    """Service health and loaded model status."""
    return {
        "status": "healthy",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "environment": settings.ENVIRONMENT,
        "models_loaded": list(models_cache.keys()),
        "author": "Guillén Concepción",
    }


@app.get("/api/v1/analytics/overview", tags=["Analytics"])
def get_system_overview():
    """Retrieve macro summary of supervisory statistics."""
    claims_path = PROCESSED_DATA_DIR / "prousuario_reclamaciones_cleaned.parquet"
    master_path = PROCESSED_DATA_DIR / "supervision_consolidated_quarterly.parquet"

    total_claims = 0
    total_restitution = 0.0
    avg_favorable_pct = 0.0
    total_sanctions = 0
    total_fines_dop = 0.0

    if claims_path.exists():
        df_c = pd.read_parquet(claims_path)
        total_claims = int(df_c["reclamaciones"].sum())
        total_restitution = float(df_c["monto_instruido_devolver"].sum())
        avg_favorable_pct = float(df_c["pct_favorable"].mean())

    if master_path.exists():
        df_m = pd.read_parquet(master_path)
        total_sanctions = int(df_m["total_sanciones_impuestas"].sum())
        total_fines_dop = float(df_m["monto_sanciones_dop"].sum())

    return {
        "period_range": "2017 - 2026",
        "consumer_protection": {
            "total_reclamaciones_atendidas": total_claims,
            "total_monto_instruido_devolver_dop": total_restitution,
            "tasa_favorable_usuario_promedio": round(avg_favorable_pct * 100, 2),
        },
        "supervisory_enforcement": {
            "total_sanciones_impuestas": total_sanctions,
            "total_multas_dop": total_fines_dop,
        },
        "system_status": "MONITORING_ACTIVE",
    }


@app.post("/api/v1/predict/risk-score", response_model=RiskScoreResponse, tags=["Early Warning System"])
def predict_risk_score(request: RiskScoreRequest):
    """Predict regulatory risk score and identify early warning triggers."""
    if "ews_model" not in models_cache:
        load_models()
        if "ews_model" not in models_cache:
            raise HTTPException(status_code=503, detail="EWS Model not loaded.")

    model = models_cache["ews_model"]
    meta = models_cache.get("ews_meta", {})
    feature_names = meta.get("features", [])

    input_dict = {
        "total_infracciones_imputadas": request.total_infracciones_imputadas,
        "total_sanciones_impuestas": request.total_sanciones_impuestas,
        "monto_sanciones_dop": request.monto_sanciones_dop,
        "total_procesos_sancionadores": request.total_procesos_sancionadores,
        "total_solicitudes_aml": request.total_solicitudes_aml,
        "total_inspecciones_eif": request.total_inspecciones_eif,
        "sanction_intensity": request.monto_sanciones_dop / (request.total_sanciones_impuestas + 1e-5),
        "infraction_per_inspection": request.total_infracciones_imputadas / (request.total_inspecciones_eif + 1e-5),
        "sanction_conversion_rate": request.total_sanciones_impuestas / (request.total_procesos_sancionadores + 1e-5),
        "aml_pressure_index": request.total_solicitudes_aml / (request.total_inspecciones_eif + 1e-5),
    }

    for f in feature_names:
        if f not in input_dict:
            input_dict[f] = 0.0

    X_input = pd.DataFrame([input_dict])[feature_names]
    prob = float(model.predict_proba(X_input)[0, 1])

    if prob >= 0.75:
        risk_level = "CRITICAL"
        alert = True
    elif prob >= 0.50:
        risk_level = "HIGH"
        alert = True
    elif prob >= 0.25:
        risk_level = "MODERATE"
        alert = False
    else:
        risk_level = "LOW"
        alert = False

    top_factors = meta.get("top_features", [])[:5]

    return RiskScoreResponse(
        high_risk_probability=round(prob, 4),
        risk_level=risk_level,
        supervisory_alert=alert,
        top_contributing_factors=top_factors,
    )


@app.post("/api/v1/forecast/claims", response_model=ForecastResponse, tags=["Forecasting Engine"])
def forecast_claims(request: ForecastRequest):
    """Generate multi-horizon forecasts for consumer claims and restitution amounts."""
    claims_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"
    if not claims_path.exists():
        raise HTTPException(status_code=404, detail="Forecasting features not found.")

    df = pd.read_parquet(claims_path).sort_values(["year", "month_num"]).reset_index(drop=True)
    last_row = df.iloc[-1]

    last_year = int(last_row["year"])
    last_month = int(last_row["month_num"])
    last_claim = float(last_row["reclamaciones"])
    last_monto = float(last_row["monto_instruido_devolver"])

    periods = []
    pred_claims = []
    pred_montos = []

    cur_year = last_year
    cur_month = last_month

    for i in range(1, request.horizon_months + 1):
        cur_month += 1
        if cur_month > 12:
            cur_month = 1
            cur_year += 1

        period_str = f"{cur_year}-{cur_month:02d}"
        periods.append(period_str)

        base_claim = (last_claim * (1.0 + 0.015 * i)) * request.scenario_claim_multiplier
        base_monto = (last_monto * (1.0 + 0.02 * i)) * request.scenario_claim_multiplier

        pred_claims.append(round(float(base_claim), 1))
        pred_montos.append(round(float(base_monto), 2))

    return ForecastResponse(
        horizon_months=request.horizon_months,
        forecast_periods=periods,
        predicted_claims=pred_claims,
        predicted_restitution_dop=pred_montos,
        total_projected_restitution_dop=round(sum(pred_montos), 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=settings.API_HOST, port=settings.API_PORT)
