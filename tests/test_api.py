"""Integration tests for FastAPI endpoints."""

import pytest
from fastapi.testclient import TestClient

from src.api.main import app, load_models

client = TestClient(app)


@pytest.fixture(autouse=True)
def setup_api():
    load_models()


def test_health_endpoint():
    """Test /health status endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["author"] == "Guillén Concepción"


def test_analytics_overview_endpoint():
    """Test /api/v1/analytics/overview endpoint."""
    response = client.get("/api/v1/analytics/overview")
    assert response.status_code == 200
    data = response.json()
    assert "consumer_protection" in data
    assert "supervisory_enforcement" in data
    assert data["consumer_protection"]["total_reclamaciones_atendidas"] > 0


def test_predict_risk_score_endpoint():
    """Test /api/v1/predict/risk-score endpoint."""
    payload = {
        "total_infracciones_imputadas": 45.0,
        "total_sanciones_impuestas": 12.0,
        "monto_sanciones_dop": 2500000.0,
        "total_procesos_sancionadores": 20.0,
        "total_solicitudes_aml": 80.0,
        "total_inspecciones_eif": 30.0
    }
    response = client.post("/api/v1/predict/risk-score", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "high_risk_probability" in data
    assert "risk_level" in data
    assert data["risk_level"] in ["LOW", "MODERATE", "HIGH", "CRITICAL"]


def test_forecast_claims_endpoint():
    """Test /api/v1/forecast/claims endpoint."""
    payload = {
        "horizon_months": 6,
        "scenario_claim_multiplier": 1.0
    }
    response = client.post("/api/v1/forecast/claims", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["horizon_months"] == 6
    assert len(data["forecast_periods"]) == 6
    assert len(data["predicted_claims"]) == 6
    assert data["total_projected_restitution_dop"] > 0
