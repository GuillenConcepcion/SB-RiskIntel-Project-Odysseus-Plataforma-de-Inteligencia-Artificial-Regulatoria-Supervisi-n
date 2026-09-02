"""Unit tests for ML models and inference outputs."""

import json

import joblib
import numpy as np
import pandas as pd

from src.config.settings import MODELS_DIR


def test_ews_model_artifacts():
    """Verify that EWS model and metadata are saved and loadable."""
    model_path = MODELS_DIR / "ews_lightgbm_model.joblib"
    meta_path = MODELS_DIR / "ews_metadata.json"

    assert model_path.exists(), "EWS model file not found."
    assert meta_path.exists(), "EWS metadata file not found."

    model = joblib.load(model_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    assert "features" in meta
    assert len(meta["features"]) > 0

    # Synthetic sample prediction test
    features = meta["features"]
    sample_df = pd.DataFrame([np.random.rand(len(features))], columns=features)
    prob = model.predict_proba(sample_df)
    assert prob.shape == (1, 2)
    assert 0.0 <= prob[0, 1] <= 1.0


def test_forecaster_model_artifacts():
    """Verify that Forecaster models and metadata are saved and functional."""
    claims_path = MODELS_DIR / "claims_forecaster_model.joblib"
    monto_path = MODELS_DIR / "restitution_forecaster_model.joblib"
    meta_path = MODELS_DIR / "forecaster_metadata.json"

    assert claims_path.exists(), "Claims forecaster model not found."
    assert monto_path.exists(), "Restitution forecaster model not found."
    assert meta_path.exists(), "Forecaster metadata not found."

    claims_model = joblib.load(claims_path)
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    features = meta["features"]
    sample_df = pd.DataFrame([np.random.rand(len(features))], columns=features)
    pred_claim = claims_model.predict(sample_df)
    assert len(pred_claim) == 1
    assert pred_claim[0] >= 0
