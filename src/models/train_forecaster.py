"""Time Series Forecasting Engine for Consumer Claims & Restitution Amounts."""

import json
import logging
import sys

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def train_claims_forecaster():
    """Train gradient boosting autoregressive forecasters for Claims and Restitution Amount."""
    print(">>> [Forecaster Training] Loading dataset...", flush=True)
    data_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Feature dataset not found at {data_path}")

    df = pd.read_parquet(data_path).sort_values(["year", "month_num"]).reset_index(drop=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    feature_cols = [
        "month_num", "month_sin", "month_cos", "quarter", "time_step",
        "reclamaciones_lag_1", "reclamaciones_lag_2", "reclamaciones_lag_3", "reclamaciones_lag_6",
        "monto_devolver_lag_1", "monto_devolver_lag_2", "monto_devolver_lag_3",
        "reclamaciones_roll_mean_3m", "reclamaciones_roll_std_3m",
        "monto_devolver_roll_mean_3m", "monto_devolver_roll_std_3m"
    ]

    X = df[feature_cols].copy()
    y_claims = df["reclamaciones"].copy()
    y_monto = df["monto_instruido_devolver"].copy()

    print(f">>> [Forecaster Training] Training on {len(df)} monthly series records...", flush=True)

    split_idx = max(len(df) - 12, int(len(df) * 0.8))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_claims_tr, y_claims_te = y_claims.iloc[:split_idx], y_claims.iloc[split_idx:]
    y_monto_tr, y_monto_te = y_monto.iloc[:split_idx], y_monto.iloc[split_idx:]

    # 1. Claims Model
    claims_params = {
        "n_estimators": 80,
        "max_depth": 4,
        "learning_rate": 0.04,
        "num_leaves": 15,
        "random_state": 42,
        "verbose": -1
    }
    claims_model = lgb.LGBMRegressor(**claims_params)
    claims_model.fit(X_train, y_claims_tr)
    preds_claims = claims_model.predict(X_test)

    rmse_claims = float(np.sqrt(mean_squared_error(y_claims_te, preds_claims)))
    mae_claims = float(mean_absolute_error(y_claims_te, preds_claims))
    wape_claims = float(np.sum(np.abs(y_claims_te - preds_claims)) / (np.sum(y_claims_te) + 1e-5))

    # 2. Restitution Amount Model
    monto_params = {
        "n_estimators": 70,
        "max_depth": 3,
        "learning_rate": 0.05,
        "num_leaves": 10,
        "random_state": 42,
        "verbose": -1
    }
    monto_model = lgb.LGBMRegressor(**monto_params)
    monto_model.fit(X_train, np.log1p(y_monto_tr))
    preds_monto = np.expm1(monto_model.predict(X_test))

    rmse_monto = float(np.sqrt(mean_squared_error(y_monto_te, preds_monto)))
    mae_monto = float(mean_absolute_error(y_monto_te, preds_monto))

    metrics = {
        "claims_rmse": rmse_claims,
        "claims_mae": mae_claims,
        "claims_wape": wape_claims,
        "restitution_rmse_dop": rmse_monto,
        "restitution_mae_dop": mae_monto,
    }
    print(f">>> [Forecaster Training] Metrics: Claims MAE={mae_claims:.2f}, WAPE={wape_claims:.2%}, Restitution MAE=DOP${mae_monto:,.2f}", flush=True)

    # Train Full Models
    full_claims_model = lgb.LGBMRegressor(**claims_params)
    full_claims_model.fit(X, y_claims)

    full_monto_model = lgb.LGBMRegressor(**monto_params)
    full_monto_model.fit(X, np.log1p(y_monto))

    # Save artifacts
    claims_path = MODELS_DIR / "claims_forecaster_model.joblib"
    monto_path = MODELS_DIR / "restitution_forecaster_model.joblib"
    meta_path = MODELS_DIR / "forecaster_metadata.json"

    joblib.dump(full_claims_model, claims_path)
    joblib.dump(full_monto_model, monto_path)

    metadata = {
        "features": feature_cols,
        "metrics": metrics,
        "last_period": df.iloc[-1]["period"],
        "total_historical_months": len(df),
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    # MLflow logging (safe optional)
    try:
        mlflow.set_experiment("SB-RiskIntel-Supervision")
        with mlflow.start_run(run_name="Consumer-Protection-Forecaster"):
            mlflow.log_params({"claims_estimators": 80, "monto_estimators": 70})
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(claims_path))
            mlflow.log_artifact(str(monto_path))
            mlflow.log_artifact(str(meta_path))
            print(">>> [Forecaster Training] Logged experiment to MLflow.", flush=True)
    except Exception as e:
        print(f">>> [Forecaster Training] MLflow logging note: {e}", flush=True)

    print(">>> [Forecaster Training] Forecaster models successfully trained and registered.", flush=True)
    return full_claims_model, full_monto_model, metadata


if __name__ == "__main__":
    train_claims_forecaster()
