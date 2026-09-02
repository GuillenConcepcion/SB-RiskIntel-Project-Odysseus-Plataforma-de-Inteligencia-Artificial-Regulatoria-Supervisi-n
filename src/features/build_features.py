"""Feature Engineering pipeline for Supervisory Risk & Early Warning System."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def build_supervision_features(df_master: pd.DataFrame) -> pd.DataFrame:
    """Engineer risk indicators, ratios, and lagged features for Early Warning System."""
    df = df_master.copy().sort_values(["year", "quarter"]).reset_index(drop=True)

    # 1. Ratios and Intensity Metrics
    df["sanction_intensity"] = df["monto_sanciones_dop"] / (df["total_sanciones_impuestas"] + 1e-5)
    df["infraction_per_inspection"] = df["total_infracciones_imputadas"] / (df["total_inspecciones_eif"] + 1e-5)
    df["sanction_conversion_rate"] = df["total_sanciones_impuestas"] / (df["total_procesos_sancionadores"] + 1e-5)
    df["aml_pressure_index"] = df["total_solicitudes_aml"] / (df["total_inspecciones_eif"] + 1e-5)

    # 2. Lagged features (t-1, t-2)
    for lag in [1, 2]:
        df[f"infracciones_lag_{lag}"] = df["total_infracciones_imputadas"].shift(lag).fillna(0)
        df[f"sanciones_lag_{lag}"] = df["total_sanciones_impuestas"].shift(lag).fillna(0)
        df[f"monto_sanciones_lag_{lag}"] = df["monto_sanciones_dop"].shift(lag).fillna(0)
        df[f"aml_solicitudes_lag_{lag}"] = df["total_solicitudes_aml"].shift(lag).fillna(0)
        df[f"procesos_sancionadores_lag_{lag}"] = df["total_procesos_sancionadores"].shift(lag).fillna(0)

    # 3. Rolling Moving Averages & Momentum
    df["infracciones_roll_mean_4q"] = df["total_infracciones_imputadas"].rolling(4, min_periods=1).mean()
    df["sanciones_roll_mean_4q"] = df["total_sanciones_impuestas"].rolling(4, min_periods=1).mean()
    df["monto_sanciones_roll_mean_4q"] = df["monto_sanciones_dop"].rolling(4, min_periods=1).mean()
    df["aml_roll_mean_4q"] = df["total_solicitudes_aml"].rolling(4, min_periods=1).mean()

    # Momentum (current vs rolling average)
    df["infracciones_momentum"] = df["total_infracciones_imputadas"] / (df["infracciones_roll_mean_4q"] + 1e-5)
    df["aml_momentum"] = df["total_solicitudes_aml"] / (df["aml_roll_mean_4q"] + 1e-5)

    # 4. Target Label for Early Warning System (EWS)
    # High risk definition: Quarters with elevated regularisation requirements or top 35% sanction fines
    monto_threshold = df["monto_sanciones_dop"].quantile(0.65)
    df["target_high_risk_period"] = (
        (df["total_planes_regularizacion"] > 0) | (df["monto_sanciones_dop"] >= monto_threshold)
    ).astype(int)

    # Composite Risk Score [0 - 100]
    norm_sanc = (df["total_sanciones_impuestas"] - df["total_sanciones_impuestas"].min()) / (df["total_sanciones_impuestas"].max() - df["total_sanciones_impuestas"].min() + 1e-5)
    norm_inf = (df["total_infracciones_imputadas"] - df["total_infracciones_imputadas"].min()) / (df["total_infracciones_imputadas"].max() - df["total_infracciones_imputadas"].min() + 1e-5)
    norm_aml = (df["total_solicitudes_aml"] - df["total_solicitudes_aml"].min()) / (df["total_solicitudes_aml"].max() - df["total_solicitudes_aml"].min() + 1e-5)

    df["supervisory_risk_index"] = ((0.4 * norm_sanc + 0.35 * norm_inf + 0.25 * norm_aml) * 100).round(2)
    return df


def build_claims_forecast_features(df_claims: pd.DataFrame) -> pd.DataFrame:
    """Engineer time-series features for consumer protection forecasting."""
    df = df_claims.copy().sort_values(["year", "month_num"]).reset_index(drop=True)

    # Cyclical calendar features
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)
    df["quarter"] = ((df["month_num"] - 1) // 3) + 1

    # Time index
    df["time_step"] = np.arange(len(df))

    # Lags for claims and restitution amounts
    for lag in [1, 2, 3, 6, 12]:
        df[f"reclamaciones_lag_{lag}"] = df["reclamaciones"].shift(lag).bfill()
        df[f"monto_devolver_lag_{lag}"] = df["monto_instruido_devolver"].shift(lag).bfill()
        df[f"pct_favorable_lag_{lag}"] = df["pct_favorable"].shift(lag).bfill()

    # Rolling statistics
    df["reclamaciones_roll_mean_3m"] = df["reclamaciones"].rolling(3, min_periods=1).mean()
    df["reclamaciones_roll_std_3m"] = df["reclamaciones"].rolling(3, min_periods=1).std().fillna(0)
    df["monto_devolver_roll_mean_3m"] = df["monto_instruido_devolver"].rolling(3, min_periods=1).mean()
    df["monto_devolver_roll_std_3m"] = df["monto_instruido_devolver"].rolling(3, min_periods=1).std().fillna(0)

    # Conduct Risk Index (CRI) at consumer level
    # High dissatisfaction = low % favorable + high monetary volume
    df["consumer_conduct_risk_index"] = (
        (1.0 - df["pct_favorable"]) * np.log1p(df["monto_instruido_devolver"])
    ).round(3)

    return df


def generate_all_features(data_dir: Path = PROCESSED_DATA_DIR):
    """Generate and save engineered feature tables."""
    logger.info("Generating engineered feature tables...")

    # Supervision EWS features
    master_path = data_dir / "supervision_consolidated_quarterly.parquet"
    if master_path.exists():
        df_master = pd.read_parquet(master_path)
        df_ews_features = build_supervision_features(df_master)
        df_ews_features.to_parquet(data_dir / "features_supervision_ews.parquet", index=False)
        logger.info(f"Saved: {data_dir / 'features_supervision_ews.parquet'} ({df_ews_features.shape[1]} features)")

    # Claims Forecaster features
    claims_path = data_dir / "prousuario_reclamaciones_cleaned.parquet"
    if claims_path.exists():
        df_claims = pd.read_parquet(claims_path)
        df_claims_features = build_claims_forecast_features(df_claims)
        df_claims_features.to_parquet(data_dir / "features_claims_forecast.parquet", index=False)
        logger.info(f"Saved: {data_dir / 'features_claims_forecast.parquet'} ({df_claims_features.shape[1]} features)")


if __name__ == "__main__":
    generate_all_features()
