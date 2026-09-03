"""Odysseus Unsupervised Anomaly Detection Engine for Regulatory Supervision.

Combines Isolation Forest, Local Outlier Factor (LOF), One-Class SVM,
and PCA Reconstruction Error to identify anomalous quarters and risk spikes.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.svm import OneClassSVM

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class SupervisoryAnomalyDetector:
    """Multi-algorithm unsupervised anomaly detection engine for SupTech regulatory data."""

    def __init__(
        self,
        contamination: float = 0.15,
        random_state: int = 42,
        models_dir: Path = MODELS_DIR,
    ):
        self.contamination = contamination
        self.random_state = random_state
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

        self.scaler = RobustScaler()
        self.iforest = IsolationForest(
            contamination=contamination,
            n_estimators=100,
            random_state=random_state,
        )
        self.ocsvm = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
        self.pca = PCA(n_components=0.90, random_state=random_state)
        self.feature_names_: List[str] = []
        self.fitted_ = False

    def fit_predict(
        self,
        df: Optional[pd.DataFrame] = None,
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Fit anomaly detection ensemble and score all records."""
        if df is None:
            data_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Feature dataset not found at {data_path}")
            df = pd.read_parquet(data_path)

        drop_cols = ["year", "quarter", "period", "target_high_risk_period", "supervisory_risk_index"]
        self.feature_names_ = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

        X_raw = df[self.feature_names_].values
        X_scaled = self.scaler.fit_transform(X_raw)

        logger.info(f"Fitting Anomaly Detectors on {len(df)} records across {len(self.feature_names_)} features...")

        # 1. Isolation Forest
        self.iforest.fit(X_scaled)
        iforest_raw = -self.iforest.score_samples(X_scaled)  # higher = more anomalous
        iforest_norm = (iforest_raw - iforest_raw.min()) / (iforest_raw.max() - iforest_raw.min() + 1e-6)

        # 2. Local Outlier Factor (Novelty / Local density)
        lof = LocalOutlierFactor(n_neighbors=min(5, len(df) - 1), contamination=self.contamination)
        lof.fit_predict(X_scaled)
        lof_raw = -lof.negative_outlier_factor_
        lof_norm = (lof_raw - lof_raw.min()) / (lof_raw.max() - lof_raw.min() + 1e-6)

        # 3. One-Class SVM
        self.ocsvm.fit(X_scaled)
        ocsvm_dist = -self.ocsvm.decision_function(X_scaled)
        ocsvm_norm = (ocsvm_dist - ocsvm_dist.min()) / (ocsvm_dist.max() - ocsvm_dist.min() + 1e-6)

        # 4. PCA Reconstruction Error
        X_pca = self.pca.fit_transform(X_scaled)
        X_reconstructed = self.pca.inverse_transform(X_pca)
        reconstruction_error = np.mean(np.square(X_scaled - X_reconstructed), axis=1)
        pca_norm = (reconstruction_error - reconstruction_error.min()) / (reconstruction_error.max() - reconstruction_error.min() + 1e-6)

        # Composite Supervisory Anomaly Score [0 - 100]
        composite_score = (
            0.35 * iforest_norm + 0.25 * lof_norm + 0.20 * ocsvm_norm + 0.20 * pca_norm
        ) * 100.0

        # Anomaly Classification Flag (Top ~15-20% or score >= 60.0)
        threshold_val = float(np.percentile(composite_score, (1.0 - self.contamination) * 100))
        anomaly_flags = (composite_score >= max(50.0, threshold_val)).astype(int)

        # Build output dataframe
        result_df = df.copy()
        result_df["anomaly_score_composite"] = np.round(composite_score, 2)
        result_df["anomaly_iforest_norm"] = np.round(iforest_norm * 100, 2)
        result_df["anomaly_lof_norm"] = np.round(lof_norm * 100, 2)
        result_df["anomaly_ocsvm_norm"] = np.round(ocsvm_norm * 100, 2)
        result_df["anomaly_pca_norm"] = np.round(pca_norm * 100, 2)
        result_df["is_regulatory_anomaly"] = anomaly_flags

        # Find top driver features per anomalous row
        z_scores = np.abs((X_raw - np.mean(X_raw, axis=0)) / (np.std(X_raw, axis=0) + 1e-5))
        top_drivers = []
        for i in range(len(df)):
            row_z = z_scores[i]
            top_3_idx = np.argsort(row_z)[-3:][::-1]
            drivers = [f"{self.feature_names_[idx]} (Z={row_z[idx]:.1f})" for idx in top_3_idx]
            top_drivers.append(", ".join(drivers))

        result_df["top_anomaly_drivers"] = top_drivers

        self.fitted_ = True

        # Save artifacts
        meta = {
            "task": "supervisory_anomaly_detection",
            "contamination": self.contamination,
            "threshold": round(threshold_val, 2),
            "total_records": len(df),
            "detected_anomalies": int(np.sum(anomaly_flags)),
            "features": self.feature_names_,
            "pca_explained_variance_ratio": [float(x) for x in self.pca.explained_variance_ratio_],
        }

        with open(self.models_dir / "anomaly_detector_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        joblib.dump(self, self.models_dir / "supervisory_anomaly_detector.joblib")
        result_df.to_parquet(self.models_dir / "supervisory_anomaly_scored.parquet", index=False)

        logger.info(f"Anomaly Detection Complete | Total Anomalies Detected: {meta['detected_anomalies']} / {len(df)}")
        return result_df, meta


def run_anomaly_pipeline():
    """Execute unsupervised anomaly detection pipeline."""
    detector = SupervisoryAnomalyDetector(contamination=0.15)
    scored_df, meta = detector.fit_predict()
    print(f">>> [Anomaly Detection] Processed {len(scored_df)} periods. Found {meta['detected_anomalies']} anomalous periods.", flush=True)
    anom_sample = scored_df[scored_df["is_regulatory_anomaly"] == 1][["period", "anomaly_score_composite", "top_anomaly_drivers"]]
    print(">>> [Anomaly Detection] Top Outlier Periods Identified:\n", anom_sample.to_string(index=False), flush=True)


if __name__ == "__main__":
    run_anomaly_pipeline()
