"""Odysseus Data & Concept Drift Monitoring Engine for Banking Supervision.

Computes Population Stability Index (PSI), Two-Sample Kolmogorov-Smirnov (KS) tests,
and Wasserstein distances to detect distributional shifts and model degradation in production.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


def calculate_psi(
    reference: np.ndarray,
    target: np.ndarray,
    bins: int = 10,
    epsilon: float = 1e-4,
) -> float:
    """Calculate Population Stability Index (PSI) between reference and target distributions."""
    ref_clean = reference[~np.isnan(reference)]
    tgt_clean = target[~np.isnan(target)]

    if len(ref_clean) == 0 or len(tgt_clean) == 0:
        return 0.0

    # Determine quantile bins based on reference dataset
    quantiles = np.linspace(0, 100, bins + 1)
    bin_edges = np.percentile(ref_clean, quantiles)
    bin_edges = np.unique(bin_edges)  # ensure strict monotonicity

    if len(bin_edges) < 2:
        return 0.0

    # Ensure edge coverage
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf

    # Calculate frequency distributions
    ref_counts, _ = np.histogram(ref_clean, bins=bin_edges)
    tgt_counts, _ = np.histogram(tgt_clean, bins=bin_edges)

    ref_pct = ref_counts / (len(ref_clean) + epsilon)
    tgt_pct = tgt_counts / (len(tgt_clean) + epsilon)

    # Smooth zero frequencies with epsilon
    ref_pct = np.clip(ref_pct, epsilon, 1.0)
    tgt_pct = np.clip(tgt_pct, epsilon, 1.0)

    # PSI Formula: sum((Actual% - Expected%) * ln(Actual% / Expected%))
    psi_val = np.sum((tgt_pct - ref_pct) * np.log(tgt_pct / ref_pct))
    return float(max(0.0, psi_val))


def calculate_ks_2samp(
    reference: np.ndarray,
    target: np.ndarray,
) -> Tuple[float, float]:
    """Calculate Two-Sample Kolmogorov-Smirnov test statistic and p-value."""
    ref_clean = reference[~np.isnan(reference)]
    tgt_clean = target[~np.isnan(target)]

    if len(ref_clean) < 3 or len(tgt_clean) < 3:
        return 0.0, 1.0

    res = stats.ks_2samp(ref_clean, tgt_clean)
    return float(res.statistic), float(res.pvalue)


def calculate_wasserstein(
    reference: np.ndarray,
    target: np.ndarray,
) -> float:
    """Calculate Wasserstein Distance (Earth Mover's Distance)."""
    ref_clean = reference[~np.isnan(reference)]
    tgt_clean = target[~np.isnan(target)]

    if len(ref_clean) == 0 or len(tgt_clean) == 0:
        return 0.0

    return float(stats.wasserstein_distance(ref_clean, tgt_clean))


class DataDriftDetector:
    """Production monitor for distributional drift in regulatory supervisory features."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_drift(
        self,
        df_reference: Optional[pd.DataFrame] = None,
        df_target: Optional[pd.DataFrame] = None,
        feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Evaluate feature-by-feature drift and overall system stability."""
        if df_reference is None or df_target is None:
            data_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"EWS features dataset not found at {data_path}")
            full_df = pd.read_parquet(data_path).sort_values(["year", "quarter"]).reset_index(drop=True)

            # Split: Reference = historical baseline (first 70%), Target = recent quarters (last 30%)
            split_idx = max(8, int(len(full_df) * 0.70))
            df_reference = full_df.iloc[:split_idx]
            df_target = full_df.iloc[split_idx:]

        if feature_cols is None:
            drop_cols = ["year", "quarter", "period", "target_high_risk_period", "supervisory_risk_index", "total_planes_regularizacion"]
            feature_cols = [c for c in df_reference.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df_reference[c])]

        logger.info(f"Evaluating Data Drift across {len(feature_cols)} features | Reference N={len(df_reference)}, Target N={len(df_target)}")

        feature_reports = []
        psi_scores = []
        drifting_features_count = 0

        for col in feature_cols:
            ref_vals = df_reference[col].values
            tgt_vals = df_target[col].values

            psi_val = calculate_psi(ref_vals, tgt_vals)
            ks_stat, ks_pval = calculate_ks_2samp(ref_vals, tgt_vals)
            w_dist = calculate_wasserstein(ref_vals, tgt_vals)

            psi_scores.append(psi_val)

            # Categorize drift status
            if psi_val < 0.10 and ks_pval > 0.05:
                status = "ESTABLE"
                badge = "🟢 Estable"
            elif psi_val < 0.25:
                status = "DERIVA_MODERADA"
                badge = "🟡 Deriva Moderada"
            else:
                status = "DERIVA_SEVERA"
                badge = "🔴 Deriva Severa"
                drifting_features_count += 1

            feature_reports.append({
                "feature": col,
                "psi": round(psi_val, 4),
                "ks_statistic": round(ks_stat, 4),
                "ks_pvalue": round(ks_pval, 4),
                "wasserstein_distance": round(w_dist, 4),
                "status": status,
                "badge": badge,
                "ref_mean": round(float(np.nanmean(ref_vals)), 2),
                "tgt_mean": round(float(np.nanmean(tgt_vals)), 2),
            })

        # Sort features by highest PSI
        feature_reports.sort(key=lambda x: x["psi"], reverse=True)

        system_psi = float(np.mean(psi_scores)) if psi_scores else 0.0
        drift_ratio = drifting_features_count / len(feature_cols) if feature_cols else 0.0

        if system_psi < 0.10 and drift_ratio < 0.15:
            system_status = "HEALTHY_STABLE"
            system_badge = "🟢 Sistema Estable (Sin Deriva Crítica)"
            recommendation = "Continuar régimen de supervisión y monitoreo ordinario off-site."
            retrain_required = False
        elif system_psi < 0.20 and drift_ratio < 0.30:
            system_status = "WARNING_MODERATE_DRIFT"
            system_badge = "🟡 Alerta: Deriva Moderada Detectada"
            recommendation = "Revisar variables regulatorias con PSI > 0.10 y monitorear el próximo trimestre."
            retrain_required = False
        else:
            system_status = "CRITICAL_DRIFT_ALERT"
            system_badge = "🔴 Alerta Crítica: Deriva Severa en Producción"
            recommendation = "Disparar reentrenamiento automático y ajuste de umbrales en MLflow."
            retrain_required = True

        drift_report = {
            "task": "supervisory_data_drift_monitoring",
            "system_psi": round(system_psi, 4),
            "system_status": system_status,
            "system_badge": system_badge,
            "retrain_required": retrain_required,
            "recommendation": recommendation,
            "total_features_evaluated": len(feature_cols),
            "drifting_features_count": drifting_features_count,
            "reference_sample_size": len(df_reference),
            "target_sample_size": len(df_target),
            "features": feature_reports,
        }

        # Save artifact
        with open(self.models_dir / "data_drift_report.json", "w", encoding="utf-8") as f:
            json.dump(drift_report, f, indent=2)

        logger.info(f"Drift Evaluation Complete | System PSI: {system_psi:.4f} | Status: {system_status}")
        return drift_report


def run_drift_monitoring_pipeline():
    """Execute Data Drift monitoring pipeline."""
    detector = DataDriftDetector()
    report = detector.evaluate_drift()
    print(f">>> [Data Drift Engine] System PSI: {report['system_psi']} | Status: {report['system_status']}", flush=True)
    print(f">>> [Data Drift Engine] Recommendation: {report['recommendation']}", flush=True)
    print(">>> [Data Drift Engine] Top 5 Features Evaluated for Drift:", flush=True)
    for f in report["features"][:5]:
        print(f"  * {f['feature']}: PSI = {f['psi']} | KS p-val = {f['ks_pvalue']} | Estado = {f['status']}", flush=True)


if __name__ == "__main__":
    run_drift_monitoring_pipeline()
