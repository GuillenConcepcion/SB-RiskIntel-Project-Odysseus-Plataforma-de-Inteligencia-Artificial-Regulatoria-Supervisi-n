"""Odysseus Explainable AI (XAI) & Interpretability Engine.

Provides unified global & local SHAP explanations, Permutation Importance,
and Partial Dependence Profiles for regulatory compliance and auditability.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.inspection import partial_dependence, permutation_importance

from src.analytics.cache import cached_inference
from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class ExplainabilityEngine:
    """Unified XAI engine for SupTech models (EWS Classifier and ProUsuario Forecaster)."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    @cached_inference(namespace="xai_instance", ttl_seconds=3600)
    def explain_instance(self, input_features: Dict[str, float]) -> Dict[str, Any]:
        """Compute fast local SHAP feature attribution for a single supervisory instance (cached)."""
        model_path = self.models_dir / "ews_champion_model.joblib"
        if not model_path.exists():
            model_path = self.models_dir / "ews_lightgbm_model.joblib"
        model = joblib.load(model_path)

        meta_path = self.models_dir / "ews_xai_profile.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                xai_meta = json.load(f)
            feature_names = xai_meta.get("features", list(input_features.keys()))
        else:
            feature_names = list(input_features.keys())

        # Build single row vector
        row = [input_features.get(f, 0.0) for f in feature_names]
        X_df = pd.DataFrame([row], columns=feature_names)

        prob = float(model.predict_proba(X_df)[0, 1])

        try:
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_df)
            if isinstance(shap_vals, list) and len(shap_vals) > 1:
                sv = shap_vals[1][0]
            elif isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 3:
                sv = shap_vals[0, :, 1]
            else:
                sv = shap_vals[0]
            base_val = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        except Exception:
            sv = np.zeros(len(feature_names))
            base_val = 0.5

        contributions = [
            {
                "feature": f,
                "value": round(float(v), 2),
                "shap_impact": round(float(s), 4),
            }
            for f, v, s in zip(feature_names, row, sv, strict=False)
        ]
        contributions.sort(key=lambda x: abs(x["shap_impact"]), reverse=True)

        return {
            "predicted_risk_probability": round(prob, 4),
            "base_expected_value": round(base_val, 4),
            "top_drivers": contributions[:6],
        }


    def explain_ews_model(
        self,
        model: Optional[Any] = None,
        df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """Compute comprehensive SHAP, permutation importance, and partial dependence for EWS."""
        if model is None:
            model_path = self.models_dir / "ews_champion_model.joblib"
            if not model_path.exists():
                model_path = self.models_dir / "ews_lightgbm_model.joblib"
            if not model_path.exists():
                raise FileNotFoundError(f"Model not found at {model_path}")
            model = joblib.load(model_path)

        if df is None:
            data_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset not found at {data_path}")
            df = pd.read_parquet(data_path)

        drop_cols = ["year", "quarter", "period", "target_high_risk_period", "supervisory_risk_index", "total_planes_regularizacion"]
        feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

        X = df[feature_cols].copy()
        y = df["target_high_risk_period"].copy()

        logger.info(f"Computing XAI Profiles for EWS Model on {len(X)} samples across {len(feature_cols)} features...")

        # 1. SHAP Values
        try:
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
            if isinstance(shap_values, list) and len(shap_values) > 1:
                # Binary classification [class 0, class 1]
                sv_matrix = shap_values[1]
            elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
                sv_matrix = shap_values[:, :, 1]
            else:
                sv_matrix = shap_values
            expected_val = float(explainer.expected_value[1] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value)
        except Exception as e:
            logger.warning(f"TreeExplainer fallback to Kernel/Permutation: {e}")
            background = shap.sample(X, min(10, len(X)))
            explainer = shap.KernelExplainer(lambda data: model.predict_proba(data)[:, 1], background)
            sv_matrix = explainer.shap_values(X)
            expected_val = float(explainer.expected_value)

        # Mean Absolute SHAP Importance
        mean_abs_shap = np.mean(np.abs(sv_matrix), axis=0)
        shap_importance = [
            {"feature": f, "mean_abs_shap": round(float(m), 4)}
            for f, m in zip(feature_cols, mean_abs_shap, strict=False)
        ]
        shap_importance.sort(key=lambda x: x["mean_abs_shap"], reverse=True)

        # 2. Permutation Importance
        perm_res = permutation_importance(model, X, y, n_repeats=5, random_state=42, scoring="roc_auc")
        perm_importance = [
            {"feature": f, "importance_mean": round(float(m), 4), "importance_std": round(float(s), 4)}
            for f, m, s in zip(feature_cols, perm_res.importances_mean, perm_res.importances_std, strict=False)
        ]
        perm_importance.sort(key=lambda x: x["importance_mean"], reverse=True)

        # 3. Partial Dependence for Top 4 Features
        top_features = [item["feature"] for item in shap_importance[:4]]
        pdp_profiles = {}

        for feat in top_features:
            try:
                pdp_res = partial_dependence(model, X, [feat], grid_resolution=15)
                # handle scikit-learn pdp outputs
                grid_vals = pdp_res["grid_values"][0] if "grid_values" in pdp_res else pdp_res[1][0]
                pdp_vals = pdp_res["average"][0] if "average" in pdp_res else pdp_res[0][0]
                pdp_profiles[feat] = {
                    "grid": [round(float(v), 2) for v in grid_vals],
                    "pdp_values": [round(float(v), 4) for v in pdp_vals],
                }
            except Exception as e:
                logger.warning(f"PDP calculation error for {feat}: {e}")

        # Local sample explanations for latest period
        latest_idx = len(X) - 1
        latest_period = str(df.iloc[latest_idx]["period"])
        local_shap = [
            {
                "feature": f,
                "feature_value": float(X.iloc[latest_idx][f]),
                "shap_contribution": round(float(sv_matrix[latest_idx, i]), 4),
            }
            for i, f in enumerate(feature_cols)
        ]
        local_shap.sort(key=lambda x: abs(x["shap_contribution"]), reverse=True)

        xai_summary = {
            "model_type": type(model).__name__,
            "features": feature_cols,
            "base_expected_value": round(expected_val, 4),
            "global_shap_importance": shap_importance,
            "permutation_importance": perm_importance,
            "partial_dependence_profiles": pdp_profiles,
            "latest_period_explanation": {
                "period": latest_period,
                "local_contributions": local_shap[:8],
            },
        }

        # Save to models_registry
        with open(self.models_dir / "ews_xai_profile.json", "w", encoding="utf-8") as f:
            json.dump(xai_summary, f, indent=2)

        # Save SHAP matrix for fast dashboard querying
        shap_df = pd.DataFrame(sv_matrix, columns=[f"shap_{c}" for c in feature_cols])
        shap_df["period"] = df["period"].values
        shap_df.to_parquet(self.models_dir / "ews_shap_values.parquet", index=False)

        logger.info("XAI Engine Successfully Generated and Saved Explanations.")
        return xai_summary


def run_xai_pipeline():
    """Execute XAI pipeline."""
    engine = ExplainabilityEngine()
    summary = engine.explain_ews_model()
    print(f">>> [XAI Engine] Base expected value: {summary['base_expected_value']}", flush=True)
    print(">>> [XAI Engine] Top 5 Drivers of Supervisory High-Risk Periods (Global SHAP):")
    for item in summary["global_shap_importance"][:5]:
        print(f"  * {item['feature']}: Mean |SHAP| = {item['mean_abs_shap']}", flush=True)


if __name__ == "__main__":
    run_xai_pipeline()
