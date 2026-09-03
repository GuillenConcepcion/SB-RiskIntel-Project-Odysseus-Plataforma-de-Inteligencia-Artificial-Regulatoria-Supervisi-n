"""Odysseus Hyperparameter Optimization & Tuning Engine.

Provides systematic Bayesian/Randomized & Grid Search Cross-Validation
to fine-tune Champion models for Early Warning System (EWS) and ProUsuario Forecaster.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import joblib
import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """Automated systematic hyperparameter optimization for SupTech models."""

    def __init__(self, models_dir: Path = MODELS_DIR, random_state: int = 42):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.random_state = random_state

    def tune_ews_classifier(
        self,
        df: Optional[pd.DataFrame] = None,
        n_iter: int = 25,
        cv_splits: int = 3,
    ) -> Tuple[Dict[str, Any], Any]:
        """Fine-tune Early Warning System (EWS) LightGBM & RandomForest classifiers."""
        if df is None:
            data_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"EWS dataset not found at {data_path}")
            df = pd.read_parquet(data_path)

        drop_cols = ["year", "quarter", "period", "target_high_risk_period", "supervisory_risk_index", "total_planes_regularizacion"]
        feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

        X = df[feature_cols].copy()
        y = df["target_high_risk_period"].copy()

        logger.info(f"Starting EWS Hyperparameter Tuning on {len(X)} samples with {cv_splits}-Fold CV...")

        # 1. Baseline Model Evaluation
        base_lgb = lgb.LGBMClassifier(random_state=self.random_state, verbose=-1)
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=self.random_state)

        # LightGBM Search Space
        lgb_param_distributions = {
            "n_estimators": [40, 60, 80, 120, 160],
            "max_depth": [3, 4, 5, 6, 8],
            "learning_rate": [0.01, 0.03, 0.05, 0.08, 0.1],
            "num_leaves": [7, 12, 15, 20, 31],
            "min_child_samples": [2, 3, 5, 8],
            "subsample": [0.6, 0.75, 0.85, 1.0],
            "colsample_bytree": [0.6, 0.75, 0.85, 1.0],
            "reg_alpha": [0.0, 0.01, 0.1, 1.0],
            "reg_lambda": [0.0, 0.01, 0.1, 1.0],
        }

        search_lgb = RandomizedSearchCV(
            estimator=base_lgb,
            param_distributions=lgb_param_distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=skf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )

        search_lgb.fit(X, y)
        best_lgb_params = search_lgb.best_params_
        best_lgb_score = float(search_lgb.best_score_)

        # 2. Tune Random Forest for comparison
        rf_param_distributions = {
            "n_estimators": [50, 80, 100, 150, 200],
            "max_depth": [3, 4, 5, 7, 10, None],
            "min_samples_split": [2, 3, 5, 8],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", None],
        }

        search_rf = RandomizedSearchCV(
            estimator=RandomForestClassifier(random_state=self.random_state),
            param_distributions=rf_param_distributions,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=skf,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search_rf.fit(X, y)
        best_rf_params = search_rf.best_params_
        best_rf_score = float(search_rf.best_score_)

        # Determine winner
        if best_lgb_score >= best_rf_score:
            champion_name = "Tuned_LightGBM"
            champion_model = search_lgb.best_estimator_
            champion_params = best_lgb_params
            best_auc = best_lgb_score
        else:
            champion_name = "Tuned_RandomForest"
            champion_model = search_rf.best_estimator_
            champion_params = best_rf_params
            best_auc = best_rf_score

        # Out-of-fold validation metrics for champion
        oof_probs = np.zeros(len(X))
        oof_preds = np.zeros(len(X))

        for train_idx, val_idx in skf.split(X, y):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_va = X.iloc[val_idx]
            champion_model.fit(X_tr, y_tr)
            probs = champion_model.predict_proba(X_va)[:, 1]
            oof_probs[val_idx] = probs
            oof_preds[val_idx] = (probs >= 0.5).astype(int)

        final_acc = float(accuracy_score(y, oof_preds))
        final_f1 = float(f1_score(y, oof_preds, zero_division=0))

        # Re-fit champion on 100% data
        champion_model.fit(X, y)

        tuning_summary = {
            "task": "hyperparameter_tuning_ews",
            "champion": champion_name,
            "best_cv_roc_auc": round(best_auc, 4),
            "cv_accuracy": round(final_acc, 4),
            "cv_f1_score": round(final_f1, 4),
            "lightgbm_best_score": round(best_lgb_score, 4),
            "lightgbm_best_params": best_lgb_params,
            "randomforest_best_score": round(best_rf_score, 4),
            "randomforest_best_params": {k: (v if v is not None else "None") for k, v in best_rf_params.items()},
            "champion_params": {k: (v if v is not None else "None") for k, v in champion_params.items()},
        }

        # Save artifacts
        with open(self.models_dir / "ews_tuning_summary.json", "w", encoding="utf-8") as f:
            json.dump(tuning_summary, f, indent=2)

        joblib.dump(champion_model, self.models_dir / "ews_tuned_champion.joblib")
        logger.info(f"EWS Tuning Complete | Champion: {champion_name} | Best CV ROC-AUC: {best_auc:.4f}")

        return tuning_summary, champion_model

    def tune_claims_forecaster(
        self,
        df: Optional[pd.DataFrame] = None,
        n_iter: int = 25,
    ) -> Tuple[Dict[str, Any], Any]:
        """Fine-tune ProUsuario Claims Regressor."""
        if df is None:
            data_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"Forecasting dataset not found at {data_path}")
            df = pd.read_parquet(data_path).sort_values(["year", "month_num"]).reset_index(drop=True)

        feature_cols = [
            "month_num", "month_sin", "month_cos", "quarter", "time_step",
            "reclamaciones_lag_1", "reclamaciones_lag_2", "reclamaciones_lag_3", "reclamaciones_lag_6",
            "monto_devolver_lag_1", "monto_devolver_lag_2", "monto_devolver_lag_3",
            "reclamaciones_roll_mean_3m", "reclamaciones_roll_std_3m",
            "monto_devolver_roll_mean_3m", "monto_devolver_roll_std_3m"
        ]

        X = df[feature_cols].copy()
        y = df["reclamaciones"].copy()

        logger.info(f"Starting ProUsuario Forecaster Hyperparameter Tuning on {len(X)} series records...")

        split_idx = max(len(df) - 12, int(len(df) * 0.8))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        # Search space for LightGBM Regressor
        lgb_param_distributions = {
            "n_estimators": [50, 80, 100, 150],
            "max_depth": [2, 3, 4, 5],
            "learning_rate": [0.01, 0.03, 0.05, 0.08],
            "num_leaves": [7, 10, 15, 20],
            "subsample": [0.7, 0.85, 1.0],
            "colsample_bytree": [0.7, 0.85, 1.0],
            "reg_alpha": [0.0, 0.1, 1.0],
            "reg_lambda": [0.0, 0.1, 1.0],
        }

        search_lgb = RandomizedSearchCV(
            estimator=lgb.LGBMRegressor(random_state=self.random_state, verbose=-1),
            param_distributions=lgb_param_distributions,
            n_iter=n_iter,
            scoring="neg_mean_absolute_error",
            cv=3,
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0,
        )
        search_lgb.fit(X_train, y_train)

        best_model = search_lgb.best_estimator_
        preds = best_model.predict(X_test)

        mae = float(mean_absolute_error(y_test, preds))
        r2 = float(r2_score(y_test, preds))
        wape = float(np.sum(np.abs(y_test - preds)) / (np.sum(y_test) + 1e-5))

        # Re-fit on full data
        best_model.fit(X, y)

        summary = {
            "task": "hyperparameter_tuning_claims_forecast",
            "model": "Tuned_LightGBM_Regressor",
            "holdout_wape": round(wape, 4),
            "holdout_mae": round(mae, 2),
            "holdout_r2": round(r2, 4),
            "best_params": search_lgb.best_params_,
        }

        with open(self.models_dir / "claims_tuning_summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        joblib.dump(best_model, self.models_dir / "claims_tuned_champion.joblib")
        logger.info(f"Forecaster Tuning Complete | Holdout WAPE: {wape:.2%} | Holdout MAE: {mae:.2f}")

        return summary, best_model


def run_full_hyperparameter_tuning():
    """Execute complete hyperparameter tuning pipeline."""
    tuner = HyperparameterTuner()
    print(">>> [Hyperparameter Tuning] Tuning Early Warning System (EWS)...", flush=True)
    ews_summary, _ = tuner.tune_ews_classifier(n_iter=25)

    print(">>> [Hyperparameter Tuning] Tuning ProUsuario Claims Forecaster...", flush=True)
    fc_summary, _ = tuner.tune_claims_forecaster(n_iter=25)

    # Safe MLflow logging
    try:
        mlflow.set_experiment("SB-RiskIntel-Supervision")
        with mlflow.start_run(run_name="Odysseus-Hyperparameter-Tuning"):
            mlflow.log_params(ews_summary["champion_params"])
            mlflow.log_metrics({
                "tuned_ews_roc_auc": ews_summary["best_cv_roc_auc"],
                "tuned_ews_f1": ews_summary["cv_f1_score"],
                "tuned_claims_wape": fc_summary["holdout_wape"],
                "tuned_claims_mae": fc_summary["holdout_mae"],
            })
            print(">>> [Hyperparameter Tuning] Logged tuning results to MLflow.", flush=True)
    except Exception as e:
        print(f">>> [Hyperparameter Tuning] MLflow logging note: {e}", flush=True)

    print(">>> [Hyperparameter Tuning] All tuning pipelines finished successfully.", flush=True)


if __name__ == "__main__":
    run_full_hyperparameter_tuning()
