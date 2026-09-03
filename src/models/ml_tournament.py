"""Odysseus Deep Machine Learning Tournament & Multi-Model Benchmarking Engine.

Provides automated training, cross-validation, hyperparameter comparison,
and champion model selection across diverse algorithm families for Banking Supervision.
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
from sklearn.ensemble import (
    ExtraTreesClassifier,
    ExtraTreesRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC, SVR

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class MLTournamentEngine:
    """Orchestrates comprehensive multi-model tournament for classification and regression."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run_classification_tournament(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: str = "target_high_risk_period",
        n_splits: int = 3,
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
        """Run supervised classification tournament for Early Warning System (EWS)."""
        if df is None:
            data_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
            if not data_path.exists():
                raise FileNotFoundError(f"EWS dataset not found at {data_path}")
            df = pd.read_parquet(data_path)

        drop_cols = [
            "year", "quarter", "period", target_col,
            "supervisory_risk_index", "total_planes_regularizacion"
        ]
        feature_cols = [c for c in df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(df[c])]

        X = df[feature_cols].copy()
        y = df[target_col].copy()

        logger.info(f"Initiating Classification Tournament | Samples: {len(X)} | Features: {len(feature_cols)}")

        # Candidate Model Definitions
        candidate_models = {
            "LightGBM": lgb.LGBMClassifier(
                n_estimators=60, max_depth=4, learning_rate=0.05, num_leaves=15,
                min_child_samples=2, subsample=0.8, colsample_bytree=0.8,
                random_state=random_state, verbose=-1
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=80, max_depth=5, min_samples_split=3,
                random_state=random_state
            ),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=80, max_depth=5, min_samples_split=2,
                random_state=random_state
            ),
            "HistGradientBoosting": HistGradientBoostingClassifier(
                max_iter=60, max_depth=4, learning_rate=0.05,
                random_state=random_state
            ),
            "LogisticRegression_L2": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(C=1.0, max_iter=500, random_state=random_state))
            ]),
            "SupportVectorClassifier": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", SVC(C=1.0, kernel="rbf", probability=True, random_state=random_state))
            ]),
            "MultiLayerPerceptron": Pipeline([
                ("scaler", StandardScaler()),
                ("clf", MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=400, random_state=random_state, early_stopping=False))
            ])
        }

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        leaderboard_rows = []
        fitted_models = {}

        for name, model in candidate_models.items():
            oof_probs = np.zeros(len(X))
            oof_preds = np.zeros(len(X))

            for train_idx, val_idx in skf.split(X, y):
                X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
                X_va = X.iloc[val_idx]

                model.fit(X_tr, y_tr)
                if hasattr(model, "predict_proba"):
                    probs = model.predict_proba(X_va)[:, 1]
                else:
                    probs = model.decision_function(X_va)
                    probs = (probs - probs.min()) / (probs.max() - probs.min() + 1e-5)

                oof_probs[val_idx] = probs
                oof_preds[val_idx] = (probs >= 0.5).astype(int)

            try:
                auc_val = float(roc_auc_score(y, oof_probs))
            except Exception:
                auc_val = 0.5

            acc = float(accuracy_score(y, oof_preds))
            prec = float(precision_score(y, oof_preds, zero_division=0))
            rec = float(recall_score(y, oof_preds, zero_division=0))
            f1 = float(f1_score(y, oof_preds, zero_division=0))
            brier = float(brier_score_loss(y, np.clip(oof_probs, 0, 1)))
            try:
                lloss = float(log_loss(y, np.clip(oof_probs, 1e-6, 1 - 1e-6)))
            except Exception:
                lloss = 0.5

            # Fit champion candidate on full data
            model.fit(X, y)
            fitted_models[name] = model

            # Composite Performance Index (70% ROC-AUC + 30% F1 - 10% Brier Penalty)
            perf_index = round((0.70 * auc_val + 0.30 * f1 - 0.10 * brier) * 100, 2)

            leaderboard_rows.append({
                "model_name": name,
                "roc_auc": round(auc_val, 4),
                "f1_score": round(f1, 4),
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "brier_score": round(brier, 4),
                "log_loss": round(lloss, 4),
                "performance_score": perf_index
            })

        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("performance_score", ascending=False).reset_index(drop=True)
        champion_name = leaderboard_df.iloc[0]["model_name"]
        champion_model = fitted_models[champion_name]

        logger.info(f"Classification Tournament Champion: {champion_name} (ROC-AUC={leaderboard_df.iloc[0]['roc_auc']}, F1={leaderboard_df.iloc[0]['f1_score']})")

        # Save tournament registry
        tournament_meta = {
            "task": "classification_ews",
            "champion": champion_name,
            "features": feature_cols,
            "total_samples": len(X),
            "leaderboard": leaderboard_df.to_dict(orient="records"),
        }

        with open(self.models_dir / "classification_tournament_meta.json", "w", encoding="utf-8") as f:
            json.dump(tournament_meta, f, indent=2)

        joblib.dump(champion_model, self.models_dir / "ews_champion_model.joblib")
        leaderboard_df.to_parquet(self.models_dir / "classification_leaderboard.parquet", index=False)

        return leaderboard_df, tournament_meta, champion_model

    def run_regression_tournament(
        self,
        df: Optional[pd.DataFrame] = None,
        target_col: str = "reclamaciones",
        random_state: int = 42,
    ) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
        """Run supervised regression tournament for ProUsuario Claims Forecasting."""
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
        y = df[target_col].copy()

        logger.info(f"Initiating Regression Tournament ({target_col}) | Samples: {len(X)} | Features: {len(feature_cols)}")

        split_idx = max(len(df) - 12, int(len(df) * 0.8))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        candidate_regressors = {
            "LightGBM_Regressor": lgb.LGBMRegressor(
                n_estimators=80, max_depth=4, learning_rate=0.04, num_leaves=15,
                random_state=random_state, verbose=-1
            ),
            "RandomForest_Regressor": RandomForestRegressor(
                n_estimators=80, max_depth=5, min_samples_split=3,
                random_state=random_state
            ),
            "ExtraTrees_Regressor": ExtraTreesRegressor(
                n_estimators=80, max_depth=5, min_samples_split=2,
                random_state=random_state
            ),
            "HistGradientBoosting_Regressor": HistGradientBoostingRegressor(
                max_iter=80, max_depth=4, learning_rate=0.04,
                random_state=random_state
            ),
            "Ridge_Regression": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", Ridge(alpha=1.0, random_state=random_state))
            ]),
            "SupportVector_Regressor": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", SVR(C=10.0, epsilon=0.1))
            ]),
            "MultiLayerPerceptron_Regressor": Pipeline([
                ("scaler", StandardScaler()),
                ("reg", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=random_state))
            ])
        }

        leaderboard_rows = []
        fitted_models = {}

        for name, model in candidate_regressors.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae = float(mean_absolute_error(y_test, preds))
            rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
            r2 = float(r2_score(y_test, preds))
            wape = float(np.sum(np.abs(y_test - preds)) / (np.sum(y_test) + 1e-5))

            # Fit on full data
            model.fit(X, y)
            fitted_models[name] = model

            # Performance score (Lower WAPE and RMSE is better, bounded 0-100)
            perf_score = round(max(0.0, 100.0 * (1.0 - wape)), 2)

            leaderboard_rows.append({
                "model_name": name,
                "wape": round(wape, 4),
                "mae": round(mae, 2),
                "rmse": round(rmse, 2),
                "r2_score": round(r2, 4),
                "forecast_efficiency_score": perf_score
            })

        leaderboard_df = pd.DataFrame(leaderboard_rows).sort_values("forecast_efficiency_score", ascending=False).reset_index(drop=True)
        champion_name = leaderboard_df.iloc[0]["model_name"]
        champion_model = fitted_models[champion_name]

        logger.info(f"Regression Tournament Champion ({target_col}): {champion_name} (WAPE={leaderboard_df.iloc[0]['wape']:.2%}, MAE={leaderboard_df.iloc[0]['mae']})")

        tournament_meta = {
            "task": f"regression_{target_col}",
            "champion": champion_name,
            "features": feature_cols,
            "total_samples": len(X),
            "leaderboard": leaderboard_df.to_dict(orient="records"),
        }

        with open(self.models_dir / f"regression_{target_col}_tournament_meta.json", "w", encoding="utf-8") as f:
            json.dump(tournament_meta, f, indent=2)

        joblib.dump(champion_model, self.models_dir / f"{target_col}_champion_model.joblib")
        leaderboard_df.to_parquet(self.models_dir / f"regression_{target_col}_leaderboard.parquet", index=False)

        return leaderboard_df, tournament_meta, champion_model


def run_all_tournaments():
    """Execute full tournament suite for classification and regression."""
    engine = MLTournamentEngine()
    print(">>> [ML Tournament] Running Early Warning System (EWS) Classification Tournament...", flush=True)
    clf_df, clf_meta, _ = engine.run_classification_tournament()

    print(">>> [ML Tournament] Running ProUsuario Claims Regression Tournament...", flush=True)
    reg_df, reg_meta, _ = engine.run_regression_tournament(target_col="reclamaciones")

    # Safe MLflow logging
    try:
        mlflow.set_experiment("SB-RiskIntel-Supervision")
        with mlflow.start_run(run_name="Odysseus-ML-Tournament"):
            mlflow.log_param("clf_champion", clf_meta["champion"])
            mlflow.log_param("reg_champion", reg_meta["champion"])
            mlflow.log_metrics({
                "clf_champion_auc": clf_df.iloc[0]["roc_auc"],
                "clf_champion_f1": clf_df.iloc[0]["f1_score"],
                "reg_champion_wape": reg_df.iloc[0]["wape"],
            })
            print(">>> [ML Tournament] Tournament results successfully logged to MLflow.", flush=True)
    except Exception as e:
        print(f">>> [ML Tournament] MLflow logging note: {e}", flush=True)

    print(">>> [ML Tournament] All tournaments completed successfully.", flush=True)


if __name__ == "__main__":
    run_all_tournaments()
