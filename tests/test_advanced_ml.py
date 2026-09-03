"""Unit & Integration Tests for Odysseus Deep Machine Learning & SupTech Intelligence."""


from fastapi.testclient import TestClient

from src.api.main import app
from src.config.settings import PROCESSED_DATA_DIR
from src.models.anomaly_detection import SupervisoryAnomalyDetector
from src.models.clustering import SupervisoryClusterEngine
from src.models.explainability import ExplainabilityEngine
from src.models.ml_tournament import MLTournamentEngine

client = TestClient(app)


def test_classification_tournament():
    """Test supervised classification tournament for EWS."""
    data_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
    assert data_path.exists(), "EWS dataset must exist"

    engine = MLTournamentEngine()
    leaderboard_df, meta, champion_model = engine.run_classification_tournament(n_splits=2)

    assert not leaderboard_df.empty
    assert "model_name" in leaderboard_df.columns
    assert "roc_auc" in leaderboard_df.columns
    assert "performance_score" in leaderboard_df.columns
    assert meta["champion"] in leaderboard_df["model_name"].values
    assert hasattr(champion_model, "predict")


def test_regression_tournament():
    """Test supervised regression tournament for ProUsuario Claims."""
    data_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"
    assert data_path.exists(), "Forecasting dataset must exist"

    engine = MLTournamentEngine()
    leaderboard_df, meta, champion_model = engine.run_regression_tournament(target_col="reclamaciones")

    assert not leaderboard_df.empty
    assert "model_name" in leaderboard_df.columns
    assert "wape" in leaderboard_df.columns
    assert "forecast_efficiency_score" in leaderboard_df.columns
    assert hasattr(champion_model, "predict")


def test_unsupervised_anomaly_detector():
    """Test multi-algorithm anomaly detection engine."""
    detector = SupervisoryAnomalyDetector(contamination=0.15)
    scored_df, meta = detector.fit_predict()

    assert not scored_df.empty
    assert "anomaly_score_composite" in scored_df.columns
    assert "is_regulatory_anomaly" in scored_df.columns
    assert "top_anomaly_drivers" in scored_df.columns
    assert meta["total_records"] == len(scored_df)
    assert 0.0 <= scored_df["anomaly_score_composite"].min()
    assert scored_df["anomaly_score_composite"].max() <= 100.0


def test_unsupervised_clustering_engine():
    """Test behavioral clustering and latent space PCA engine."""
    engine = SupervisoryClusterEngine()
    clustered_df, meta = engine.fit_predict_clusters(k_range=(2, 4))

    assert not clustered_df.empty
    assert "cluster_id" in clustered_df.columns
    assert "cluster_archetype" in clustered_df.columns
    assert "pca_1" in clustered_df.columns
    assert "pca_2" in clustered_df.columns
    assert "pca_3" in clustered_df.columns
    assert meta["optimal_k"] >= 2
    assert len(meta["cluster_profiles"]) == meta["optimal_k"]


def test_explainability_engine():
    """Test XAI global & local SHAP, Permutation, and PDP profiles."""
    engine = ExplainabilityEngine()
    xai_summary = engine.explain_ews_model()

    assert "global_shap_importance" in xai_summary
    assert "permutation_importance" in xai_summary
    assert "partial_dependence_profiles" in xai_summary
    assert len(xai_summary["global_shap_importance"]) > 0
    assert "base_expected_value" in xai_summary


def test_api_ml_endpoints():
    """Test FastAPI Deep ML endpoints."""
    # 1. Tournament Leaderboard
    res_lead = client.get("/api/v1/ml/tournament-leaderboard")
    assert res_lead.status_code == 200
    data_lead = res_lead.json()
    assert data_lead["status"] == "success"
    assert "classification_ews" in data_lead

    # 2. Supervisory Clusters
    res_clust = client.get("/api/v1/ml/supervisory-clusters")
    assert res_clust.status_code == 200
    data_clust = res_clust.json()
    assert data_clust["status"] == "success"
    assert "clustered_periods" in data_clust

    # 3. Anomalies
    res_anom = client.get("/api/v1/ml/anomalies")
    assert res_anom.status_code == 200
    data_anom = res_anom.json()
    assert data_anom["status"] == "success"
    assert "anomalous_periods" in data_anom

    # 4. Explainability
    res_xai = client.get("/api/v1/ml/explainability")
    assert res_xai.status_code == 200
    data_xai = res_xai.json()
    assert data_xai["status"] == "success"
    assert "xai_profile" in data_xai

    # 5. Hyperparameter Tuning Endpoint
    res_tune = client.get("/api/v1/ml/hyperparameter-tuning")
    assert res_tune.status_code == 200
    data_tune = res_tune.json()
    assert data_tune["status"] == "success"
    assert "ews_tuning" in data_tune


def test_hyperparameter_tuner_execution():
    """Test HyperparameterTuner optimizer module."""
    from src.models.hyperparameter_tuning import HyperparameterTuner

    tuner = HyperparameterTuner()
    ews_res, model = tuner.tune_ews_classifier(n_iter=2, cv_splits=2)

    assert "champion" in ews_res
    assert "best_cv_roc_auc" in ews_res
    assert hasattr(model, "predict")


def test_conformal_forecaster():
    """Test Split-Conformal Prediction calibration and intervals."""
    from src.models.conformal_forecaster import ConformalForecaster

    cf = ConformalForecaster()
    meta = cf.calibrate()

    assert "quantiles" in meta
    assert meta["quantiles"]["reclamaciones"]["q95"] >= meta["quantiles"]["reclamaciones"]["q90"]
    assert meta["quantiles"]["monto_devolver_dop"]["q95"] >= meta["quantiles"]["monto_devolver_dop"]["q90"]

    res = cf.predict_intervals(horizon_months=6)
    assert len(res["forecast_intervals"]) == 6
    for row in res["forecast_intervals"]:
        assert row["claims_lower_95"] <= row["claims_point"] <= row["claims_upper_95"]
        assert row["restitution_dop_lower_95"] <= row["restitution_dop_point"] <= row["restitution_dop_upper_95"]


def test_conformal_api_endpoint():
    """Test /api/v1/forecast/conformal endpoint."""
    payload = {"horizon_months": 6, "scenario_multiplier": 1.0}
    res = client.post("/api/v1/forecast/conformal", json=payload)

    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "total_conformal_95_restitution_dop" in data
    assert len(data["forecast_intervals"]) == 6


def test_ml_inference_cache():
    """Test Multi-Tier LRU Cache and deterministic hashing."""
    from src.analytics.cache import MLInferenceCache, make_cache_key

    cache = MLInferenceCache(max_size=3, default_ttl_seconds=10)
    key1 = make_cache_key("test", a=1, b="hello")
    key2 = make_cache_key("test", a=1, b="hello")
    assert key1 == key2, "Hash must be deterministic"

    # Set and Get
    cache.set(key1, {"result": 42})
    assert cache.get(key1) == {"result": 42}
    assert cache.hits == 1

    # Miss
    assert cache.get("non_existent_key") is None
    assert cache.misses == 1

    # LRU Eviction test
    cache.set("k2", 2)
    cache.set("k3", 3)
    cache.set("k4", 4)  # should evict key1
    assert cache.get(key1) is None
    assert cache.evictions >= 1

    # Stats
    stats = cache.get_stats()
    assert stats["tier1_memory_items"] <= 3
    assert stats["status"] == "OPERATIONAL"


def test_cache_and_instance_explain_api_endpoints():
    """Test /api/v1/cache/stats, /api/v1/cache/clear, and /api/v1/ml/explain-instance."""
    # 1. Instance explain (cached)
    payload = {
        "features": {
            "total_infracciones_imputadas": 80.0,
            "total_sanciones_impuestas": 15.0,
            "monto_sanciones_dop": 5000000.0,
            "total_procesos_sancionadores": 18.0,
            "total_solicitudes_aml": 4.0,
            "total_inspecciones_eif": 12.0,
        }
    }
    res_xai1 = client.post("/api/v1/ml/explain-instance", json=payload)
    assert res_xai1.status_code == 200
    data_xai1 = res_xai1.json()
    assert data_xai1["status"] == "success"
    assert "top_drivers" in data_xai1

    # Second call should hit cache
    res_xai2 = client.post("/api/v1/ml/explain-instance", json=payload)
    assert res_xai2.status_code == 200

    # 2. Cache stats
    res_stats = client.get("/api/v1/cache/stats")
    assert res_stats.status_code == 200
    stats = res_stats.json()["cache_metrics"]
    assert stats["hits"] >= 1

    # 3. Cache clear
    res_clear = client.post("/api/v1/cache/clear")
    assert res_clear.status_code == 200
    cleared_stats = res_clear.json()["cache_metrics"]
    assert cleared_stats["tier1_memory_items"] == 0


def test_data_drift_detector():
    """Test Population Stability Index (PSI), KS-test, and Wasserstein distance."""
    import numpy as np

    from src.analytics.drift_detection import (
        DataDriftDetector,
        calculate_ks_2samp,
        calculate_psi,
        calculate_wasserstein,
    )

    # 1. Identical distributions -> PSI ~ 0, KS pval ~ 1
    ref = np.random.normal(100, 15, 200)
    tgt_same = ref.copy()
    psi_same = calculate_psi(ref, tgt_same)
    assert psi_same < 0.05
    ks_stat, ks_pval = calculate_ks_2samp(ref, tgt_same)
    assert ks_stat < 0.1
    assert ks_pval > 0.9

    # 2. Shifted distributions -> PSI > 0.25
    tgt_shifted = np.random.normal(150, 25, 200)
    psi_shifted = calculate_psi(ref, tgt_shifted)
    assert psi_shifted > 0.25
    w_dist = calculate_wasserstein(ref, tgt_shifted)
    assert w_dist > 30.0

    # 3. Full Detector pipeline
    detector = DataDriftDetector()
    report = detector.evaluate_drift()
    assert "system_psi" in report
    assert "features" in report
    assert len(report["features"]) > 0


def test_data_drift_api_endpoints():
    """Test /api/v1/ml/drift/report and /api/v1/ml/drift/evaluate."""
    # 1. Report endpoint
    res = client.get("/api/v1/ml/drift/report")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "drift_report" in data
    assert "system_psi" in data["drift_report"]

    # 2. Custom evaluate endpoint
    payload = {
        "target_data": [
            {"total_sanciones_impuestas": 50.0, "total_infracciones_imputadas": 200.0, "total_solicitudes_aml": 10.0},
            {"total_sanciones_impuestas": 60.0, "total_infracciones_imputadas": 250.0, "total_solicitudes_aml": 12.0},
            {"total_sanciones_impuestas": 45.0, "total_infracciones_imputadas": 180.0, "total_solicitudes_aml": 9.0},
            {"total_sanciones_impuestas": 55.0, "total_infracciones_imputadas": 210.0, "total_solicitudes_aml": 11.0},
        ]
    }
    res_eval = client.post("/api/v1/ml/drift/evaluate", json=payload)
    assert res_eval.status_code == 200
    data_eval = res_eval.json()
    assert data_eval["status"] == "success"
    assert "drift_evaluation" in data_eval


def test_monte_carlo_stress_testing():
    """Test N=10,000 Monte Carlo correlated stochastic stress tester."""
    from src.analytics.stress_testing import MonteCarloStressTester

    tester = MonteCarloStressTester()
    res = tester.run_simulation(n_simulations=1000, horizon_months=12, scenario="combined_macro_stress")

    assert "metrics" in res
    m = res["metrics"]
    assert m["var_95_dop"] <= m["var_99_dop"]
    assert m["var_95_dop"] <= m["cvar_95_expected_shortfall_dop"]
    assert m["stress_liquidity_buffer_required_95_dop"] >= 0
    assert len(res["distribution_bins"]["bin_centers_millions_dop"]) > 0


def test_auth_jwt_and_rbac():
    """Test PBKDF2 password hashing, JWT token lifecycle, and role verification."""
    from src.api.auth import (
        authenticate_user,
        create_access_token,
        decode_access_token,
        hash_password,
        verify_password,
    )

    # 1. Hashing
    h = hash_password("secret_pass_2026")
    assert verify_password("secret_pass_2026", h)
    assert not verify_password("wrong_pass", h)

    # 2. Authentication
    user = authenticate_user("inspector", "inspector123")
    assert user is not None
    assert user.role == "AUDITOR_INSPECTOR"

    invalid_user = authenticate_user("inspector", "wrong_password")
    assert invalid_user is None

    # 3. JWT Lifecycle
    token = create_access_token({"sub": "inspector", "role": "AUDITOR_INSPECTOR"})
    payload = decode_access_token(token)
    assert payload["sub"] == "inspector"
    assert payload["role"] == "AUDITOR_INSPECTOR"

    # 4. Auth API Login endpoint
    login_res = client.post("/api/v1/auth/token", data={"username": "datascientist", "password": "mlops123"})
    assert login_res.status_code == 200
    token_data = login_res.json()
    assert "access_token" in token_data
    assert token_data["role"] == "DATA_SCIENTIST"

    # 5. /api/v1/auth/me endpoint
    headers = {"Authorization": f"Bearer {token_data['access_token']}"}
    me_res = client.get("/api/v1/auth/me", headers=headers)
    assert me_res.status_code == 200
    assert me_res.json()["user"]["username"] == "datascientist"


def test_monte_carlo_api_endpoint():
    """Test /api/v1/ml/stress-test/simulate endpoint."""
    payload = {
        "n_simulations": 1000,
        "horizon_months": 12,
        "scenario": "combined_macro_stress",
        "sanctions_shock_pct": 0.30,
        "aml_shock_pct": 0.50,
        "claims_shock_pct": 0.25,
    }
    res = client.post("/api/v1/ml/stress-test/simulate", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "success"
    assert "metrics" in data
    assert data["metrics"]["var_95_dop"] > 0





