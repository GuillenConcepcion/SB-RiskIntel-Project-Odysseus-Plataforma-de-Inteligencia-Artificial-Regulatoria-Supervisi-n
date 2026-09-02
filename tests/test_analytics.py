"""Unit tests for Analytics: Data Quality, Descriptive & Prescriptive Engines."""

import numpy as np
import pandas as pd
import pytest

from src.analytics.data_quality import (
    DataQualityAuditor,
    generate_overall_data_quality_report,
)
from src.analytics.descriptive import (
    calculate_concentration_indices,
    compute_advanced_descriptive_stats,
    compute_correlation_matrix,
    decompose_supervisory_time_series,
)
from src.analytics.prescriptive import (
    calculate_restitution_liquidity_buffer,
    optimize_supervisory_inspection_allocation,
    prescribe_early_warning_regulatory_actions,
)


@pytest.fixture
def sample_supervision_df():
    """Create sample supervision metrics dataframe."""
    np.random.seed(42)
    return pd.DataFrame({
        "infracciones": np.random.poisson(lam=15, size=24),
        "sanciones": np.random.poisson(lam=5, size=24),
        "monto_sanciones": np.random.exponential(scale=1000000, size=24),
        "solicitudes_aml": np.random.poisson(lam=8, size=24),
    })


def test_data_quality_auditor(sample_supervision_df):
    """Test DataQualityAuditor and scorecard."""
    auditor = DataQualityAuditor()
    report = auditor.audit(sample_supervision_df, dataset_name="TestSupervision")

    assert report.total_rows == 24
    assert report.total_columns == 4
    assert report.overall_completeness_score == 100.0
    assert 0.0 <= report.overall_quality_score <= 100.0

    df_metrics = report.to_dataframe()
    assert len(df_metrics) == 4
    assert "Columna" in df_metrics.columns
    assert "Outliers (IQR)" in df_metrics.columns

    # Test multi-dataset summary
    scorecard = generate_overall_data_quality_report({"test_df": sample_supervision_df})
    assert len(scorecard) == 1
    assert "Data Quality Score" in scorecard.columns


def test_descriptive_statistics(sample_supervision_df):
    """Test advanced descriptive statistics computation."""
    stats_df = compute_advanced_descriptive_stats(sample_supervision_df)
    assert len(stats_df) == 4
    assert "Media" in stats_df.columns
    assert "Asimetría (Skew)" in stats_df.columns
    assert "Normality (JB p-val)" in stats_df.columns

    corr_df = compute_correlation_matrix(sample_supervision_df)
    assert corr_df.shape == (4, 4)
    assert np.allclose(np.diag(corr_df), 1.0)


def test_time_series_decomposition():
    """Test seasonal decomposition on synthetic series."""
    t = np.arange(36)
    seasonal = 10 * np.sin(2 * np.pi * t / 12)
    trend = 0.5 * t
    noise = np.random.normal(0, 1, 36)
    series = pd.Series(50 + trend + seasonal + noise)

    decomp = decompose_supervisory_time_series(series, period=12)
    assert "observed" in decomp
    assert "trend" in decomp
    assert "seasonal" in decomp
    assert "residual" in decomp
    assert len(decomp["trend"].dropna()) > 0


def test_concentration_indices():
    """Test HHI, Gini and CR ratios."""
    shares = np.array([40, 30, 20, 10])
    indices = calculate_concentration_indices(shares)

    assert indices["HHI"] == 3000.0
    assert 0.0 <= indices["Gini"] <= 1.0
    assert indices["CR4 (%)"] == 100.0
    assert "Alta Concentración" in indices["Concentration_Level"]


def test_prescriptive_early_warning():
    """Test prescriptive decision tree output."""
    low_presc = prescribe_early_warning_regulatory_actions(0.10, conduct_risk_index=5.0, momentum_ratio=0.8)
    assert "Tier 1" in low_presc.risk_tier
    assert not low_presc.immediate_escalation

    high_presc = prescribe_early_warning_regulatory_actions(0.85, conduct_risk_index=80.0, momentum_ratio=2.2)
    assert "Tier 4" in high_presc.risk_tier
    assert high_presc.immediate_escalation


def test_prescriptive_inspection_optimization():
    """Test inspection allocation optimization across entities."""
    entities_df = pd.DataFrame({
        "entidad": ["Banco A", "Banco B", "Banco C", "Banco D"],
        "risk_score": [85.0, 60.0, 35.0, 15.0],
    })

    optimized = optimize_supervisory_inspection_allocation(entities_df, total_auditor_hours=1000.0)
    assert len(optimized) == 4
    assert np.isclose(optimized["allocated_hours_optimal"].sum(), 1000.0, atol=1.0)
    assert optimized.loc[0, "entidad"] == "Banco A"
    assert optimized.loc[0, "allocated_hours_optimal"] > optimized.loc[3, "allocated_hours_optimal"]


def test_restitution_liquidity_buffer():
    """Test prescriptive restitution liquidity buffer with VaR/CVaR."""
    history = pd.Series([100000, 120000, 150000, 90000, 200000, 130000, 110000, 180000, 220000, 160000])
    forecast = 190000.0

    buffer_res = calculate_restitution_liquidity_buffer(history, forecast)
    assert buffer_res["Forecast_Base_DOP"] == forecast
    assert buffer_res["Recommended_Liquidity_Buffer_DOP"] >= forecast
    assert buffer_res["Parametric_VaR_95_DOP"] > 0
    assert buffer_res["Expected_Shortfall_CVaR_95_DOP"] > 0
