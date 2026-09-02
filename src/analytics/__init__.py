"""Analytics module for SB-RiskIntel: Descriptive, Prescriptive & Data Quality engines."""

from src.analytics.data_quality import (
    DataQualityAuditor,
    audit_dataframe_quality,
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

__all__ = [
    "DataQualityAuditor",
    "audit_dataframe_quality",
    "generate_overall_data_quality_report",
    "compute_advanced_descriptive_stats",
    "compute_correlation_matrix",
    "decompose_supervisory_time_series",
    "calculate_concentration_indices",
    "optimize_supervisory_inspection_allocation",
    "prescribe_early_warning_regulatory_actions",
    "calculate_restitution_liquidity_buffer",
]
