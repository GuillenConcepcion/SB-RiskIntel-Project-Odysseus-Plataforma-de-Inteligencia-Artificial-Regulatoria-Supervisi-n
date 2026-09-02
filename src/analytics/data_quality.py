"""Data Quality & Integrity Assessment Engine for SupTech datasets."""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ColumnQualityMetric:
    """Quality metrics for a single column."""

    column_name: str
    dtype: str
    total_count: int
    null_count: int
    null_percentage: float
    unique_count: int
    unique_ratio: float
    zero_count: int
    negative_count: int
    outliers_iqr_count: int
    outliers_zscore_count: int
    min_value: Optional[Any] = None
    max_value: Optional[Any] = None
    mean_value: Optional[float] = None
    std_value: Optional[float] = None


@dataclass
class DatasetQualityReport:
    """Comprehensive Data Quality report for a dataset."""

    dataset_name: str
    total_rows: int
    total_columns: int
    duplicate_rows_count: int
    duplicate_rows_percentage: float
    overall_completeness_score: float  # [0, 100]
    overall_quality_score: float  # [0, 100]
    column_metrics: List[ColumnQualityMetric] = field(default_factory=list)
    quality_alerts: List[str] = field(default_factory=list)

    def to_dataframe(self) -> pd.DataFrame:
        """Convert column metrics to pandas DataFrame."""
        records = []
        for m in self.column_metrics:
            records.append({
                "Columna": m.column_name,
                "Tipo": m.dtype,
                "Total Filas": m.total_count,
                "Nulos": m.null_count,
                "% Nulos": round(m.null_percentage, 2),
                "Únicos": m.unique_count,
                "Ceros": m.zero_count,
                "Negativos": m.negative_count,
                "Outliers (IQR)": m.outliers_iqr_count,
                "Outliers (Z>3)": m.outliers_zscore_count,
                "Mínimo": m.min_value,
                "Máximo": m.max_value,
                "Media": round(m.mean_value, 2) if m.mean_value is not None else None,
            })
        return pd.DataFrame(records)


class DataQualityAuditor:
    """Auditor class to perform thorough data quality evaluations."""

    def __init__(self, z_threshold: float = 3.0, iqr_multiplier: float = 1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier

    def audit(self, df: pd.DataFrame, dataset_name: str = "Dataset") -> DatasetQualityReport:
        """Perform comprehensive data quality audit on a pandas DataFrame."""
        total_rows, total_cols = df.shape
        if total_rows == 0:
            return DatasetQualityReport(
                dataset_name=dataset_name,
                total_rows=0,
                total_columns=total_cols,
                duplicate_rows_count=0,
                duplicate_rows_percentage=0.0,
                overall_completeness_score=0.0,
                overall_quality_score=0.0,
                quality_alerts=["El dataset está vacío."],
            )

        duplicates = int(df.duplicated().sum())
        duplicate_pct = (duplicates / total_rows) * 100.0
        alerts: List[str] = []

        if duplicate_pct > 0.0:
            alerts.append(f"Se detectaron {duplicates} filas duplicadas ({duplicate_pct:.2f}%).")

        total_cells = total_rows * total_cols
        total_nulls = int(df.isnull().sum().sum())
        completeness = ((total_cells - total_nulls) / total_cells) * 100.0

        column_metrics: List[ColumnQualityMetric] = []
        outlier_penalties = 0

        for col in df.columns:
            s = df[col]
            null_c = int(s.isnull().sum())
            null_p = (null_c / total_rows) * 100.0
            unique_c = int(s.nunique(dropna=False))
            unique_r = unique_c / total_rows

            zero_c = 0
            neg_c = 0
            iqr_outliers = 0
            z_outliers = 0
            min_val, max_val, mean_val, std_val = None, None, None, None

            if pd.api.types.is_numeric_dtype(s):
                valid_num = s.dropna()
                zero_c = int((valid_num == 0).sum())
                neg_c = int((valid_num < 0).sum())
                if len(valid_num) > 0:
                    min_val = float(valid_num.min())
                    max_val = float(valid_num.max())
                    mean_val = float(valid_num.mean())
                    std_val = float(valid_num.std())

                    # IQR Outliers
                    q25 = np.percentile(valid_num, 25)
                    q75 = np.percentile(valid_num, 75)
                    iqr = q75 - q25
                    lower_bound = q25 - self.iqr_multiplier * iqr
                    upper_bound = q75 + self.iqr_multiplier * iqr
                    iqr_outliers = int(((valid_num < lower_bound) | (valid_num > upper_bound)).sum())

                    # Z-score Outliers
                    if std_val > 1e-9 and len(valid_num) > 3:
                        z_scores = np.abs(stats.zscore(valid_num))
                        z_outliers = int((z_scores > self.z_threshold).sum())

                if neg_c > 0:
                    alerts.append(f"Columna numérica '{col}' contiene {neg_c} valores negativos.")
            else:
                valid_cat = s.dropna()
                if len(valid_cat) > 0:
                    min_val = str(valid_cat.min())[:30]
                    max_val = str(valid_cat.max())[:30]

            outlier_penalties += min(z_outliers, 10)

            column_metrics.append(
                ColumnQualityMetric(
                    column_name=str(col),
                    dtype=str(s.dtype),
                    total_count=total_rows,
                    null_count=null_c,
                    null_percentage=null_p,
                    unique_count=unique_c,
                    unique_ratio=unique_r,
                    zero_count=zero_c,
                    negative_count=neg_c,
                    outliers_iqr_count=iqr_outliers,
                    outliers_zscore_count=z_outliers,
                    min_value=min_val,
                    max_value=max_val,
                    mean_value=mean_val,
                    std_value=std_val,
                )
            )

        # Calculate Quality Score [0-100]
        dup_score = max(0.0, 100.0 - duplicate_pct * 5.0)
        anomaly_score = max(0.0, 100.0 - (outlier_penalties / max(1, total_cols)) * 5.0)
        quality_score = round(0.5 * completeness + 0.25 * dup_score + 0.25 * anomaly_score, 2)

        return DatasetQualityReport(
            dataset_name=dataset_name,
            total_rows=total_rows,
            total_columns=total_cols,
            duplicate_rows_count=duplicates,
            duplicate_rows_percentage=round(duplicate_pct, 2),
            overall_completeness_score=round(completeness, 2),
            overall_quality_score=quality_score,
            column_metrics=column_metrics,
            quality_alerts=alerts,
        )


def audit_dataframe_quality(df: pd.DataFrame, dataset_name: str = "Dataset") -> DatasetQualityReport:
    """Convenience function to audit a dataframe quality."""
    auditor = DataQualityAuditor()
    return auditor.audit(df, dataset_name=dataset_name)


def generate_overall_data_quality_report(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate consolidated scorecard across multiple datasets."""
    summary_rows = []
    auditor = DataQualityAuditor()

    for name, df in datasets.items():
        report = auditor.audit(df, dataset_name=name)
        summary_rows.append({
            "Dataset": name,
            "Filas": report.total_rows,
            "Columnas": report.total_columns,
            "Duplicados (%)": round(report.duplicate_rows_percentage, 1),
            "Completitud (%)": round(report.overall_completeness_score, 1),
            "Data Quality Score": round(report.overall_quality_score, 1),
            "Alertas": len(report.quality_alerts),
            "Estado": "✅ Óptimo" if report.overall_quality_score >= 85 else ("⚠️ Aceptable" if report.overall_quality_score >= 70 else "❌ Crítico"),
        })

    return pd.DataFrame(summary_rows)
