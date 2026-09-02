"""Descriptive Statistics & Exploratory Analytics Engine for SupTech Risk Intelligence."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.seasonal import seasonal_decompose


def compute_advanced_descriptive_stats(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    round_digits: int = 3,
) -> pd.DataFrame:
    """Compute rich parametric and non-parametric descriptive statistics."""
    if columns is None:
        target_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        target_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    results = []
    for col in target_cols:
        s = df[col].dropna()
        n = len(s)
        if n == 0:
            continue

        mean_val = s.mean()
        std_val = s.std(ddof=1) if n > 1 else 0.0
        median_val = s.median()
        q25 = np.percentile(s, 25)
        q75 = np.percentile(s, 75)
        iqr_val = q75 - q25
        p05 = np.percentile(s, 5)
        p95 = np.percentile(s, 95)
        min_val = s.min()
        max_val = s.max()

        # Skewness & Kurtosis
        skew_val = float(stats.skew(s, bias=False)) if n > 2 else 0.0
        kurt_val = float(stats.kurtosis(s, bias=False)) if n > 3 else 0.0

        # Normality test (Jarque-Bera)
        if n >= 8:
            _, jb_pvalue = stats.jarque_bera(s)
        else:
            jb_pvalue = np.nan

        # Coefficient of variation (CV)
        cv = (std_val / mean_val) if abs(mean_val) > 1e-9 else np.nan

        results.append({
            "Variable": col,
            "N": n,
            "Media": round(mean_val, round_digits),
            "Desv. Est.": round(std_val, round_digits),
            "Coef. Var.": round(cv, round_digits) if not np.isnan(cv) else None,
            "Mediana (Q2)": round(median_val, round_digits),
            "IQR": round(iqr_val, round_digits),
            "Mín": round(min_val, round_digits),
            "P05": round(p05, round_digits),
            "P25 (Q1)": round(q25, round_digits),
            "P75 (Q3)": round(q75, round_digits),
            "P95": round(p95, round_digits),
            "Máx": round(max_val, round_digits),
            "Asimetría (Skew)": round(skew_val, round_digits),
            "Curtosis (Kurt)": round(kurt_val, round_digits),
            "Normality (JB p-val)": round(jb_pvalue, 4) if not np.isnan(jb_pvalue) else np.nan,
            "Distribución": "Gaussiana" if (not np.isnan(jb_pvalue) and jb_pvalue > 0.05) else "No-Paramétrica",
        })

    return pd.DataFrame(results)


def compute_correlation_matrix(
    df: pd.DataFrame,
    columns: Optional[List[str]] = None,
    method: str = "pearson",
) -> pd.DataFrame:
    """Compute correlation matrix with specified method ('pearson' or 'spearman')."""
    if columns is None:
        target_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    else:
        target_cols = [c for c in columns if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]

    clean_df = df[target_cols].dropna()
    return clean_df.corr(method=method).round(3)


def decompose_supervisory_time_series(
    series: pd.Series,
    period: int = 12,
    model: str = "additive",
) -> Dict[str, pd.Series]:
    """Decompose time series into trend, seasonal, and residual components."""
    clean_series = series.dropna()
    if len(clean_series) < 2 * period:
        adj_period = max(2, min(period, len(clean_series) // 2))
    else:
        adj_period = period

    decomp = seasonal_decompose(clean_series, model=model, period=adj_period, extrapolate_trend="period")
    return {
        "observed": clean_series,
        "trend": decomp.trend,
        "seasonal": decomp.seasonal,
        "residual": decomp.resid,
    }


def calculate_concentration_indices(
    series: Union[pd.Series, np.ndarray],
    top_n: Tuple[int, int] = (4, 8),
) -> Dict[str, float]:
    """Calculate market / risk concentration indicators: HHI, Gini Coefficient, CR4, CR8."""
    arr = np.array(series, dtype=float)
    arr = arr[~np.isnan(arr)]
    arr = arr[arr > 0]

    if len(arr) == 0:
        return {
            "HHI": 0.0,
            "Gini": 0.0,
            "CR4": 0.0,
            "CR8": 0.0,
            "Concentration_Level": "Sin datos",
        }

    total_sum = np.sum(arr)
    shares = (arr / total_sum) * 100.0

    # Herfindahl-Hirschman Index (HHI) [0, 10000]
    hhi = float(np.sum(shares ** 2))

    # Gini Coefficient
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    index = np.arange(1, n + 1)
    gini = float((2 * np.sum(index * sorted_arr)) / (n * np.sum(sorted_arr)) - (n + 1) / n)
    gini = max(0.0, min(1.0, gini))

    # Concentration Ratios (CR_top)
    sorted_shares = np.sort(shares)[::-1]
    cr4 = float(np.sum(sorted_shares[: top_n[0]])) if len(sorted_shares) >= top_n[0] else float(np.sum(sorted_shares))
    cr8 = float(np.sum(sorted_shares[: top_n[1]])) if len(sorted_shares) >= top_n[1] else float(np.sum(sorted_shares))

    # Classification by Department of Justice / Regulatory Standards
    if hhi < 1500:
        conc_level = "Baja Concentración (Diversificado)"
    elif hhi <= 2500:
        conc_level = "Concentración Moderada"
    else:
        conc_level = "Alta Concentración (Riesgo Sistémico Focalizado)"

    return {
        "HHI": round(hhi, 2),
        "Gini": round(gini, 3),
        f"CR{top_n[0]} (%)": round(cr4, 2),
        f"CR{top_n[1]} (%)": round(cr8, 2),
        "Concentration_Level": conc_level,
    }
