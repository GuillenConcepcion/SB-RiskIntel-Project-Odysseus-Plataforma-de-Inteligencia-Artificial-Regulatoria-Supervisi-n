"""Prescriptive Analytics & Decision Science Engine for Banking Supervision."""

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class RegulatoryActionPrescription:
    """Actionable prescription for a supervisory period or entity."""

    risk_tier: str
    ews_probability: float
    recommended_action: str
    regulatory_mandate: str
    audit_frequency_days: int
    required_reserve_ratio: float
    immediate_escalation: bool


def prescribe_early_warning_regulatory_actions(
    ews_probability: float,
    conduct_risk_index: float = 0.0,
    momentum_ratio: float = 1.0,
) -> RegulatoryActionPrescription:
    """Prescribe tailored supervisory action based on quantitative risk thresholds."""
    # Composite risk assessment
    combined_score = 0.65 * ews_probability + 0.20 * min(conduct_risk_index / 100.0, 1.0) + 0.15 * min(momentum_ratio / 2.0, 1.0)

    if combined_score < 0.35:
        return RegulatoryActionPrescription(
            risk_tier="🟢 Tier 1: Riesgo Bajo / Controlado",
            ews_probability=round(ews_probability, 4),
            recommended_action="Monitoreo Ordinario Off-Site y reporte trimestral estándar.",
            regulatory_mandate="Circular SB-2023-01 / Supervisión Continua",
            audit_frequency_days=180,
            required_reserve_ratio=1.0,
            immediate_escalation=False,
        )
    elif combined_score < 0.55:
        return RegulatoryActionPrescription(
            risk_tier="🟡 Tier 2: Riesgo Moderado / Vigilancia Preventiva",
            ews_probability=round(ews_probability, 4),
            recommended_action="Requerimiento de información ampliada sobre canales y revisión de tiempos de respuesta ProUsuario.",
            regulatory_mandate="Art. 56 Ley Monetaria y Financiera (LMYF)",
            audit_frequency_days=90,
            required_reserve_ratio=1.15,
            immediate_escalation=False,
        )
    elif combined_score < 0.75:
        return RegulatoryActionPrescription(
            risk_tier="🟠 Tier 3: Riesgo Alto / Alerta Temprana Activa",
            ews_probability=round(ews_probability, 4),
            recommended_action="Despliegue de Auditoría In-Situ focalizada y exigencia de Plan de Regularización a 30 días.",
            regulatory_mandate="Art. 60 LMYF / Plan de Regularización Obligatorio",
            audit_frequency_days=30,
            required_reserve_ratio=1.35,
            immediate_escalation=True,
        )
    else:
        return RegulatoryActionPrescription(
            risk_tier="🔴 Tier 4: Riesgo Crítico / Intervención Inmediata",
            ews_probability=round(ews_probability, 4),
            recommended_action="Apertura formal de Proceso Administrativo Sancionador, imposición de medidas cautelares y auditoría diaria.",
            regulatory_mandate="Art. 67 LMYF / Procedimiento Sancionador SB",
            audit_frequency_days=7,
            required_reserve_ratio=1.60,
            immediate_escalation=True,
        )


def optimize_supervisory_inspection_allocation(
    entities_risk_df: pd.DataFrame,
    total_auditor_hours: float = 2400.0,
    min_hours_per_entity: float = 40.0,
    max_hours_per_entity: float = 400.0,
) -> pd.DataFrame:
    """Optimize supervisory inspection hours allocation across financial entities based on risk weight."""
    df = entities_risk_df.copy()
    if "risk_score" not in df.columns:
        raise ValueError("DataFrame must contain a 'risk_score' column [0-100].")

    # Risk-weighted allocation
    df["normalized_risk"] = df["risk_score"] / (df["risk_score"].sum() + 1e-9)

    # Initial proportional allocation
    df["allocated_hours_raw"] = df["normalized_risk"] * total_auditor_hours

    # Apply boundary constraints
    df["allocated_hours"] = df["allocated_hours_raw"].clip(
        lower=min_hours_per_entity, upper=max_hours_per_entity
    )

    # Re-normalize to match total available budget exactly
    scale_factor = total_auditor_hours / df["allocated_hours"].sum()
    df["allocated_hours_optimal"] = (df["allocated_hours"] * scale_factor).round(1)

    # Suggested inspection team size (assuming 40h per auditor per week)
    df["auditores_estimados"] = np.ceil(df["allocated_hours_optimal"] / 80.0).astype(int)
    df["prioridad_inspeccion"] = pd.qcut(
        df["risk_score"], q=min(4, len(df)), labels=["Baja", "Media", "Alta", "Crítica"][:min(4, len(df))]
    )

    return df.sort_values("risk_score", ascending=False).reset_index(drop=True)


def calculate_restitution_liquidity_buffer(
    historical_restitution_series: pd.Series,
    forecast_next_period: float,
    confidence_levels: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Calculate regulatory liquidity reserve buffer for consumer restitutions using Parametric & Historical VaR/CVaR."""
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    clean_series = historical_restitution_series.dropna()
    arr = clean_series.values
    n = len(arr)

    if n < 5:
        return {
            "Forecast_Base_DOP": float(forecast_next_period),
            "VaR_95_DOP": float(forecast_next_period * 1.25),
            "CVaR_95_DOP": float(forecast_next_period * 1.40),
            "Recommended_Liquidity_Buffer_DOP": float(forecast_next_period * 1.30),
        }

    mean_val = np.mean(arr)
    std_val = np.std(arr, ddof=1)

    # Parametric VaR (Normal distribution)
    z_95 = stats.norm.ppf(0.95)
    z_99 = stats.norm.ppf(0.99)
    param_var_95 = mean_val + z_95 * std_val
    param_var_99 = mean_val + z_99 * std_val

    # Historical VaR & Expected Shortfall (CVaR)
    hist_var_95 = np.percentile(arr, 95)
    hist_var_99 = np.percentile(arr, 99)

    tail_95 = arr[arr >= hist_var_95]
    cvar_95 = np.mean(tail_95) if len(tail_95) > 0 else hist_var_95

    tail_99 = arr[arr >= hist_var_99]
    cvar_99 = np.mean(tail_99) if len(tail_99) > 0 else hist_var_99

    # Prescriptive buffer recommendation: Max(Forecast * 1.15, CVaR_95)
    recommended_buffer = max(forecast_next_period * 1.20, float(cvar_95))

    return {
        "Forecast_Base_DOP": round(float(forecast_next_period), 2),
        "Parametric_VaR_95_DOP": round(float(param_var_95), 2),
        "Parametric_VaR_99_DOP": round(float(param_var_99), 2),
        "Historical_VaR_95_DOP": round(float(hist_var_95), 2),
        "Historical_VaR_99_DOP": round(float(hist_var_99), 2),
        "Expected_Shortfall_CVaR_95_DOP": round(float(cvar_95), 2),
        "Expected_Shortfall_CVaR_99_DOP": round(float(cvar_99), 2),
        "Recommended_Liquidity_Buffer_DOP": round(float(recommended_buffer), 2),
        "Safety_Margin_Pct": round(((recommended_buffer / forecast_next_period) - 1.0) * 100.0, 2),
    }
