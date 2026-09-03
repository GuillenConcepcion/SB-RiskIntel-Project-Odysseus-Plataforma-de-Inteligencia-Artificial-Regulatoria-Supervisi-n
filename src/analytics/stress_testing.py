"""Odysseus Monte Carlo Stress Testing & Correlated Stochastic Simulation Engine.

Simulates multivariate regulatory, macroeconomic, and claims shocks using Cholesky decomposition
over empirical covariance matrices (N=10,000 iterations) to calculate VaR, CVaR (Expected Shortfall),
and capital/liquidity buffer requirements for banking supervision.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


class MonteCarloStressTester:
    """Multivariate stochastic simulator for systemic banking shocks and ProUsuario claims."""

    def __init__(self, models_dir: Path = MODELS_DIR):
        self.models_dir = models_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def run_simulation(
        self,
        n_simulations: int = 10000,
        horizon_months: int = 12,
        scenario: str = "combined_macro_stress",
        sanctions_shock_pct: float = 0.30,
        aml_shock_pct: float = 0.50,
        claims_shock_pct: float = 0.25,
        random_seed: int = 42,
    ) -> Dict[str, Any]:
        """Execute N=10,000 correlated Monte Carlo iterations under regulatory stress shocks."""
        np.random.seed(random_seed)

        # Load historical claims and supervision data
        claims_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"
        if not claims_path.exists():
            raise FileNotFoundError(f"Claims dataset not found at {claims_path}")
        df_claims = pd.read_parquet(claims_path)

        # Core stochastic variables
        var_cols = ["reclamaciones", "monto_instruido_devolver", "reclamaciones_roll_mean_3m"]
        available_cols = [c for c in var_cols if c in df_claims.columns]

        data_matrix = df_claims[available_cols].dropna().values
        if len(data_matrix) < 5:
            raise ValueError("Insufficient historical data points for covariance estimation.")

        # Compute empirical mean and covariance matrix
        means = np.mean(data_matrix, axis=0)
        cov_matrix = np.cov(data_matrix, rowvar=False)

        # Ensure positive semi-definiteness for Cholesky decomposition
        cov_matrix += np.eye(len(available_cols)) * 1e-6
        cholesky_l = np.linalg.cholesky(cov_matrix)

        # Apply stress scenario multipliers to expected means
        stressed_means = means.copy()
        scenario_description = "Línea Base Histórica Ordinaria"

        if scenario == "conduct_shock":
            stressed_means[0] *= (1.0 + claims_shock_pct)
            stressed_means[1] *= (1.0 + claims_shock_pct * 1.5)
            scenario_description = f"Choque de Conducta (+{claims_shock_pct*100:.0f}% Reclamos, +{claims_shock_pct*150:.0f}% Montos)"
        elif scenario == "aml_surge":
            stressed_means[0] *= (1.0 + aml_shock_pct * 0.5)
            stressed_means[1] *= (1.0 + aml_shock_pct)
            scenario_description = f"Presión Antilavado / AML (+{aml_shock_pct*100:.0f}% Solicitudes Judiciales)"
        elif scenario == "combined_macro_stress":
            multiplier = 1.0 + (sanctions_shock_pct + aml_shock_pct + claims_shock_pct) / 2.0
            stressed_means *= multiplier
            scenario_description = f"Choque Macroeconómico Combinado (+{sanctions_shock_pct*100:.0f}% Sanciones, +{aml_shock_pct*100:.0f}% AML, +{claims_shock_pct*100:.0f}% Reclamos)"

        logger.info(f"Generating N={n_simulations} Correlated Monte Carlo Iterations under scenario: {scenario}...")

        # Generate N independent standard normal vectors: Z ~ N(0, I)
        z_random = np.random.normal(0, 1, size=(n_simulations, len(available_cols)))

        # Correlated simulated shocks: X_sim = Stressed_Means + L * Z
        correlated_shocks = stressed_means + np.dot(z_random, cholesky_l.T)
        correlated_shocks = np.maximum(0, correlated_shocks)  # claims and amounts non-negative

        sim_claims_monthly = correlated_shocks[:, 0]
        sim_montos_monthly = correlated_shocks[:, 1]

        # Aggregate over planning horizon
        sim_claims_horizon = sim_claims_monthly * horizon_months
        sim_montos_horizon = sim_montos_monthly * horizon_months

        # Calculate Risk Metrics: VaR and CVaR (Expected Shortfall)
        var_95_montos = float(np.percentile(sim_montos_horizon, 95))
        var_99_montos = float(np.percentile(sim_montos_horizon, 99))
        cvar_95_montos = float(np.mean(sim_montos_horizon[sim_montos_horizon >= var_95_montos]))
        cvar_99_montos = float(np.mean(sim_montos_horizon[sim_montos_horizon >= var_99_montos]))

        var_95_claims = float(np.percentile(sim_claims_horizon, 95))
        var_99_claims = float(np.percentile(sim_claims_horizon, 99))
        cvar_95_claims = float(np.mean(sim_claims_horizon[sim_claims_horizon >= var_95_claims]))

        mean_expected_montos = float(np.mean(sim_montos_horizon))
        stress_liquidity_buffer_95 = float(max(0, var_95_montos - mean_expected_montos))
        stress_liquidity_buffer_99 = float(max(0, var_99_montos - mean_expected_montos))

        # Probability of Extreme Outlier Event (Claims > 1.5x Historical Max)
        hist_max_claims = float(np.max(df_claims["reclamaciones"])) * horizon_months
        prob_extreme_event = float(np.mean(sim_claims_horizon > hist_max_claims) * 100)

        # Generate distribution histogram data for visual rendering
        hist_counts, bin_edges = np.histogram(sim_montos_horizon / 1e6, bins=40)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

        simulation_results = {
            "task": "monte_carlo_multivariate_stress_testing",
            "scenario": scenario,
            "scenario_description": scenario_description,
            "n_simulations": n_simulations,
            "horizon_months": horizon_months,
            "metrics": {
                "mean_expected_restitution_dop": round(mean_expected_montos, 2),
                "var_95_dop": round(var_95_montos, 2),
                "var_99_dop": round(var_99_montos, 2),
                "cvar_95_expected_shortfall_dop": round(cvar_95_montos, 2),
                "cvar_99_expected_shortfall_dop": round(cvar_99_montos, 2),
                "stress_liquidity_buffer_required_95_dop": round(stress_liquidity_buffer_95, 2),
                "stress_liquidity_buffer_required_99_dop": round(stress_liquidity_buffer_99, 2),
                "mean_expected_claims": round(float(np.mean(sim_claims_horizon)), 0),
                "var_95_claims": round(var_95_claims, 0),
                "var_99_claims": round(var_99_claims, 0),
                "cvar_95_claims": round(cvar_95_claims, 0),
                "prob_extreme_event_pct": round(prob_extreme_event, 2),
            },
            "distribution_bins": {
                "bin_centers_millions_dop": [round(float(b), 2) for b in bin_centers],
                "frequencies": [int(c) for c in hist_counts],
            },
        }

        # Save summary artifact
        with open(self.models_dir / "monte_carlo_stress_summary.json", "w", encoding="utf-8") as f:
            json.dump(simulation_results, f, indent=2)

        logger.info(f"Monte Carlo Stress Testing Complete | VaR 95%: DOP ${var_95_montos:,.2f} | CVaR 95%: DOP ${cvar_95_montos:,.2f}")
        return simulation_results


def run_monte_carlo_cli():
    """Execute Monte Carlo simulator from CLI."""
    tester = MonteCarloStressTester()
    res = tester.run_simulation(n_simulations=10000, horizon_months=12)
    m = res["metrics"]
    print(f">>> [Monte Carlo Simulation] Scenario: {res['scenario_description']}", flush=True)
    print(f"  * Expected Restitution (12M): DOP ${m['mean_expected_restitution_dop']:,.2f}", flush=True)
    print(f"  * VaR 95% (DOP): DOP ${m['var_95_dop']:,.2f}", flush=True)
    print(f"  * CVaR 95% (Expected Shortfall): DOP ${m['cvar_95_expected_shortfall_dop']:,.2f}", flush=True)
    print(f"  * VaR 99% (DOP): DOP ${m['var_99_dop']:,.2f}", flush=True)
    print(f"  * Required Stress Liquidity Buffer (95%): DOP ${m['stress_liquidity_buffer_required_95_dop']:,.2f}", flush=True)


if __name__ == "__main__":
    run_monte_carlo_cli()
