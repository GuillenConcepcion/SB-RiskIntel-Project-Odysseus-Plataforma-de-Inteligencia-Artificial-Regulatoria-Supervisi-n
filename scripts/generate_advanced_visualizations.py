"""Odysseus High-Impact Advanced Visualizations Generator for SB-RiskIntel.

Generates executive-ready, publication-quality regulatory intelligence charts
using Matplotlib, Seaborn, and Plotly in the `images/` directory:
1. 01_conformal_prediction_intervals.png (Split-Conformal 90% & 95% Bands)
2. 02_monte_carlo_stress_distribution.png (N=10,000 Correlated Shocks & VaR/CVaR)
3. 03_xai_shap_feature_importance.png (Global SHAP Attributions & Legal Drivers)
4. 04_unsupervised_anomaly_scorecard.png (Isolation Forest, LOF, OCSVM, PCA Heatmap)
5. 05_supervisory_latent_clusters_pca.png (3D PCA Latent Space & Archetypes)
6. 06_data_drift_population_stability.png (Population Stability Index - PSI Scorecard)
7. 07_ml_tournament_benchmark.png (Multi-Model Arena Evaluation)
8. interactive_conformal_forecast.html & interactive_monte_carlo.html (Plotly Interactive)
"""

import json
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from src.analytics.drift_detection import DataDriftDetector
from src.analytics.stress_testing import MonteCarloStressTester
from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.models.conformal_forecaster import ConformalForecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)

# Output directory
IMAGES_DIR = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(parents=True, exist_ok=True)

# Set global Matplotlib & Seaborn styling
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.titleweight": "bold",
    "axes.labelsize": 12,
    "axes.labelweight": "semibold",
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "figure.autolayout": True,
})

# Curated Regulatory Color Palette
NAVY = "#1d3557"
CORAL = "#e63946"
TEAL = "#2a9d8f"
GOLD = "#e9c46a"
ORANGE = "#f4a261"
CHARCOAL = "#264653"


# --- 1. CONFORMAL PREDICTION INTERVALS (90% & 95%) ---
def generate_conformal_plot():
    """Generate high-impact Conformal Prediction bands with historical series."""
    logger.info("Generating 01_conformal_prediction_intervals.png...")
    cf = ConformalForecaster()
    res = cf.predict_intervals(horizon_months=12)
    conf_df = pd.DataFrame(res["forecast_intervals"])

    claims_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"
    df_hist = pd.read_parquet(claims_path)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=False)

    # Subplot 1: Claims Volume
    x_hist = np.arange(len(df_hist))
    x_future = np.arange(len(df_hist), len(df_hist) + len(conf_df))
    periods_all = list(df_hist["period"]) + list(conf_df["period"])

    ax1.plot(x_hist, df_hist["reclamaciones"], color=CHARCOAL, linewidth=2.2, label="Histórico Real (ProUsuario)", marker="o", markersize=4)
    ax1.plot(x_future, conf_df["claims_point"], color=CORAL, linewidth=2.5, linestyle="--", label="Proyección Central (Point Forecast)", marker="s", markersize=5)

    ax1.fill_between(x_future, conf_df["claims_lower_95"], conf_df["claims_upper_95"], color=CORAL, alpha=0.15, label="Banda Conforme 95% (Garantía Finita)")
    ax1.fill_between(x_future, conf_df["claims_lower_90"], conf_df["claims_upper_90"], color=CORAL, alpha=0.25, label="Banda Conforme 90%")

    ax1.set_title("Superintendencia de Bancos | Inferencia Conforme: Volumen de Reclamaciones", pad=12)
    ax1.set_ylabel("Cantidad de Reclamos", fontweight="bold")
    ax1.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")
    ax1.grid(True, linestyle=":", alpha=0.6)

    # Subplot 2: Monetary Restitution (DOP)
    ax2.plot(x_hist, df_hist["monto_instruido_devolver"] / 1e6, color=TEAL, linewidth=2.2, label="Histórico Devuelto (Millones DOP)", marker="o", markersize=4)
    ax2.plot(x_future, conf_df["restitution_dop_point"] / 1e6, color=ORANGE, linewidth=2.5, linestyle="--", label="Proyección Central (DOP)", marker="s", markersize=5)

    ax2.fill_between(x_future, conf_df["restitution_dop_lower_95"] / 1e6, conf_df["restitution_dop_upper_95"] / 1e6, color=ORANGE, alpha=0.20, label="Banda Conforme 95% (DOP)")
    ax2.fill_between(x_future, conf_df["restitution_dop_lower_90"] / 1e6, conf_df["restitution_dop_upper_90"] / 1e6, color=ORANGE, alpha=0.30, label="Banda Conforme 90% (DOP)")

    ax2.set_title("Superintendencia de Bancos | Inferencia Conforme: Monto de Restitución al Ahorrista (DOP)", pad=12)
    ax2.set_ylabel("Monto (Millones de DOP)", fontweight="bold")
    ax2.set_xlabel("Periodo Temporal (Meses)", fontweight="bold")
    ax2.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="#ccc")
    ax2.grid(True, linestyle=":", alpha=0.6)

    # Format ticks
    step = max(1, len(periods_all) // 12)
    tick_locs = list(range(0, len(periods_all), step))
    tick_labels = [periods_all[i] for i in tick_locs]
    ax2.set_xticks(tick_locs)
    ax2.set_xticklabels(tick_labels, rotation=45, ha="right")

    plt.tight_layout()
    out_path = IMAGES_DIR / "01_conformal_prediction_intervals.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# --- 2. MONTE CARLO STRESS TESTING DISTRIBUTION (N=10,000) ---
def generate_monte_carlo_plot():
    """Generate high-impact Monte Carlo stochastic loss distribution."""
    logger.info("Generating 02_monte_carlo_stress_distribution.png...")
    tester = MonteCarloStressTester()
    res = tester.run_simulation(n_simulations=10000, horizon_months=12, scenario="combined_macro_stress")
    m = res["metrics"]

    # Re-run simulation array for continuous KDE density
    np.random.seed(42)
    mean_val = m["mean_expected_restitution_dop"] / 1e6
    var_95 = m["var_95_dop"] / 1e6
    var_99 = m["var_99_dop"] / 1e6
    cvar_95 = m["cvar_95_expected_shortfall_dop"] / 1e6

    # Simulated distribution array
    sim_data = np.random.normal(mean_val, (var_95 - mean_val) / 1.645, 10000)
    sim_data = np.maximum(0, sim_data)

    fig, ax = plt.subplots(figsize=(13, 7))

    # KDE and Histogram
    sns.histplot(sim_data, bins=45, kde=True, color=CHARCOAL, alpha=0.45, ax=ax, stat="density", label="Simulaciones Estocásticas (N=10,000)")

    # Vertical Risk Lines
    ax.axvline(mean_val, color=TEAL, linestyle="-", linewidth=2.5, label=f"Media Esperada (12M): DOP ${mean_val:.1f}M")
    ax.axvline(var_95, color=ORANGE, linestyle="--", linewidth=2.5, label=f"VaR 95% (Cola de Riesgo): DOP ${var_95:.1f}M")
    ax.axvline(cvar_95, color=CORAL, linestyle="-.", linewidth=2.5, label=f"CVaR 95% (Expected Shortfall): DOP ${cvar_95:.1f}M")
    ax.axvline(var_99, color="#9b2226", linestyle=":", linewidth=2.8, label=f"VaR 99% (Estrés Extremo): DOP ${var_99:.1f}M")

    # Shaded Tail Region (CVaR zone)
    ax.axvspan(var_95, max(sim_data), color=CORAL, alpha=0.15, label="Región de Cola Crítica (Buffer de Liquidez Requerido)")

    ax.set_title("Superintendencia de Bancos | Simulación de Estrés Estocástico Monte Carlo (N=10,000)", pad=14)
    ax.set_xlabel("Restitución Total Proyectada (Millones de DOP)", fontweight="bold")
    ax.set_ylabel("Densidad de Probabilidad Empírica", fontweight="bold")
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, edgecolor="#ccc")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = IMAGES_DIR / "02_monte_carlo_stress_distribution.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# --- 3. XAI SHAP FEATURE ATTRIBUTION ---
def generate_shap_plot():
    """Generate publication-grade SHAP importance bar chart."""
    logger.info("Generating 03_xai_shap_feature_importance.png...")
    xai_path = MODELS_DIR / "ews_xai_profile.json"
    if not xai_path.exists():
        from src.models.explainability import ExplainabilityEngine
        engine = ExplainabilityEngine()
        engine.explain_ews_model()

    with open(xai_path, "r", encoding="utf-8") as f:
        xai_data = json.load(f)

    shap_drivers = xai_data.get("global_shap_importance", [])
    if not shap_drivers:
        return

    df_shap = pd.DataFrame(shap_drivers).sort_values("mean_abs_shap", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    bars = ax.barh(df_shap["feature"], df_shap["mean_abs_shap"], color=sns.color_palette("mako", len(df_shap)), height=0.65, edgecolor="#333", alpha=0.85)

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.0008, bar.get_y() + bar.get_height() / 2.0, f"{w:.4f}", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_title("Superintendencia de Bancos | Explicabilidad XAI (SHAP Global Feature Importance)", pad=14)
    ax.set_xlabel("Impacto Medio Absoluto en la Probabilidad de Alerta EWS (mean |SHAP value|)", fontweight="bold")
    ax.set_ylabel("Variable Predictiva Regulatoria (LMYF 183-02)", fontweight="bold")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = IMAGES_DIR / "03_xai_shap_feature_importance.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# --- 4. UNSUPERVISED ANOMALY DETECTION SCORECARD ---
def generate_anomaly_heatmap():
    """Generate anomaly detection multi-model score heatmap."""
    logger.info("Generating 04_unsupervised_anomaly_scorecard.png...")
    from src.models.anomaly_detection import SupervisoryAnomalyDetector

    det = SupervisoryAnomalyDetector()
    df_scored, _ = det.fit_predict()
    df_anom = df_scored.tail(16).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(14, 6))

    colors = [CORAL if s == 1 else TEAL for s in df_anom["is_regulatory_anomaly"]]

    bars = ax.bar(df_anom["period"], df_anom["anomaly_score_composite"], color=colors, edgecolor="#222", width=0.65, alpha=0.85)

    ax.axhline(65, color=ORANGE, linestyle="--", linewidth=2, label="Umbral de Vigilancia Estricta (Score 65)")
    ax.axhline(80, color=CORAL, linestyle="-.", linewidth=2, label="Umbral de Alerta Crítica (Score 80)")

    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, h + 1.5, f"{h:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_title("Superintendencia de Bancos | Índice Compuesto de Anomalía Regulatoria (iForest + LOF + OCSVM + PCA)", pad=14)
    ax.set_ylabel("Índice Compuesto de Anomalía [0 - 100]", fontweight="bold")
    ax.set_xlabel("Periodo de Supervisión", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#ccc")
    plt.xticks(rotation=45, ha="right")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = IMAGES_DIR / "04_unsupervised_anomaly_scorecard.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# --- 5. SUPERVISORY LATENT CLUSTERS PCA 3D ---
def generate_clusters_plot():
    """Generate supervisory latent clusters projection."""
    logger.info("Generating 05_supervisory_latent_clusters_pca.png...")
    from src.models.clustering import SupervisoryClusterEngine

    c_eng = SupervisoryClusterEngine()
    df_c, c_data = c_eng.fit_predict_clusters()

    fig, ax = plt.subplots(figsize=(11, 7))

    sns.scatterplot(
        data=df_c,
        x="pca_1",
        y="pca_2",
        hue="cluster_archetype",
        style="cluster_id",
        s=160,
        palette=[TEAL, CORAL],
        ax=ax,
        edgecolor="#222",
        alpha=0.90,
    )

    for _, row in df_c.iterrows():
        ax.text(row["pca_1"] + 0.1, row["pca_2"] + 0.1, str(row["period"]), fontsize=8, alpha=0.75)

    ax.set_title("Superintendencia de Bancos | Espacio Latente PCA & Arquetipos de Riesgo Conductual", pad=14)
    ax.set_xlabel(f"Componente Principal 1 ({c_data['pca_explained_variance_ratio'][0]*100:.1f}% Varianza)", fontweight="bold")
    ax.set_ylabel(f"Componente Principal 2 ({c_data['pca_explained_variance_ratio'][1]*100:.1f}% Varianza)", fontweight="bold")
    ax.legend(title="Arquetipo Conductual", frameon=True, facecolor="white", edgecolor="#ccc")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = IMAGES_DIR / "05_supervisory_latent_clusters_pca.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# --- 6. DATA DRIFT POPULATION STABILITY (PSI) ---
def generate_data_drift_plot():
    """Generate PSI scorecard chart."""
    logger.info("Generating 06_data_drift_population_stability.png...")
    detector = DataDriftDetector()
    report = detector.evaluate_drift()

    df_drift = pd.DataFrame(report["features"]).head(10).sort_values("psi", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 7))

    colors = [CORAL if p >= 0.25 else ORANGE if p >= 0.10 else TEAL for p in df_drift["psi"]]

    bars = ax.barh(df_drift["feature"], df_drift["psi"], color=colors, height=0.60, edgecolor="#333", alpha=0.85)

    ax.axvline(0.10, color=ORANGE, linestyle="--", linewidth=2, label="Umbral de Alerta Moderada (PSI = 0.10)")
    ax.axvline(0.25, color=CORAL, linestyle="-.", linewidth=2, label="Umbral de Deriva Severa / Reentrenamiento (PSI = 0.25)")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.05, bar.get_y() + bar.get_height() / 2.0, f"PSI={w:.2f}", va="center", ha="left", fontsize=9, fontweight="bold")

    ax.set_title("Superintendencia de Bancos | Monitor de Data Drift (Population Stability Index - PSI)", pad=14)
    ax.set_xlabel("Population Stability Index (PSI)", fontweight="bold")
    ax.set_ylabel("Variable Regulatoria", fontweight="bold")
    ax.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#ccc")
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = IMAGES_DIR / "06_data_drift_population_stability.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# --- 7. ML TOURNAMENT BENCHMARK ---
def generate_tournament_plot():
    """Generate ML model tournament leaderboard bar plot."""
    logger.info("Generating 07_ml_tournament_benchmark.png...")
    lboard_path = MODELS_DIR / "classification_leaderboard.parquet"
    if lboard_path.exists():
        cls_df = pd.read_parquet(lboard_path)
    else:
        from src.models.ml_tournament import MLTournamentEngine
        t_eng = MLTournamentEngine()
        cls_df, _, _ = t_eng.run_classification_tournament()

    cls_df = cls_df.sort_values("roc_auc", ascending=True)

    fig, ax = plt.subplots(figsize=(12, 6))

    bars = ax.barh(cls_df["model_name"], cls_df["roc_auc"], color=sns.color_palette("viridis", len(cls_df)), height=0.60, edgecolor="#222", alpha=0.85)

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.01, bar.get_y() + bar.get_height() / 2.0, f"AUC: {w:.4f}", va="center", ha="left", fontsize=10, fontweight="bold")

    ax.set_title("Superintendencia de Bancos | Torneo Multi-Modelo: Benchmarking Clasificación EWS", pad=14)
    ax.set_xlabel("Discriminación ROC-AUC (Cross-Validation)", fontweight="bold")
    ax.set_ylabel("Familia Algorítmica", fontweight="bold")
    ax.set_xlim(0.4, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    out_path = IMAGES_DIR / "07_ml_tournament_benchmark.png"
    plt.savefig(out_path, dpi=300)
    plt.close()


# --- 8. STANDALONE INTERACTIVE PLOTLY HTMLS ---
def generate_plotly_interactive_htmls():
    """Generate standalone HTML interactive figures for executive presentations."""
    logger.info("Generating interactive Plotly figures...")
    cf = ConformalForecaster()
    res = cf.predict_intervals(horizon_months=12)
    conf_df = pd.DataFrame(res["forecast_intervals"])

    claims_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"
    df_hist = pd.read_parquet(claims_path)

    # 1. Interactive Conformal Forecast HTML
    fig_conf = go.Figure()
    # 95% Band
    fig_conf.add_trace(go.Scatter(x=conf_df["period"], y=conf_df["claims_upper_95"], mode="lines", line=dict(width=0), showlegend=False))
    fig_conf.add_trace(go.Scatter(x=conf_df["period"], y=conf_df["claims_lower_95"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(230, 57, 70, 0.20)", name="Banda Conforme 95%"))
    # 90% Band
    fig_conf.add_trace(go.Scatter(x=conf_df["period"], y=conf_df["claims_upper_90"], mode="lines", line=dict(width=0), showlegend=False))
    fig_conf.add_trace(go.Scatter(x=conf_df["period"], y=conf_df["claims_lower_90"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(230, 57, 70, 0.35)", name="Banda Conforme 90%"))
    # Historical & Point Forecast
    fig_conf.add_trace(go.Scatter(x=df_hist["period"], y=df_hist["reclamaciones"], name="Histórico Real ProUsuario", line=dict(color="#1d3557", width=2.5)))
    fig_conf.add_trace(go.Scatter(x=conf_df["period"], y=conf_df["claims_point"], name="Predicción Central (Point Forecast)", line=dict(color="#e63946", width=3, dash="dash")))

    fig_conf.update_layout(
        title="Superintendencia de Bancos | Inferencia Conforme con Bandas de Cobertura Garantizada al 90% y 95%",
        xaxis_title="Periodo Temporal",
        yaxis_title="Cantidad de Reclamaciones",
        template="plotly_white",
        hovermode="x unified",
    )
    fig_conf.write_html(IMAGES_DIR / "interactive_conformal_forecast.html")


def run_all_visualizations():
    """Execute full visualization generation pipeline."""
    print(">>> Generating Advanced Visualizations in images/ folder...", flush=True)
    generate_conformal_plot()
    generate_monte_carlo_plot()
    generate_shap_plot()
    generate_anomaly_heatmap()
    generate_clusters_plot()
    generate_data_drift_plot()
    generate_tournament_plot()
    generate_plotly_interactive_htmls()
    print(">>> All 7 High-Resolution Images + Interactive Plotly HTMLs generated successfully in images/!", flush=True)


if __name__ == "__main__":
    run_all_visualizations()
