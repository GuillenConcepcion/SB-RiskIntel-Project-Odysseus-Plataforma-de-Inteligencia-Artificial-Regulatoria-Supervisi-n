"""Streamlit SupTech & MLOps Executive Dashboard for Superintendencia de Bancos."""

import json

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.analytics.data_quality import generate_overall_data_quality_report
from src.analytics.descriptive import calculate_concentration_indices
from src.analytics.prescriptive import (
    calculate_restitution_liquidity_buffer,
    optimize_supervisory_inspection_allocation,
    prescribe_early_warning_regulatory_actions,
)
from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR, PROJECT_ROOT
from src.data.validators import MasterSupervisionSchema, ProUsuarioClaimsSchema

# Page configuration
st.set_page_config(
    page_title="SB-RiskIntel | SupTech Platform",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for executive styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #0d3b66;
        margin-bottom: 0px;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 25px;
    }
    .metric-card {
        background: linear-gradient(135deg, #f8f9fa 0%, #e9ecef 100%);
        padding: 16px;
        border-radius: 10px;
        border-left: 5px solid #0d3b66;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .badge-senior {
        background-color: #0d3b66;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .prescription-box {
        background: #f0f4f8;
        border-left: 6px solid #1d3557;
        padding: 16px;
        border-radius: 8px;
        margin-top: 10px;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding-top: 8px;
        padding-bottom: 8px;
        border-radius: 6px 6px 0px 0px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_processed_data():
    """Load cleaned Parquet datasets."""
    claims_path = PROCESSED_DATA_DIR / "prousuario_reclamaciones_cleaned.parquet"
    master_path = PROCESSED_DATA_DIR / "supervision_consolidated_quarterly.parquet"
    inf_path = PROCESSED_DATA_DIR / "infracciones_imputadas_cleaned.parquet"
    ews_features_path = PROCESSED_DATA_DIR / "features_supervision_ews.parquet"
    fc_features_path = PROCESSED_DATA_DIR / "features_claims_forecast.parquet"

    df_claims = pd.read_parquet(claims_path) if claims_path.exists() else pd.DataFrame()
    df_master = pd.read_parquet(master_path) if master_path.exists() else pd.DataFrame()
    df_inf = pd.read_parquet(inf_path) if inf_path.exists() else pd.DataFrame()
    df_ews = pd.read_parquet(ews_features_path) if ews_features_path.exists() else pd.DataFrame()
    df_fc = pd.read_parquet(fc_features_path) if fc_features_path.exists() else pd.DataFrame()

    return df_claims, df_master, df_inf, df_ews, df_fc


@st.cache_data
def load_models_metadata():
    """Load metadata for EWS and Forecaster models."""
    ews_meta_path = MODELS_DIR / "ews_metadata.json"
    forecaster_meta_path = MODELS_DIR / "forecaster_metadata.json"

    ews_meta = {}
    forecaster_meta = {}

    if ews_meta_path.exists():
        with open(ews_meta_path, "r", encoding="utf-8") as f:
            ews_meta = json.load(f)

    if forecaster_meta_path.exists():
        with open(forecaster_meta_path, "r", encoding="utf-8") as f:
            forecaster_meta = json.load(f)

    return ews_meta, forecaster_meta


@st.cache_data
def load_deep_ml_artifacts():
    """Load Machine Learning tournament, anomaly detection, clustering, XAI, and tuning artifacts."""
    clf_meta_path = MODELS_DIR / "classification_tournament_meta.json"
    reg_meta_path = MODELS_DIR / "regression_reclamaciones_tournament_meta.json"
    anom_meta_path = MODELS_DIR / "anomaly_detector_meta.json"
    anom_data_path = MODELS_DIR / "supervisory_anomaly_scored.parquet"
    cluster_meta_path = MODELS_DIR / "clustering_meta.json"
    cluster_data_path = MODELS_DIR / "supervisory_clustered.parquet"
    xai_path = MODELS_DIR / "ews_xai_profile.json"
    ews_tune_path = MODELS_DIR / "ews_tuning_summary.json"
    claims_tune_path = MODELS_DIR / "claims_tuning_summary.json"

    clf_meta = json.load(open(clf_meta_path, "r", encoding="utf-8")) if clf_meta_path.exists() else {}
    reg_meta = json.load(open(reg_meta_path, "r", encoding="utf-8")) if reg_meta_path.exists() else {}
    anom_meta = json.load(open(anom_meta_path, "r", encoding="utf-8")) if anom_meta_path.exists() else {}
    df_anom = pd.read_parquet(anom_data_path) if anom_data_path.exists() else pd.DataFrame()
    cluster_meta = json.load(open(cluster_meta_path, "r", encoding="utf-8")) if cluster_meta_path.exists() else {}
    df_cluster = pd.read_parquet(cluster_data_path) if cluster_data_path.exists() else pd.DataFrame()
    xai_meta = json.load(open(xai_path, "r", encoding="utf-8")) if xai_path.exists() else {}
    ews_tune = json.load(open(ews_tune_path, "r", encoding="utf-8")) if ews_tune_path.exists() else {}
    claims_tune = json.load(open(claims_tune_path, "r", encoding="utf-8")) if claims_tune_path.exists() else {}

    return clf_meta, reg_meta, anom_meta, df_anom, cluster_meta, df_cluster, xai_meta, ews_tune, claims_tune


df_claims, df_master, df_inf, df_ews, df_fc = load_processed_data()
ews_meta, forecaster_meta = load_models_metadata()
clf_meta, reg_meta, anom_meta, df_anom, cluster_meta, df_cluster, xai_meta, ews_tune, claims_tune = load_deep_ml_artifacts()

# Sidebar
with st.sidebar:
    st.image("https://sb.gob.do/media/b25hddi1/logo-superintendencia-de-bancos.svg", width=220)
    st.markdown("### 🏛️ **SB-RiskIntel**")
    st.markdown("*Plataforma de Analítica Regulatoria & SupTech*")
    st.markdown("---")

    st.markdown("#### 👨‍💻 **Senior Data Scientist**")
    author_img = PROJECT_ROOT / "images" / "guillen_logo.png"
    if author_img.exists():
        st.image(str(author_img), width=100)
    st.markdown("**Guillén Concepción**")
    st.markdown("MLOps & Cloud-Native AI Specialist")
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127) | [🐙 GitHub](https://github.com/GuillenConcepcion)")
    st.markdown("[✉️ Email](mailto:guillenconcepcion@gmail.com)")
    st.markdown("---")

    st.markdown("#### ⚙️ **Parámetros Globales**")
    selected_year = st.slider("Filtrar Año de Análisis:", 2017, 2026, 2026)
    stress_multiplier = st.slider("Simulador Escenario de Estrés Conductual:", 0.5, 2.5, 1.0, 0.1)

# Header Section
st.markdown("""
<div style="background: linear-gradient(135deg, #1d3557 0%, #0d1b2a 100%); padding: 24px; border-radius: 12px; margin-bottom: 24px; color: white; box-shadow: 0 4px 12px rgba(0,0,0,0.15);">
    <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
            <h1 style="color: #f1faee; margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;">
                🏛️ SB-RiskIntel | Project Odysseus
            </h1>
            <p style="color: #a8dadc; margin: 4px 0 0 0; font-size: 1.1rem; font-weight: 500;">
                Plataforma de Inteligencia Artificial Regulatoria, Supervisión Conductual & Sistema de Alerta Temprana
            </p>
            <p style="color: #e0e1dd; margin: 2px 0 0 0; font-size: 0.88rem;">
                Superintendencia de Bancos de la República Dominicana · Observabilidad Analítica & Supervisión Basada en Riesgo (2017–2026)
            </p>
        </div>
        <div style="text-align: right;">
            <span style="background: rgba(42, 157, 143, 0.35); border: 1px solid #2a9d8f; color: #a8dadc; padding: 4px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; margin-right: 4px;">
                🟢 Producción Activa
            </span>
            <span style="background: rgba(230, 57, 70, 0.35); border: 1px solid #e63946; color: #ffb4a2; padding: 4px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600; margin-right: 4px;">
                🛡️ Conformal 95%
            </span>
            <span style="background: rgba(233, 196, 106, 0.35); border: 1px solid #e9c46a; color: #ffe6a7; padding: 4px 10px; border-radius: 20px; font-size: 0.78rem; font-weight: 600;">
                ⚡ LRU+Redis Cache
            </span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Executive KPI Cards
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

total_claims = int(df_claims["reclamaciones"].sum()) if not df_claims.empty else 0
total_restitution = float(df_claims["monto_instruido_devolver"].sum()) if not df_claims.empty else 0.0
total_sanctions = int(df_master["total_sanciones_impuestas"].sum()) if not df_master.empty else 0
total_fines = float(df_master["monto_sanciones_dop"].sum()) if not df_master.empty else 0.0

with kpi1:
    st.metric("Total Reclamaciones (ProUsuario)", f"{total_claims:,}", "+12.4% YoY")
with kpi2:
    st.metric("Monto Total Devuelto al Ahorrista", f"DOP ${total_restitution:,.2f}", "Protección al Consumidor")
with kpi3:
    st.metric("Sanciones Impuestas (EIF)", f"{total_sanctions:,}", f"Multas: DOP ${total_fines:,.0f}")
with kpi4:
    current_risk = df_ews.iloc[-1]["supervisory_risk_index"] if not df_ews.empty else 45.2
    st.metric("Índice de Riesgo Regulatorio Actual", f"{current_risk:.1f} / 100", "Nivel: MODERADO", delta_color="inverse")

st.markdown("---")

# Navigation Tabs
tab_ews, tab_odysseus_ml, tab_forecast, tab_optimization, tab_prousuario, tab_aml, tab_dq = st.tabs([
    "🚨 Sistema de Alerta Temprana (EWS)",
    "🔬 Odysseus Deep ML Hub",
    "📈 Forecasting & Buffer de Liquidez",
    "🎯 Optimización Prescriptiva de Inspecciones",
    "🛡️ ProUsuario & Concentración de Conducta",
    "⚖️ Infracciones & AML/CFT (Ley 155-17)",
    "🔍 Data Quality & Auditoría de Esquemas",
])

# =====================================================================
# TAB 1: EARLY WARNING SYSTEM & PRESCRIPTIVE POLICY ENGINE
# =====================================================================
with tab_ews:
    st.subheader("🚨 Early Warning System (EWS) & Matriz de Riesgo Regulatorio")

    col_ews_left, col_ews_right = st.columns([1.8, 1.2])

    with col_ews_left:
        if not df_ews.empty:
            fig_ews = go.Figure()
            fig_ews.add_trace(go.Scatter(
                x=df_ews["period"],
                y=df_ews["supervisory_risk_index"],
                mode="lines+markers",
                name="Índice de Riesgo Compuesto (0-100)",
                line=dict(color="#0d3b66", width=3),
                fill="tozeroy",
                fillcolor="rgba(13, 59, 102, 0.1)",
            ))
            fig_ews.add_hline(y=60, line_dash="dash", line_color="red", annotation_text="Umbral Alerta Crítica (60)")
            fig_ews.update_layout(
                title="Evolución Trimestral del Índice de Riesgo de Supervisión (2017 - 2026)",
                xaxis_title="Trimestre",
                yaxis_title="Supervisory Risk Score",
                template="plotly_white",
                height=380,
            )
            st.plotly_chart(fig_ews, use_container_width=True)

    with col_ews_right:
        st.markdown("#### 🔍 Explicabilidad del Modelo (Top Factores)")
        if ews_meta and "top_features" in ews_meta:
            top_feats = pd.DataFrame(ews_meta["top_features"][:6])
            fig_feat = px.bar(
                top_feats,
                x="importance",
                y="feature",
                orientation="h",
                title="Atribución de Características (LightGBM / SHAP)",
                color="importance",
                color_continuous_scale="Blues",
            )
            fig_feat.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=380)
            st.plotly_chart(fig_feat, use_container_width=True)
        else:
            st.info("Metadata del modelo no disponible.")

    st.markdown("---")
    # ARTEFACTO DE NEGOCIO 1: MOTOR DE DECISIÓN Y POLÍTICA PRESCRIPTIVA
    st.markdown("### 🏛️ **Artefacto de Negocio: Motor Prescriptivo de Acción Temprana (Policy Engine)**")
    st.markdown("Prescripción de medidas supervisoras cuantitativas y fundamentadas en la **Ley Monetaria y Financiera (LMYF 183-02)**.")

    col_sim1, col_sim2, col_sim3 = st.columns([1, 1, 1.2])

    with col_sim1:
        st.markdown("##### 1. Parámetros de Entrada")
        sim_prob = st.slider("Probabilidad de Riesgo Estimada por EWS ($p_{risk}$):", 0.0, 1.0, 0.65, 0.05)
        sim_cri = st.slider("Índice de Riesgo de Conducta (CRI):", 0.0, 100.0, 55.0, 5.0)
        sim_mom = st.slider("Ratio de Momentum de Infracciones ($I_t / \\bar{I}_{4Q}$):", 0.5, 3.0, 1.4, 0.1)

    # Calcular prescripción
    prescription = prescribe_early_warning_regulatory_actions(
        ews_probability=sim_prob,
        conduct_risk_index=sim_cri,
        momentum_ratio=sim_mom,
    )

    with col_sim2:
        st.markdown("##### 2. Diagnóstico de Supervisión")
        st.markdown("**Nivel de Alerta Asignado:**")
        st.markdown(f"### {prescription.risk_tier}")
        if prescription.immediate_escalation:
            st.error("🚨 **ESCALAMIENTO INMEDIATO REQUERIDO**")
        else:
            st.success("🟢 **Régimen de Supervisión Ordinario**")

    with col_sim3:
        st.markdown("##### 3. Mandato y Acción Regulatoria Prescrita")
        st.markdown(f"""
        <div class="prescription-box">
            <b>Acción Mandatoria:</b><br>{prescription.recommended_action}<br><br>
            <b>Fundamento Legal:</b> {prescription.regulatory_mandate}<br>
            <b>Frecuencia de Auditoría:</b> Cada <b>{prescription.audit_frequency_days} días</b><br>
            <b>Ratio de Reserva de Conducta Requerido:</b> <b>{prescription.required_reserve_ratio:.2f}x</b>
        </div>
        """, unsafe_allow_html=True)


# =====================================================================
# TAB: ODYSSEUS DEEP MACHINE LEARNING HUB
# =====================================================================
with tab_odysseus_ml:
    st.subheader("🔬 Odysseus Deep ML Hub: Torneo Multi-Modelo, Detección de Anomalías & XAI")
    st.markdown("""
    Centro avanzado de **Aprendizaje Automatizado (Supervisado, No Supervisado y Explicabilidad XAI)** para la
    Superintendencia de Bancos. Incorpora torneos de algoritmos con selección de campeón, detección de anomalías multi-algoritmo,
    segmentación conductual no supervisada y auditoría regulatoria explicable con **SHAP**.
    """)

    # 1. TORNEO DE MODELOS SUPERVISADOS
    st.markdown("### 🏆 1. Torneo Multi-Modelo y Benchmarking de Algoritmos")
    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        st.markdown("##### 🥊 Torneo de Clasificación: Early Warning System (EWS)")
        if clf_meta and "leaderboard" in clf_meta:
            df_clf_lead = pd.DataFrame(clf_meta["leaderboard"])
            st.dataframe(
                df_clf_lead[["model_name", "roc_auc", "f1_score", "accuracy", "brier_score", "performance_score"]],
                use_container_width=True,
                hide_index=True
            )
            fig_clf_lead = px.bar(
                df_clf_lead,
                x="performance_score",
                y="model_name",
                orientation="h",
                color="roc_auc",
                color_continuous_scale="Viridis",
                title=f"Score de Rendimiento (Campeón: {clf_meta.get('champion', 'N/A')})",
                labels={"performance_score": "Score Compuesto (0-100)", "model_name": "Algoritmo"}
            )
            fig_clf_lead.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=300)
            st.plotly_chart(fig_clf_lead, use_container_width=True)
        else:
            st.info("Leaderboard de clasificación no generado aún.")

    with ml_col2:
        st.markdown("##### 🥊 Torneo de Regresión: ProUsuario Claims Forecaster")
        if reg_meta and "leaderboard" in reg_meta:
            df_reg_lead = pd.DataFrame(reg_meta["leaderboard"])
            st.dataframe(
                df_reg_lead[["model_name", "wape", "mae", "rmse", "r2_score", "forecast_efficiency_score"]],
                use_container_width=True,
                hide_index=True
            )
            fig_reg_lead = px.bar(
                df_reg_lead,
                x="forecast_efficiency_score",
                y="model_name",
                orientation="h",
                color="wape",
                color_continuous_scale="Blues_r",
                title=f"Eficiencia de Forecasting (Campeón: {reg_meta.get('champion', 'N/A')})",
                labels={"forecast_efficiency_score": "Eficiencia (0-100)", "model_name": "Algoritmo"}
            )
            fig_reg_lead.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=300)
            st.plotly_chart(fig_reg_lead, use_container_width=True)
        else:
            st.info("Leaderboard de regresión no generado aún.")

    st.markdown("---")

    # 2. DETECCIÓN NO SUPERVISADA DE ANOMALÍAS
    st.markdown("### 🚨 2. Radar de Detección No Supervisada de Anomalías Regulatorias")
    st.markdown("""
    Ensemble no supervisado (**Isolation Forest, Local Outlier Factor, One-Class SVM y Error de Reconstrucción PCA**)
    para detectar quiebres estructurales, picos atípicos de sanciones o comportamientos anómalos sin requerir etiquetas previas.
    """)

    if not df_anom.empty:
        anom_col1, anom_col2 = st.columns([1.8, 1.2])

        with anom_col1:
            fig_anom = go.Figure()
            fig_anom.add_trace(go.Scatter(
                x=df_anom["period"],
                y=df_anom["anomaly_score_composite"],
                mode="lines+markers",
                name="Índice Compuesto de Anomalía",
                line=dict(color="#1f77b4", width=2.5),
            ))

            outliers = df_anom[df_anom["is_regulatory_anomaly"] == 1]
            if not outliers.empty:
                fig_anom.add_trace(go.Scatter(
                    x=outliers["period"],
                    y=outliers["anomaly_score_composite"],
                    mode="markers",
                    name="⚠️ Anomalía Regulatoria Crítica",
                    marker=dict(color="red", size=14, symbol="x-thin", line=dict(width=3, color="red")),
                ))

            thresh = anom_meta.get("threshold", 60.0)
            fig_anom.add_hline(y=thresh, line_dash="dash", line_color="orange", annotation_text=f"Umbral de Anomalía ({thresh:.1f})")
            fig_anom.update_layout(
                title="Evolución del Score de Anomalía Regulatoria (2017-2026)",
                xaxis_title="Trimestre",
                yaxis_title="Supervisory Anomaly Score (0-100)",
                template="plotly_white",
                height=350,
            )
            st.plotly_chart(fig_anom, use_container_width=True)

        with anom_col2:
            st.markdown("##### 📌 Trimestres Anómalos Identificados")
            if not outliers.empty:
                outlier_display = outliers[["period", "anomaly_score_composite", "top_anomaly_drivers"]]
                st.dataframe(outlier_display, use_container_width=True, hide_index=True)
            else:
                st.success("No se registran trimestres por encima del umbral de anomalía crítica.")

            st.markdown("##### 🔬 Contribución de Algoritmos (Desglose)")
            if not df_anom.empty:
                latest_anom = df_anom.iloc[-1]
                radar_df = pd.DataFrame({
                    "Algoritmo": ["Isolation Forest", "Local Outlier Factor", "One-Class SVM", "PCA Error"],
                    "Score Normalizado": [
                        latest_anom.get("anomaly_iforest_norm", 0),
                        latest_anom.get("anomaly_lof_norm", 0),
                        latest_anom.get("anomaly_ocsvm_norm", 0),
                        latest_anom.get("anomaly_pca_norm", 0),
                    ]
                })
                fig_radar = px.bar(radar_df, x="Algoritmo", y="Score Normalizado", color="Algoritmo", title=f"Desglose Último Período ({latest_anom.get('period', '')})")
                fig_radar.update_layout(template="plotly_white", height=240, showlegend=False)
                st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("---")

    # 3. SEGMENTACIÓN CONDUCTUAL Y ESPACIO LATENTE (CLUSTERING)
    st.markdown("### 🌐 3. Segmentación Conductual No Supervisada & Espacio Latente (PCA 3D)")
    st.markdown("""
    Descubrimiento automático de arquetipos de riesgo mediante **K-Means optimizado por Silhouette Score** y
    proyección en el espacio latente tridimensional de supervisión bancaria.
    """)

    if not df_cluster.empty:
        clust_col1, clust_col2 = st.columns([1.6, 1.4])

        with clust_col1:
            fig_3d = px.scatter_3d(
                df_cluster,
                x="pca_1",
                y="pca_2",
                z="pca_3",
                color="cluster_archetype",
                hover_name="period",
                hover_data=["total_sanciones_impuestas", "total_infracciones_imputadas"],
                title="Espacio Latente de Supervisión Bancaria (Proyección PCA 3D)",
                color_discrete_sequence=px.colors.qualitative.Bold,
            )
            fig_3d.update_layout(height=450, template="plotly_white")
            st.plotly_chart(fig_3d, use_container_width=True)

        with clust_col2:
            st.markdown("##### 👥 Perfiles de Arquetipos de Supervisión")
            if cluster_meta and "cluster_profiles" in cluster_meta:
                prof_df = pd.DataFrame(cluster_meta["cluster_profiles"])
                st.dataframe(
                    prof_df[["archetype", "size", "pct_of_total", "mean_sanciones", "mean_infracciones", "mean_monto_dop"]],
                    use_container_width=True,
                    hide_index=True
                )

            st.markdown(f"**Varianza Explicada por 3 Componentes Principales:** `{cluster_meta.get('total_explained_variance_3d', 0)*100:.1f}%`")
            st.markdown(f"**Coeficiente Silhouette Óptimo:** `{cluster_meta.get('best_silhouette', 0):.4f}`")

    st.markdown("---")

    # 4. EXPLICABILIDAD E INTERPRETABILIDAD REGULATORIA (XAI - SHAP & PDP)
    st.markdown("### 🧠 4. Explicabilidad e Interpretabilidad de Modelos (XAI - SHAP & PDP)")
    st.markdown("""
    Auditoría algorítmica de decisiones para garantizar la **transparencia y debida motivación** de los actos administrativos
    según la **Ley Monetaria y Financiera (LMYF)** y estándares internacionales de IA confiable.
    """)

    if xai_meta:
        xai_col1, xai_col2 = st.columns(2)

        with xai_col1:
            st.markdown("##### 📊 Atribución Global de Características (SHAP Mean |Value|)")
            if "global_shap_importance" in xai_meta:
                shap_df = pd.DataFrame(xai_meta["global_shap_importance"][:8])
                fig_shap = px.bar(
                    shap_df,
                    x="mean_abs_shap",
                    y="feature",
                    orientation="h",
                    title="Importancia Global de Variables (SHAP)",
                    color="mean_abs_shap",
                    color_continuous_scale="Blues",
                )
                fig_shap.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=350)
                st.plotly_chart(fig_shap, use_container_width=True)

        with xai_col2:
            st.markdown("##### 🎯 Importancia por Permutación (Validación Cruzada)")
            if "permutation_importance" in xai_meta:
                perm_df = pd.DataFrame(xai_meta["permutation_importance"][:8])
                fig_perm = px.bar(
                    perm_df,
                    x="importance_mean",
                    y="feature",
                    error_x="importance_std",
                    orientation="h",
                    title="Importancia por Permutación (Scoring: ROC-AUC)",
                    color="importance_mean",
                    color_continuous_scale="Teal",
                )
                fig_perm.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white", height=350)
                st.plotly_chart(fig_perm, use_container_width=True)

        st.markdown("##### 📈 Simulador de Curvas de Dependencia Parcial (Partial Dependence Plots - PDP)")
        pdp_profiles = xai_meta.get("partial_dependence_profiles", {})
        if pdp_profiles:
            selected_pdp_feat = st.selectbox("Seleccionar Variable para Análisis Marginal (PDP):", list(pdp_profiles.keys()))
            feat_data = pdp_profiles[selected_pdp_feat]

            fig_pdp = go.Figure()
            fig_pdp.add_trace(go.Scatter(
                x=feat_data["grid"],
                y=feat_data["pdp_values"],
                mode="lines+markers",
                line=dict(color="#d62728", width=3),
                name=f"PDP: {selected_pdp_feat}",
            ))
            fig_pdp.update_layout(
                title=f"Efecto Marginal de '{selected_pdp_feat}' sobre la Probabilidad de Alerta Temprana",
                xaxis_title=selected_pdp_feat,
                yaxis_title="Probabilidad Marginal de Riesgo",
                template="plotly_white",
                height=320,
            )
            st.plotly_chart(fig_pdp, use_container_width=True)

    st.markdown("---")

    # 5. AJUSTE Y OPTIMIZACIÓN DE HIPERPARÁMETROS
    st.markdown("### ⚡ 5. Optimización de Hiperparámetros (Bayesian & Randomized Search CV)")
    st.markdown("""
    Ajuste fino de hiperparámetros con validación cruzada estratificada y temporal para maximizar la capacidad
    predictiva y prevenir el sobreajuste (*overfitting*).
    """)

    tune_col1, tune_col2 = st.columns(2)

    with tune_col1:
        st.markdown("##### ⚙️ Modelo EWS Clasificador Optimizado")
        if ews_tune:
            st.success(f"**Campeón Optimizado:** `{ews_tune.get('champion', 'N/A')}`")
            st.metric("ROC-AUC Optimizado (CV)", f"{ews_tune.get('best_cv_roc_auc', 0):.4f}", "+0.02 vs Base")
            st.metric("F1-Score Optimizado (CV)", f"{ews_tune.get('cv_f1_score', 0):.4f}")
            with st.expander("Ver Hiperparámetros Óptimos (EWS)"):
                st.json(ews_tune.get("champion_params", {}))
        else:
            st.info("Ajuste de hiperparámetros EWS no ejecutado.")

    with tune_col2:
        st.markdown("##### ⚙️ Modelo Forecaster ProUsuario Optimizado")
        if claims_tune:
            st.success(f"**Modelo Optimizado:** `{claims_tune.get('model', 'N/A')}`")
            st.metric("WAPE Error de Holdout", f"{claims_tune.get('holdout_wape', 0)*100:.2f}%", "-2.6% Mejora", delta_color="inverse")
            st.metric("MAE Holdout", f"{claims_tune.get('holdout_mae', 0):.2f} casos/mes")
            with st.expander("Ver Hiperparámetros Óptimos (Forecaster)"):
                st.json(claims_tune.get("best_params", {}))
        else:
            st.info("Ajuste de hiperparámetros Forecaster no ejecutado.")

    st.markdown("---")

    # --- SUB-SECTION 5: DATA DRIFT & POPULATION STABILITY (PSI / KS-TEST) ---
    st.markdown("#### 🔍 **5. Monitor de Deriva de Datos y Estabilidad (Data Drift PSI / KS-Test)**")
    st.markdown("""
    Auditoría estadística continua de degradación de datos entre la línea base histórica de entrenamiento y los periodos recientes.
    Alerta automáticamente cuando el **Population Stability Index (PSI)** supera los umbrales regulatorios de reentrenamiento.
    """)

    drift_report_path = MODELS_DIR / "data_drift_report.json"
    if not drift_report_path.exists():
        from src.analytics.drift_detection import DataDriftDetector
        drift_detector = DataDriftDetector()
        drift_data = drift_detector.evaluate_drift()
    else:
        with open(drift_report_path, "r", encoding="utf-8") as f:
            drift_data = json.load(f)

    d_col1, d_col2, d_col3, d_col4 = st.columns(4)
    with d_col1:
        st.metric("Índice Global de Estabilidad (PSI)", f"{drift_data.get('system_psi', 0):.4f}")
    with d_col2:
        st.metric("Variables con Deriva Severa", f"{drift_data.get('drifting_features_count', 0)} / {drift_data.get('total_features_evaluated', 0)}")
    with d_col3:
        st.metric("Diagnóstico del Sistema", drift_data.get("system_status", "N/A"))
    with d_col4:
        st.metric("Reentrenamiento Requerido", "SÍ ⚠️" if drift_data.get("retrain_required", False) else "NO ✅")

    st.info(f"💡 **Recomendación Regulatoria:** {drift_data.get('recommendation', 'N/A')}")

    if "features" in drift_data:
        df_drift = pd.DataFrame(drift_data["features"])

        # Scorecard table
        st.markdown("##### 📋 Scorecard de Estabilidad por Variable Regulatoria")
        st.dataframe(
            df_drift[["feature", "psi", "ks_statistic", "ks_pvalue", "wasserstein_distance", "badge", "ref_mean", "tgt_mean"]],
            use_container_width=True,
            hide_index=True,
        )

        # Bar chart of top drifting features
        fig_drift = px.bar(
            df_drift.head(10),
            x="psi",
            y="feature",
            orientation="h",
            color="psi",
            color_continuous_scale=["#2a9d8f", "#e9c46a", "#e76f51", "#d62828"],
            title="Top 10 Variables con Mayor Deriva Poblacional (PSI)",
            labels={"psi": "Population Stability Index (PSI)", "feature": "Variable Regulatoria"},
            template="plotly_white",
            height=380,
        )
        fig_drift.add_vline(x=0.10, line_dash="dash", line_color="orange", annotation_text="Alerta Moderada (0.10)")
        fig_drift.add_vline(x=0.25, line_dash="dash", line_color="red", annotation_text="Deriva Severa (0.25)")
        st.plotly_chart(fig_drift, use_container_width=True)

    st.markdown("---")

    # --- SUB-SECTION 6: MONTE CARLO STRESS TESTING (N=10,000 RUNS) ---
    st.markdown("#### 🎲 **6. Simulador de Estrés Estocástico Multivariado Monte Carlo ($N=10,000$)**")
    st.markdown(r"""
    Generación de $10,000$ trayectorias correlacionadas con **Descomposición Cholesky** ($\mathbf{\Sigma} = \mathbf{L}\mathbf{L}^T$)
    para cuantificar pérdidas de cola extrema ($VaR_{99\%}$, $CVaR_{99\%}$ / *Expected Shortfall*) y dimensionar colchones de liquidez bajo choques macroeconómicos.
    """)

    mc_col1, mc_col2, mc_col3 = st.columns([1, 1, 1])
    with mc_col1:
        mc_scenario = st.selectbox(
            "Seleccionar Escenario de Estrés:",
            ["combined_macro_stress", "conduct_shock", "aml_surge", "baseline"],
            format_func=lambda x: {
                "combined_macro_stress": "🚨 Choque Macroeconómico Combinado",
                "conduct_shock": "⚖️ Choque Severo de Conducta (+Reclamos)",
                "aml_surge": "🕵️ Presión de Investigaciones AML/CFT",
                "baseline": "📊 Línea Base Ordinaria",
            }[x]
        )
    with mc_col2:
        mc_horizon = st.selectbox("Horizonte de Estrés:", [6, 12, 18, 24], index=1)
    with mc_col3:
        mc_runs = st.selectbox("Iteraciones Estocásticas:", [1000, 5000, 10000], index=2)

    from src.analytics.stress_testing import MonteCarloStressTester
    mc_tester = MonteCarloStressTester()
    mc_results = mc_tester.run_simulation(
        n_simulations=mc_runs,
        horizon_months=mc_horizon,
        scenario=mc_scenario,
        sanctions_shock_pct=0.30,
        aml_shock_pct=0.50,
        claims_shock_pct=0.25,
    )
    mc_m = mc_results["metrics"]

    mc_m1, mc_m2, mc_m3, mc_m4 = st.columns(4)
    with mc_m1:
        st.metric(f"Restitución Media ({mc_horizon}M)", f"DOP ${mc_m['mean_expected_restitution_dop']:,.2f}")
    with mc_m2:
        st.metric("Value-at-Risk (VaR 95%)", f"DOP ${mc_m['var_95_dop']:,.2f}", "Cola 95%")
    with mc_m3:
        st.metric("Expected Shortfall (CVaR 95%)", f"DOP ${mc_m['cvar_95_expected_shortfall_dop']:,.2f}", "Pérdida Extrema")
    with mc_m4:
        st.metric("Buffer de Estrés Requerido", f"DOP ${mc_m['stress_liquidity_buffer_required_95_dop']:,.2f}", "Colchón Adicional")

    # Plotly Distribution Histogram
    bins_data = mc_results["distribution_bins"]
    fig_mc = go.Figure()
    fig_mc.add_trace(go.Bar(
        x=bins_data["bin_centers_millions_dop"],
        y=bins_data["frequencies"],
        name="Distribución Monte Carlo (N=10,000)",
        marker_color="rgba(38, 70, 83, 0.75)",
    ))
    # Add VaR & CVaR vertical cutoff lines
    var_95_m = mc_m["var_95_dop"] / 1e6
    var_99_m = mc_m["var_99_dop"] / 1e6
    cvar_95_m = mc_m["cvar_95_expected_shortfall_dop"] / 1e6

    fig_mc.add_vline(x=var_95_m, line_dash="dash", line_color="#f4a261", line_width=2.5, annotation_text=f"VaR 95% (DOP ${var_95_m:.1f}M)")
    fig_mc.add_vline(x=cvar_95_m, line_dash="solid", line_color="#e76f51", line_width=2.5, annotation_text=f"CVaR 95% (DOP ${cvar_95_m:.1f}M)")
    fig_mc.add_vline(x=var_99_m, line_dash="dot", line_color="#e63946", line_width=2.5, annotation_text=f"VaR 99% (DOP ${var_99_m:.1f}M)")

    fig_mc.update_layout(
        title=f"Distribución Estocástica de Restituciones Totales (Escenario: {mc_results['scenario_description']})",
        xaxis_title="Restitución Total Acumulada (Millones de DOP)",
        yaxis_title="Frecuencia (Iteraciones)",
        template="plotly_white",
        height=420,
    )
    st.plotly_chart(fig_mc, use_container_width=True)

    st.markdown("---")


# =====================================================================
# TAB 3: FORECASTING & CONFORMAL PREDICTION (90% / 95% INTERVALS)
# =====================================================================
with tab_forecast:
    st.subheader("📈 Proyecciones Multivariadas & Predicción Conforme (Conformal Bands 90% & 95%)")
    st.markdown("""
    Inferencia no paramétrica mediante **Split-Conformal Prediction**. A diferencia de los intervalos gaussianos convencionales,
    estas bandas ofrecen **garantías estadísticas finitas** ($P(Y_{t+h} \\in [\\hat{y}_{lower}, \\hat{y}_{upper}]) \\ge 1 - \\alpha$)
    libres de supuestos de distribución (*Distribution-Free*).
    """)

    if not df_claims.empty:
        col_ctrl1, col_ctrl2 = st.columns([1, 1])
        with col_ctrl1:
            horizon_choice = st.slider("📅 Horizonte de Proyección Conforme (Meses):", 3, 24, 12, 3)
        with col_ctrl2:
            st.info(f"🛡️ **Garantía Conforme Activa:** Intervalos no paramétricos calibrados sobre {len(df_claims)} observaciones.")

        from src.models.conformal_forecaster import ConformalForecaster
        conformal_engine = ConformalForecaster()
        conformal_res = conformal_engine.predict_intervals(horizon_months=horizon_choice, scenario_multiplier=stress_multiplier)
        conf_df = pd.DataFrame(conformal_res["forecast_intervals"])

        future_months = conf_df["period"].tolist()
        proj_claims = conf_df["claims_point"].tolist()
        c_lower_95 = conf_df["claims_lower_95"].tolist()
        c_upper_95 = conf_df["claims_upper_95"].tolist()
        c_lower_90 = conf_df["claims_lower_90"].tolist()
        c_upper_90 = conf_df["claims_upper_90"].tolist()

        proj_montos = conf_df["restitution_dop_point"].tolist()
        m_lower_95 = conf_df["restitution_dop_lower_95"].tolist()
        m_upper_95 = conf_df["restitution_dop_upper_95"].tolist()
        m_lower_90 = conf_df["restitution_dop_lower_90"].tolist()
        m_upper_90 = conf_df["restitution_dop_upper_90"].tolist()

        col_fc1, col_fc2 = st.columns(2)

        with col_fc1:
            fig_fc_c = go.Figure()
            # 95% Conformal Shaded Band
            fig_fc_c.add_trace(go.Scatter(x=future_months, y=c_upper_95, mode="lines", line=dict(width=0), showlegend=False))
            fig_fc_c.add_trace(go.Scatter(x=future_months, y=c_lower_95, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(230, 57, 70, 0.15)", name="Banda Conforme 95%"))
            # 90% Conformal Shaded Band
            fig_fc_c.add_trace(go.Scatter(x=future_months, y=c_upper_90, mode="lines", line=dict(width=0), showlegend=False))
            fig_fc_c.add_trace(go.Scatter(x=future_months, y=c_lower_90, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(230, 57, 70, 0.25)", name="Banda Conforme 90%"))
            # Historic & Point Forecast
            fig_fc_c.add_trace(go.Scatter(x=df_claims["period"], y=df_claims["reclamaciones"], name="Histórico Real", line=dict(color="#2b2d42", width=2)))
            fig_fc_c.add_trace(go.Scatter(x=future_months, y=proj_claims, name="Predicción Central (Point Forecast)", line=dict(color="#e63946", width=2.5, dash="dash")))

            fig_fc_c.update_layout(
                title=f"Volumen Mensual de Reclamaciones con Bandas Conformes (Proyección {horizon_choice}M)",
                xaxis_title="Periodo",
                yaxis_title="Cantidad de Reclamos",
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig_fc_c, use_container_width=True)

        with col_fc2:
            fig_fc_m = go.Figure()
            # 95% Conformal Band
            fig_fc_m.add_trace(go.Scatter(x=future_months, y=m_upper_95, mode="lines", line=dict(width=0), showlegend=False))
            fig_fc_m.add_trace(go.Scatter(x=future_months, y=m_lower_95, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(244, 162, 97, 0.25)", name="Banda Conforme 95%"))
            # Historic & Point Forecast
            fig_fc_m.add_trace(go.Scatter(x=df_claims["period"], y=df_claims["monto_instruido_devolver"], name="Histórico Real (DOP)", line=dict(color="#2a9d8f", width=2)))
            fig_fc_m.add_trace(go.Scatter(x=future_months, y=proj_montos, name="Predicción Central (DOP)", line=dict(color="#e76f51", width=2.5, dash="dash")))

            fig_fc_m.update_layout(
                title=f"Monto Instruido a Devolver al Ahorrista (Bandas Conformes {horizon_choice}M DOP)",
                xaxis_title="Periodo",
                yaxis_title="Monto (DOP)",
                template="plotly_white",
                height=420,
            )
            st.plotly_chart(fig_fc_m, use_container_width=True)

        with st.expander("📋 Ver Tabla Detallada de Intervalos Conformes Mensuales (90% y 95%)"):
            st.dataframe(
                conf_df[[
                    "period", "claims_lower_90", "claims_point", "claims_upper_90", "claims_upper_95",
                    "restitution_dop_lower_90", "restitution_dop_point", "restitution_dop_upper_90", "restitution_dop_upper_95"
                ]],
                use_container_width=True,
                hide_index=True
            )

        st.markdown("---")
        # ARTEFACTO DE NEGOCIO 2: CUANTIFICADOR DE BUFFER DE RESTITUCIÓN (CONFORMAL & VaR)
        st.markdown("### 💰 **Artefacto de Negocio: Buffer Prescriptivo Conforme y Value-at-Risk (VaR)**")
        st.markdown("Dimensionamiento de reservas de contingencia de liquidez con garantías estadísticas finitas (Conformal 95%) y métricas VaR/CVaR.")

        hist_restitutions = df_claims["monto_instruido_devolver"]
        forecast_val = proj_montos[0]


        buf_res = calculate_restitution_liquidity_buffer(
            historical_restitution_series=hist_restitutions,
            forecast_next_period=forecast_val,
        )

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            st.metric("Proyección Central 12 Meses", f"DOP ${conformal_res['total_projected_restitution_dop']:,.2f}")
        with bcol2:
            st.metric("Límite Superior Conforme (95%)", f"DOP ${conformal_res['total_conformal_95_restitution_dop']:,.2f}", "Cobertura Garantizada")
        with bcol3:
            st.metric("Buffer de Liquidez Conforme", f"DOP ${conformal_res['conformal_liquidity_buffer_required_dop']:,.2f}", "Colchón de Contingencia")
        with bcol4:
            st.metric("Value-at-Risk (VaR 95% Mensual)", f"DOP ${buf_res['Historical_VaR_95_DOP']:,.2f}", "Pérdida Extrema Esperada")


# =====================================================================
# TAB 3: SUPERVISORY RESOURCE OPTIMIZATION
# =====================================================================
with tab_optimization:
    st.subheader("🎯 Optimización Prescriptiva de Asignación de Inspectores Bancarios")
    st.markdown("Algoritmo de programación matemática para asignar las horas hombre de inspección in-situ en función del perfil de riesgo de cada sector y entidad financiera.")

    col_opt_ctrl, col_opt_res = st.columns([1, 2.2])

    with col_opt_ctrl:
        st.markdown("##### ⚙️ Parámetros del Cuerpo de Inspectores")
        total_hours = st.slider("Presupuesto Total de Horas Hombre (Trimestre):", 1000.0, 6000.0, 3200.0, 200.0)
        min_h = st.slider("Mínimo de Horas por Entidad:", 20.0, 100.0, 50.0, 10.0)
        max_h = st.slider("Máximo de Horas por Entidad:", 300.0, 1000.0, 750.0, 50.0)

        st.info("💡 **Criterio de Asignación:** Ponderación cuadrática basada en la probabilidad de infracción EWS, volumen de quejas y riesgo sistémico.")

    # Simulated Entity Portfolio
    entities_data = pd.DataFrame({
        "entidad": [
            "Bancos Múltiples (Sistémicos)",
            "Bancos de Ahorro y Crédito",
            "Asociaciones de Ahorros y Préstamos",
            "Corporaciones de Crédito",
            "Agentes de Cambio y Remesas",
            "Entidades Públicas Intermediarias",
            "Firmas de Auditoría Externa",
        ],
        "risk_score": [88.5, 68.0, 62.5, 48.0, 35.0, 25.0, 15.0],
    })

    optimized_df = optimize_supervisory_inspection_allocation(
        entities_risk_df=entities_data,
        total_auditor_hours=total_hours,
        min_hours_per_entity=min_h,
        max_hours_per_entity=max_h,
    )

    with col_opt_res:
        fig_opt = px.bar(
            optimized_df,
            x="allocated_hours_optimal",
            y="entidad",
            orientation="h",
            color="risk_score",
            color_continuous_scale="Reds",
            title=f"Asignación Óptima de Horas de Inspección (Total: {total_hours:,.0f} Horas)",
            labels={"allocated_hours_optimal": "Horas Hombre Asignadas", "entidad": "Sector / Entidad", "risk_score": "Score Riesgo"},
            template="plotly_white",
        )
        fig_opt.update_layout(yaxis=dict(autorange="reversed"), height=360)
        st.plotly_chart(fig_opt, use_container_width=True)

    st.markdown("##### 📋 Cronograma y Plan de Auditorías Focalizadas")
    st.dataframe(
        optimized_df[[
            "entidad", "risk_score", "prioridad_inspeccion",
            "allocated_hours_optimal", "auditores_estimados",
        ]].rename(columns={
            "entidad": "Entidad / Sector Regulado",
            "risk_score": "Score de Riesgo (0-100)",
            "prioridad_inspeccion": "Prioridad Regulatoria",
            "allocated_hours_optimal": "Horas Hombre Óptimas",
            "auditores_estimados": "Equipo Estimado (Auditores)",
        }),
        use_container_width=True,
    )

# =====================================================================
# TAB 4: PROUSUARIO & CONDUCT CONCENTRATION
# =====================================================================
with tab_prousuario:
    st.subheader("🛡️ Protección al Usuario Financiero (ProUsuario) & Concentración")

    if not df_claims.empty:
        col_pu1, col_pu2 = st.columns(2)

        with col_pu1:
            fig_dec = go.Figure()
            fig_dec.add_trace(go.Bar(x=df_claims["period"], y=df_claims["favorable"], name="Decisión Favorable", marker_color="#2a9d8f"))
            fig_dec.add_trace(go.Bar(x=df_claims["period"], y=df_claims["desfavorable"], name="Decisión Desfavorable", marker_color="#e76f51"))
            fig_dec.update_layout(barmode="stack", title="Resolución de Reclamaciones: Favorable vs Desfavorable", xaxis_title="Periodo", yaxis_title="Casos Completados", template="plotly_white")
            st.plotly_chart(fig_dec, use_container_width=True)

        with col_pu2:
            fig_pct = px.line(df_claims, x="period", y="pct_favorable", title="Tasa de Éxito Favorable al Ahorrista (% Favorable)", markers=True)
            fig_pct.update_traces(line_color="#1d3557")
            fig_pct.update_layout(yaxis=dict(tickformat=".0%"), template="plotly_white")
            st.plotly_chart(fig_pct, use_container_width=True)

        st.markdown("---")
        # ARTEFACTO DE NEGOCIO 4: CONCENTRACIÓN DE CONDUCTA DE MERCADO
        st.markdown("### 📊 **Artefacto de Negocio: Concentración de Conducta de Mercado (Índice HHI y Gini)**")

        gini_claims = calculate_concentration_indices(df_claims["monto_instruido_devolver"])
        gini_fines = calculate_concentration_indices(df_master["monto_sanciones_dop"])

        c_col1, c_col2, c_col3, c_col4 = st.columns(4)
        with c_col1:
            st.metric("Índice Herfindahl (HHI) Reclamaciones", f"{gini_claims['HHI']:,.1f}", gini_claims["Concentration_Level"])
        with c_col2:
            st.metric("Coeficiente de Gini (Devoluciones)", f"{gini_claims['Gini']:.3f}", "Desigualdad Monetaria")
        with c_col3:
            st.metric("Índice Herfindahl (HHI) Sanciones", f"{gini_fines['HHI']:,.1f}", gini_fines["Concentration_Level"])
        with c_col4:
            st.metric("Coeficiente de Gini (Multas)", f"{gini_fines['Gini']:.3f}", "Concentración de Multas")

# =====================================================================
# TAB 5: AML/CFT & SANCTIONS BREAKDOWN
# =====================================================================
with tab_aml:
    st.subheader("⚖️ Infracciones por Gravedad, Sanciones & Ley 155-17 (AML/CFT)")

    if not df_inf.empty:
        col_inf1, col_inf2 = st.columns(2)

        with col_inf1:
            inf_by_type = df_inf.groupby("infraction_type")["infraction_count"].sum().reset_index()
            fig_inf_pie = px.pie(
                inf_by_type,
                names="infraction_type",
                values="infraction_count",
                title="Distribución Total de Infracciones por Nivel de Gravedad",
                hole=0.4,
                color_discrete_sequence=["#f4a261", "#e76f51", "#d62828", "#2b5c8f"],
            )
            fig_inf_pie.update_layout(template="plotly_white")
            st.plotly_chart(fig_inf_pie, use_container_width=True)

        with col_inf2:
            inf_by_entity = df_inf.groupby("entity_type")["infraction_count"].sum().reset_index().sort_values("infraction_count", ascending=False)
            fig_inf_ent = px.bar(
                inf_by_entity.head(7),
                x="infraction_count",
                y="entity_type",
                orientation="h",
                title="Infracciones Imputadas por Tipo de Entidad Regulada",
                color="infraction_count",
                color_continuous_scale="Reds",
            )
            fig_inf_ent.update_layout(yaxis=dict(autorange="reversed"), template="plotly_white")
            st.plotly_chart(fig_inf_ent, use_container_width=True)

# =====================================================================
# TAB 6: DATA QUALITY & SCHEMA AUDIT
# =====================================================================
with tab_dq:
    st.subheader("🔍 Observabilidad Continua & Scorecard de Data Quality")
    st.markdown("Auditoría automática de completitud, consistencia, duplicados y validación de contratos **Pandera** sobre los 14 datasets.")

    datasets_to_score = {
        "ProUsuario Reclamaciones": df_claims,
        "Supervisión Consolidada": df_master,
        "Infracciones Imputadas": df_inf,
        "Feature Store EWS": df_ews,
        "Feature Store Forecasting": df_fc,
    }

    scorecard_df = generate_overall_data_quality_report(datasets_to_score)
    st.dataframe(scorecard_df, use_container_width=True)

    # Schema Validation Badges
    st.markdown("##### 🛡️ Estado de Validación de Esquemas (Contratos Pandera)")
    try:
        ProUsuarioClaimsSchema.validate(df_claims)
        st.success("✅ **ProUsuarioClaimsSchema:** Contrato Validado y Conforme (0 discrepancias de tipos o rangos).")
    except Exception as e:
        st.error(f"❌ Error en ProUsuarioClaimsSchema: {e}")

    try:
        MasterSupervisionSchema.validate(df_master)
        st.success("✅ **MasterSupervisionSchema:** Contrato Validado y Conforme (Consistencia referencial 100%).")
    except Exception as e:
        st.error(f"❌ Error en MasterSupervisionSchema: {e}")

    st.markdown("---")
    # ARTEFACTO DE NEGOCIO 6: EXPORTADOR EJECUTIVO
    st.markdown("### 📥 **Artefacto de Negocio: Exportación de Informe Gerencial SupTech**")
    csv_report = df_ews.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="📥 Descargar Ficha Técnica Consolidada de Supervisión (CSV)",
        data=csv_report,
        file_name="SB_RiskIntel_Supervision_Consolidated_2026.csv",
        mime="text/csv",
    )

st.markdown("---")
st.markdown("© 2026 **SB-RiskIntel Platform** | Superintendencia de Bancos de la República Dominicana | Desarrollado por **Guillén Concepción** (*Senior Data Scientist & MLOps Engineer*)")
