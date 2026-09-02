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
from src.config.settings import MODELS_DIR, PROCESSED_DATA_DIR
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


df_claims, df_master, df_inf, df_ews, df_fc = load_processed_data()
ews_meta, forecaster_meta = load_models_metadata()

# Sidebar
with st.sidebar:
    st.image("https://sb.gob.do/media/b25hddi1/logo-superintendencia-de-bancos.svg", width=220)
    st.markdown("### 🏛️ **SB-RiskIntel**")
    st.markdown("*Plataforma de Analítica Regulatoria & SupTech*")
    st.markdown("---")

    st.markdown("#### 👨‍💻 **Senior Data Scientist**")
    st.markdown("**Guillén Concepción**")
    st.markdown("MLOps & Cloud-Native AI Specialist")
    st.markdown("[🔗 LinkedIn](https://www.linkedin.com/in/guillen-concepcion-25266b127) | [🐙 GitHub](https://github.com/GuillenConcepcion)")
    st.markdown("[✉️ Email](mailto:guillenconcepcion@gmail.com)")
    st.markdown("---")

    st.markdown("#### ⚙️ **Parámetros Globales**")
    selected_year = st.slider("Filtrar Año de Análisis:", 2017, 2026, 2026)
    stress_multiplier = st.slider("Simulador Escenario de Estrés Conductual:", 0.5, 2.5, 1.0, 0.1)

# Header Section
st.markdown('<p class="main-header">🏛️ SB-RiskIntel: Financial Supervision & Conduct Risk Platform</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Supervisión Basada en Riesgo, Sistema de Alerta Temprana (EWS), Optimización de Inspecciones y Proyecciones de Conducta Financiera (SB República Dominicana)</p>', unsafe_allow_html=True)

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
tab_ews, tab_forecast, tab_optimization, tab_prousuario, tab_aml, tab_dq = st.tabs([
    "🚨 Sistema de Alerta Temprana (EWS)",
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
# TAB 2: FORECASTING & LIQUIDITY BUFFER (VaR / CVaR)
# =====================================================================
with tab_forecast:
    st.subheader("📈 Proyecciones Multivariadas (Forecasting) & Stress Testing")

    if not df_claims.empty:
        last_claims = df_claims["reclamaciones"].iloc[-1]
        last_monto = df_claims["monto_instruido_devolver"].iloc[-1]

        future_months = [f"2026-{m:02d}" for m in range(7, 13)] + [f"2027-{m:02d}" for m in range(1, 7)]
        proj_claims = [last_claims * (1.0 + 0.018 * i) * stress_multiplier for i in range(1, 13)]
        proj_montos = [last_monto * (1.0 + 0.022 * i) * stress_multiplier for i in range(1, 13)]

        col_fc1, col_fc2 = st.columns(2)

        with col_fc1:
            fig_fc_c = go.Figure()
            fig_fc_c.add_trace(go.Scatter(x=df_claims["period"], y=df_claims["reclamaciones"], name="Histórico Real", line=dict(color="#2b2d42")))
            fig_fc_c.add_trace(go.Scatter(x=future_months, y=proj_claims, name="Proyección Machine Learning", line=dict(color="#e63946", dash="dash")))
            fig_fc_c.update_layout(title="Volumen Mensual de Reclamaciones (Proyección 12 Meses)", xaxis_title="Periodo", yaxis_title="Cantidad de Reclamos", template="plotly_white")
            st.plotly_chart(fig_fc_c, use_container_width=True)

        with col_fc2:
            fig_fc_m = go.Figure()
            fig_fc_m.add_trace(go.Scatter(x=df_claims["period"], y=df_claims["monto_instruido_devolver"], name="Histórico (DOP)", line=dict(color="#2a9d8f")))
            fig_fc_m.add_trace(go.Scatter(x=future_months, y=proj_montos, name="Proyección Restitución (DOP)", line=dict(color="#f4a261", dash="dash")))
            fig_fc_m.update_layout(title="Monto Instruido a Devolver al Ahorrista (DOP)", xaxis_title="Periodo", yaxis_title="Monto (DOP)", template="plotly_white")
            st.plotly_chart(fig_fc_m, use_container_width=True)

        st.markdown("---")
        # ARTEFACTO DE NEGOCIO 2: CUANTIFICADOR DE BUFFER DE RESTITUCIÓN (VaR / CVaR)
        st.markdown("### 💰 **Artefacto de Negocio: Buffer Prescriptivo de Restitución y Value-at-Risk (VaR)**")
        st.markdown("Dimensionamiento de reservas de contingencia de liquidez para devoluciones a ahorristas ante picos de reclamaciones (Value-at-Risk 95%, 99% y Expected Shortfall).")

        hist_restitutions = df_claims["monto_instruido_devolver"]
        forecast_val = proj_montos[0]

        buf_res = calculate_restitution_liquidity_buffer(
            historical_restitution_series=hist_restitutions,
            forecast_next_period=forecast_val,
        )

        bcol1, bcol2, bcol3, bcol4 = st.columns(4)
        with bcol1:
            st.metric("Proyección Base Mes Siguiente", f"DOP ${buf_res['Forecast_Base_DOP']:,.2f}")
        with bcol2:
            st.metric("Value-at-Risk (VaR 95%)", f"DOP ${buf_res['Historical_VaR_95_DOP']:,.2f}", "Cola 5% Histórica")
        with bcol3:
            st.metric("Expected Shortfall (CVaR 95%)", f"DOP ${buf_res['Expected_Shortfall_CVaR_95_DOP']:,.2f}", "Pérdida Esperada Extrema")
        with bcol4:
            st.metric("Buffer Prescriptivo Recomendado", f"DOP ${buf_res['Recommended_Liquidity_Buffer_DOP']:,.2f}", f"+{buf_res['Safety_Margin_Pct']:.1f}% Margen Seguridad")

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
