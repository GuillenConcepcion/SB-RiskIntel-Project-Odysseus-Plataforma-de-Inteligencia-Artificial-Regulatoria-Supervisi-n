"""ETL and Data Cleaning pipeline for SB Open Datasets."""

import logging
import os
from pathlib import Path

os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

import numpy as np
import pandas as pd

from src.config.settings import PROCESSED_DATA_DIR, RAW_DATA_DIR
from src.data.validators import MasterSupervisionSchema, ProUsuarioClaimsSchema

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

SPANISH_MONTHS = {
    "ene": 1, "enero": 1,
    "feb": 2, "febrero": 2,
    "mar": 3, "marzo": 3,
    "abr": 4, "abril": 4,
    "may": 5, "mayo": 5,
    "jun": 6, "junio": 6,
    "jul": 7, "julio": 7,
    "ago": 8, "agosto": 8,
    "sep": 9, "sept": 9, "septiembre": 9, "setiembre": 9,
    "oct": 10, "octubre": 10,
    "nov": 11, "noviembre": 11,
    "dic": 12, "diciembre": 12,
}

QUARTER_MAP = {
    "ene-mar": 1, "q1": 1, "trimestre 1": 1, "1": 1,
    "abr-jun": 2, "q2": 2, "trimestre 2": 2, "2": 2,
    "jul-sep": 3, "q3": 3, "trimestre 3": 3, "3": 3,
    "oct-dic": 4, "q4": 4, "trimestre 4": 4, "4": 4,
}


def clean_currency(val) -> float:
    """Convert monetary / numerical strings to float."""
    if pd.isna(val) or val is None:
        return 0.0
    if isinstance(val, (int, float)):
        return float(val)
    s = str(val).replace("RD$", "").replace("$", "").replace("DOP", "").replace(",", "").strip()
    try:
        return float(s)
    except ValueError:
        return 0.0


def clean_prousuario_reclamaciones(raw_path: Path) -> pd.DataFrame:
    """Clean ProUsuario claims monthly dataset."""
    logger.info("Cleaning ProUsuario claims dataset...")
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig", sep=None, engine="python")

    res = pd.DataFrame()
    res["year"] = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce").fillna(2020).astype(int)
    res["month_name"] = df_raw.iloc[:, 1].astype(str).str.strip().str.lower()
    res["month_num"] = res["month_name"].map(SPANISH_MONTHS).fillna(1).astype(int)
    res["period"] = res.apply(lambda r: f"{int(r['year']):04d}-{int(r['month_num']):02d}", axis=1)

    # Metrics
    res["casos_recibidos"] = pd.to_numeric(df_raw.iloc[:, 2].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)
    res["reclamaciones"] = pd.to_numeric(df_raw.iloc[:, 5].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)
    res["completadas"] = pd.to_numeric(df_raw.iloc[:, 7].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)
    res["favorable"] = pd.to_numeric(df_raw.iloc[:, 17].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)
    res["desfavorable"] = pd.to_numeric(df_raw.iloc[:, 20].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)

    # % Favorable
    pct_fav = pd.to_numeric(df_raw.iloc[:, 23], errors="coerce").fillna(0.0)
    res["pct_favorable"] = np.where(pct_fav > 1.0, pct_fav / 100.0, pct_fav)

    # Monto instruido a devolver
    res["monto_instruido_devolver"] = df_raw.iloc[:, 27].apply(clean_currency)
    res["tiempo_respuesta_dias"] = pd.to_numeric(df_raw.iloc[:, 12], errors="coerce").fillna(0.0)

    # Filter valid range and sort
    res = res[(res["year"] >= 2020) & (res["year"] <= 2030)].sort_values(["year", "month_num"]).reset_index(drop=True)

    # Validate with Pandera
    res = ProUsuarioClaimsSchema.validate(res)
    return res


def clean_infracciones(raw_path: Path) -> pd.DataFrame:
    """Clean Infractions dataset."""
    logger.info("Cleaning Infracciones Imputadas dataset...")
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig", sep=None, engine="python")

    res = pd.DataFrame()
    res["year"] = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce").fillna(2017).astype(int)
    q_str = df_raw.iloc[:, 1].astype(str).str.strip().str.lower()
    res["quarter"] = q_str.map(QUARTER_MAP).fillna(1).astype(int)
    res["entity_type"] = df_raw.iloc[:, 2].astype(str).str.strip()
    res["infraction_type"] = df_raw.iloc[:, 3].astype(str).str.strip()
    res["infraction_count"] = pd.to_numeric(df_raw.iloc[:, 4], errors="coerce").fillna(0).astype(int)
    res["period"] = res.apply(lambda r: f"{r['year']}-Q{r['quarter']}", axis=1)

    return res.sort_values(["year", "quarter"]).reset_index(drop=True)


def clean_sanciones(raw_path: Path) -> pd.DataFrame:
    """Clean Sanciones dataset."""
    logger.info("Cleaning Sanciones Impuestas dataset...")
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig", sep=None, engine="python")

    res = pd.DataFrame()
    res["year"] = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce").fillna(2017).astype(int)
    q_str = df_raw.iloc[:, 1].astype(str).str.strip().str.lower()
    res["quarter"] = q_str.map(QUARTER_MAP).fillna(1).astype(int)
    res["period"] = res.apply(lambda r: f"{r['year']}-Q{r['quarter']}", axis=1)
    res["total_sanciones"] = pd.to_numeric(df_raw.iloc[:, 2], errors="coerce").fillna(0).astype(int)
    res["monto_sanciones_dop"] = df_raw.iloc[:, 12].apply(clean_currency)

    return res.sort_values(["year", "quarter"]).reset_index(drop=True)


def clean_procesos_sancionadores(raw_path: Path) -> pd.DataFrame:
    """Clean Procesos Sancionadores dataset."""
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig", sep=None, engine="python")
    res = pd.DataFrame()
    res["year"] = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce").fillna(2017).astype(int)
    q_str = df_raw.iloc[:, 1].astype(str).str.strip().str.lower()
    res["quarter"] = q_str.map(QUARTER_MAP).fillna(1).astype(int)
    res["period"] = res.apply(lambda r: f"{r['year']}-Q{r['quarter']}", axis=1)
    res["total_procesos_sancionadores"] = pd.to_numeric(df_raw.iloc[:, 2], errors="coerce").fillna(0).astype(int)
    return res.sort_values(["year", "quarter"]).reset_index(drop=True)


def clean_planes_regularizacion(raw_path: Path) -> pd.DataFrame:
    """Clean Planes de Regularizacion dataset."""
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig", sep=None, engine="python")
    res = pd.DataFrame()
    res["year"] = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce").fillna(2017).astype(int)
    res["total_planes_regularizacion"] = pd.to_numeric(df_raw.iloc[:, 2], errors="coerce").fillna(0).astype(int)
    return res


def clean_aml_solicitudes(raw_path: Path) -> pd.DataFrame:
    """Clean AML/CFT Solicitudes Ley 155-17 dataset."""
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig", sep=None, engine="python")
    res = pd.DataFrame()
    res["year"] = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce").fillna(2017).astype(int)
    q_str = df_raw.iloc[:, 1].astype(str).str.strip().str.lower()
    res["quarter"] = q_str.map(QUARTER_MAP).fillna(1).astype(int)
    res["period"] = res.apply(lambda r: f"{r['year']}-Q{r['quarter']}", axis=1)
    res["total_solicitudes_aml"] = pd.to_numeric(df_raw.iloc[:, 2], errors="coerce").fillna(0).astype(int)
    return res.sort_values(["year", "quarter"]).reset_index(drop=True)


def clean_inspecciones_eif(raw_path: Path) -> pd.DataFrame:
    """Clean Inspecciones Realizadas dataset."""
    df_raw = pd.read_csv(raw_path, encoding="utf-8-sig", sep=None, engine="python")
    res = pd.DataFrame()
    res["year"] = pd.to_numeric(df_raw.iloc[:, 0], errors="coerce").fillna(2017).astype(int)
    res["total_inspecciones_eif"] = pd.to_numeric(df_raw.iloc[:, 2], errors="coerce").fillna(0).astype(int)
    return res


def build_consolidated_quarterly_dataset() -> pd.DataFrame:
    """Consolidate quarterly supervisory indicators across all sources into a single master time series."""
    logger.info("Building consolidated master quarterly dataset...")

    records = []
    for y in range(2017, 2027):
        for q in range(1, 5):
            if y == 2026 and q > 3:
                continue
            records.append({"year": y, "quarter": q, "period": f"{y}-Q{q}"})

    base_df = pd.DataFrame(records)

    # 1. Infracciones
    infracciones_raw = RAW_DATA_DIR / "infracciones_imputadas_2017_2026.csv"
    if infracciones_raw.exists():
        df_inf = clean_infracciones(infracciones_raw)
        inf_agg = df_inf.groupby(["year", "quarter"])["infraction_count"].sum().reset_index()
        inf_agg.rename(columns={"infraction_count": "total_infracciones_imputadas"}, inplace=True)
        base_df = base_df.merge(inf_agg, on=["year", "quarter"], how="left")
    else:
        base_df["total_infracciones_imputadas"] = 0

    # 2. Sanciones
    sanciones_raw = RAW_DATA_DIR / "sanciones_impuestas_2017_2026.csv"
    if sanciones_raw.exists():
        df_sanc = clean_sanciones(sanciones_raw)
        sanc_agg = df_sanc.groupby(["year", "quarter"]).agg({
            "total_sanciones": "sum",
            "monto_sanciones_dop": "sum"
        }).reset_index().rename(columns={"total_sanciones": "total_sanciones_impuestas"})
        base_df = base_df.merge(sanc_agg, on=["year", "quarter"], how="left")
    else:
        base_df["total_sanciones_impuestas"] = 0
        base_df["monto_sanciones_dop"] = 0.0

    # 3. Procesos Sancionadores
    proc_raw = RAW_DATA_DIR / "procesos_sancionadores_2017_2026.csv"
    if proc_raw.exists():
        df_proc = clean_procesos_sancionadores(proc_raw)
        proc_agg = df_proc.groupby(["year", "quarter"])["total_procesos_sancionadores"].sum().reset_index()
        base_df = base_df.merge(proc_agg, on=["year", "quarter"], how="left")
    else:
        base_df["total_procesos_sancionadores"] = 0

    # 4. AML Solicitudes
    aml_raw = RAW_DATA_DIR / "aml_solicitudes_ley155_2017_2026.csv"
    if aml_raw.exists():
        df_aml = clean_aml_solicitudes(aml_raw)
        aml_agg = df_aml.groupby(["year", "quarter"])["total_solicitudes_aml"].sum().reset_index()
        base_df = base_df.merge(aml_agg, on=["year", "quarter"], how="left")
    else:
        base_df["total_solicitudes_aml"] = 0

    # 5. Inspecciones EIF
    insp_raw = RAW_DATA_DIR / "inspecciones_eif_2017_2026.csv"
    if insp_raw.exists():
        df_insp = clean_inspecciones_eif(insp_raw)
        insp_by_year = df_insp.groupby("year")["total_inspecciones_eif"].sum().to_dict()
        base_df["total_inspecciones_eif"] = base_df["year"].map(insp_by_year).fillna(0) / 4.0
    else:
        base_df["total_inspecciones_eif"] = 0

    # 6. Planes Regularizacion
    plan_raw = RAW_DATA_DIR / "planes_regularizacion_2017_2026.csv"
    if plan_raw.exists():
        df_plan = clean_planes_regularizacion(plan_raw)
        plan_by_year = df_plan.groupby("year")["total_planes_regularizacion"].sum().to_dict()
        base_df["total_planes_regularizacion"] = base_df["year"].map(plan_by_year).fillna(0) / 4.0
    else:
        base_df["total_planes_regularizacion"] = 0

    # Fill NaNs
    base_df = base_df.fillna(0.0)

    # Validate Schema
    base_df = MasterSupervisionSchema.validate(base_df)
    return base_df


def run_full_etl(raw_dir: Path = RAW_DATA_DIR, out_dir: Path = PROCESSED_DATA_DIR):
    """Run full ETL pipeline and persist clean Parquet datasets."""
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Executing full ETL data cleaning pipeline...")

    # 1. ProUsuario Claims
    claims_raw = raw_dir / "prousuario_reclamaciones_2020_2026.csv"
    if claims_raw.exists():
        df_claims = clean_prousuario_reclamaciones(claims_raw)
        df_claims.to_parquet(out_dir / "prousuario_reclamaciones_cleaned.parquet", index=False)
        logger.info(f"Saved: {out_dir / 'prousuario_reclamaciones_cleaned.parquet'} ({len(df_claims)} rows)")

    # 2. Infracciones
    inf_raw = raw_dir / "infracciones_imputadas_2017_2026.csv"
    if inf_raw.exists():
        df_inf = clean_infracciones(inf_raw)
        df_inf.to_parquet(out_dir / "infracciones_imputadas_cleaned.parquet", index=False)
        logger.info(f"Saved: {out_dir / 'infracciones_imputadas_cleaned.parquet'} ({len(df_inf)} rows)")

    # 3. Master Consolidated Supervision
    df_master = build_consolidated_quarterly_dataset()
    df_master.to_parquet(out_dir / "supervision_consolidated_quarterly.parquet", index=False)
    logger.info(f"Saved: {out_dir / 'supervision_consolidated_quarterly.parquet'} ({len(df_master)} rows)")

    logger.info("ETL Pipeline completed successfully.")


if __name__ == "__main__":
    run_full_etl()
