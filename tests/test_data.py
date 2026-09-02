"""Unit tests for Data Engineering & ETL components."""

import pandas as pd

from src.config.settings import PROCESSED_DATA_DIR
from src.data.validators import MasterSupervisionSchema, ProUsuarioClaimsSchema


def test_processed_claims_data():
    """Verify cleaned ProUsuario claims dataset schema and values."""
    claims_path = PROCESSED_DATA_DIR / "prousuario_reclamaciones_cleaned.parquet"
    assert claims_path.exists(), "Processed claims parquet file does not exist."

    df = pd.read_parquet(claims_path)
    assert len(df) > 0, "Processed claims dataset is empty."
    assert "reclamaciones" in df.columns
    assert "monto_instruido_devolver" in df.columns
    assert df["pct_favorable"].between(0.0, 1.0).all()

    # Pandera validation
    validated_df = ProUsuarioClaimsSchema.validate(df)
    assert len(validated_df) == len(df)


def test_processed_master_supervision_data():
    """Verify master consolidated supervision dataset."""
    master_path = PROCESSED_DATA_DIR / "supervision_consolidated_quarterly.parquet"
    assert master_path.exists(), "Processed master supervision parquet file does not exist."

    df = pd.read_parquet(master_path)
    assert len(df) >= 30, "Master quarterly dataset has fewer periods than expected."
    assert "total_sanciones_impuestas" in df.columns
    assert "total_infracciones_imputadas" in df.columns

    validated_df = MasterSupervisionSchema.validate(df)
    assert len(validated_df) == len(df)
