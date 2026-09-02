"""Data validation schemas using Pandera for SB-RiskIntel platform."""

import os

os.environ["DISABLE_PANDERA_IMPORT_WARNING"] = "True"

import pandera.pandas as pa
from pandera.typing import Series


class ProUsuarioClaimsSchema(pa.DataFrameModel):
    """Validation schema for cleaned ProUsuario monthly claims data."""
    year: Series[int] = pa.Field(ge=2018, le=2030)
    month_name: Series[str] = pa.Field(nullable=False)
    month_num: Series[int] = pa.Field(ge=1, le=12)
    period: Series[str] = pa.Field(nullable=False)
    casos_recibidos: Series[int] = pa.Field(ge=0)
    reclamaciones: Series[int] = pa.Field(ge=0)
    completadas: Series[int] = pa.Field(ge=0)
    favorable: Series[int] = pa.Field(ge=0)
    desfavorable: Series[int] = pa.Field(ge=0)
    pct_favorable: Series[float] = pa.Field(ge=0.0, le=1.0)
    monto_instruido_devolver: Series[float] = pa.Field(ge=0.0)

    class Config:
        strict = False
        coerce = True


class MasterSupervisionSchema(pa.DataFrameModel):
    """Validation schema for consolidated quarterly supervisory dataset."""
    year: Series[int] = pa.Field(ge=2015, le=2030)
    quarter: Series[int] = pa.Field(ge=1, le=4)
    period: Series[str] = pa.Field(nullable=False)
    total_inspecciones_eif: Series[float] = pa.Field(ge=0)
    total_procesos_sancionadores: Series[float] = pa.Field(ge=0)
    total_sanciones_impuestas: Series[float] = pa.Field(ge=0)
    monto_sanciones_dop: Series[float] = pa.Field(ge=0.0)
    total_planes_regularizacion: Series[float] = pa.Field(ge=0)
    total_infracciones_imputadas: Series[float] = pa.Field(ge=0)
    total_solicitudes_aml: Series[float] = pa.Field(ge=0)

    class Config:
        strict = False
        coerce = True
