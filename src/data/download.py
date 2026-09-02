"""Data download module for Superintendencia de Bancos Open Datasets."""

import logging
from pathlib import Path
from urllib.error import URLError
from urllib.parse import quote, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from src.config.settings import RAW_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Canonical mapping of SB Open Datasets
SB_DATASETS_REGISTRY = {
    "prousuario_reclamaciones": {
        "title": "Reclamaciones atendidas por ProUsuario",
        "csv_url": "https://sb.gob.do/media/uj5nc3vn/estadísticas-de-reclamaciones-atendidas-por-prousuario-2020-2026.csv",
        "filename": "prousuario_reclamaciones_2020_2026.csv",
    },
    "prousuario_usuarios": {
        "title": "Usuarios atendidos por ProUsuario por canal",
        "csv_url": "https://sb.gob.do/media/u3apbvok/estadísticas-de-usuarios-atendidos-por-prousuario-2019-2026.csv",
        "filename": "prousuario_usuarios_2019_2026.csv",
    },
    "portal_311_casos": {
        "title": "Casos recibidos a través del Portal 311",
        "csv_url": "https://sb.gob.do/media/iavcmtlg/estadísticas-de-casos-recibidos-a-través-del-portal-311-2018-2026.csv",
        "filename": "portal_311_casos_2018_2026.csv",
    },
    "entidades_autorizadas": {
        "title": "Listado de Entidades Autorizadas a Operar",
        "csv_url": "https://sb.gob.do/media/4g4nrdxa/listado-de-entidades-autorizadas-a-operar-2018-2026.csv",
        "filename": "entidades_autorizadas_2018_2026.csv",
    },
    "infracciones_imputadas": {
        "title": "Infracciones Imputadas por la SB",
        "csv_url": "https://sb.gob.do/media/wxsf42ys/estadísticas-de-infracciones-imputadas-2017-2026.csv",
        "filename": "infracciones_imputadas_2017_2026.csv",
    },
    "sanciones_impuestas": {
        "title": "Sanciones Impuestas por la SB",
        "csv_url": "https://sb.gob.do/media/iy1crbjw/estadísticas-de-sanciones-impuestas-2017-2026.csv",
        "filename": "sanciones_impuestas_2017_2026.csv",
    },
    "procesos_sancionadores": {
        "title": "Procesos Sancionadores Iniciados",
        "csv_url": "https://sb.gob.do/media/b4lfst1a/estadísticas-de-procesos-sancionadores-iniciados-2017-2026.csv",
        "filename": "procesos_sancionadores_2017_2026.csv",
    },
    "planes_regularizacion": {
        "title": "Planes de Regularización Requeridos a EIF",
        "csv_url": "https://sb.gob.do/media/mcqoiobj/estadísticas-de-planes-de-regularización-requeridos-a-eif-2017-2026.csv",
        "filename": "planes_regularizacion_2017_2026.csv",
    },
    "inspecciones_eif": {
        "title": "Inspecciones Realizadas a EIF y Cambiarias",
        "csv_url": "https://sb.gob.do/media/fzvj0em5/estadísticas-de-inspecciones-realizadas-a-eifyc-2017-2026.csv",
        "filename": "inspecciones_eif_2017_2026.csv",
    },
    "aml_inspecciones_ley_155_17": {
        "title": "Inspecciones Ley 155-17 (AML/CFT)",
        "csv_url": "https://sb.gob.do/media/f4mhzi0h/estadísticas-de-inspecciones-a-sujetos-obligados-en-materia-de-la-ley-155-17-2017-2026.csv",
        "filename": "aml_inspecciones_ley155_2017_2026.csv",
    },
    "aml_solicitudes_ley_155_17": {
        "title": "Solicitudes atendidas Ley 155-17 (AML/CFT)",
        "csv_url": "https://sb.gob.do/media/rhwfz5zq/estadísticas-de-solicitudes-atendidas-en-materia-de-la-ley-155-17-2017-2026.csv",
        "filename": "aml_solicitudes_ley155_2017_2026.csv",
    },
    "solicitudes_atendidas_eif": {
        "title": "Solicitudes atendidas a las EIF",
        "csv_url": "https://sb.gob.do/media/3ljfvdhr/estadísticas-de-solicitudes-atendidas-a-las-eif-2018-2026.csv",
        "filename": "solicitudes_atendidas_eif_2018_2026.csv",
    },
    "normativas_emitidas": {
        "title": "Normativas emitidas por la SB",
        "csv_url": "https://sb.gob.do/media/bgnfmzoc/estadísticas-de-normativas-emitidas-por-la-superintendencia-de-bancos-2003-2026.csv",
        "filename": "normativas_emitidas_2003_2026.csv",
    },
    "ifil_ahorristas_resarcidos": {
        "title": "Programa IFIL - Ahorristas Resarcidos y Devoluciones",
        "csv_url": "https://sb.gob.do/media/m1kjq2tw/estadísticas-del-programa-ifil-2005-2026.csv",
        "filename": "ifil_ahorristas_resarcidos_2005_2026.csv",
    },
}


def sanitize_url(url: str) -> str:
    """Properly encode unicode path components in URL."""
    scheme, netloc, path, query, fragment = urlsplit(url)
    path = quote(path)
    return urlunsplit((scheme, netloc, path, query, fragment))


def download_file(url: str, destination: Path) -> bool:
    """Download a file with User-Agent header and proper encoding handling."""
    encoded_url = sanitize_url(url)
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) SB-RiskIntel-Pipeline/1.0"
    }
    req = Request(encoded_url, headers=headers)
    try:
        with urlopen(req, timeout=30) as response:
            content = response.read()
            destination.write_bytes(content)
        logger.info(f"Downloaded: {destination.name} ({len(content)} bytes)")
        return True
    except URLError as e:
        logger.error(f"Failed to download from {url} ({encoded_url}): {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        return False


def download_all_datasets(dest_dir: Path = RAW_DATA_DIR) -> dict[str, Path]:
    """Download all registered Superintendencia de Bancos open datasets."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    downloaded = {}

    logger.info(f"Starting download of {len(SB_DATASETS_REGISTRY)} datasets into {dest_dir}...")
    for key, meta in SB_DATASETS_REGISTRY.items():
        dest_file = dest_dir / meta["filename"]
        success = download_file(meta["csv_url"], dest_file)
        if success and dest_file.exists():
            downloaded[key] = dest_file

    logger.info(f"Successfully downloaded {len(downloaded)} / {len(SB_DATASETS_REGISTRY)} datasets.")
    return downloaded


if __name__ == "__main__":
    download_all_datasets()
