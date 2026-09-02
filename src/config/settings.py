"""Project settings and configuration management."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models_registry"
MLRUNS_DIR = PROJECT_ROOT / "mlruns"


class Settings(BaseSettings):
    PROJECT_NAME: str = "SB-RiskIntel"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "production"

    # Storage Paths
    RAW_DATA_PATH: Path = RAW_DATA_DIR
    PROCESSED_DATA_PATH: Path = PROCESSED_DATA_DIR
    MODELS_PATH: Path = MODELS_DIR

    # MLflow
    MLFLOW_TRACKING_URI: str = str(MLRUNS_DIR)
    MLFLOW_EXPERIMENT_NAME: str = "SB-RiskIntel-Supervision"

    # API
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # SB Portal Base URLs
    SB_OPEN_DATA_BASE_URL: str = "https://sb.gob.do/transparencia/datos-abiertos"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
