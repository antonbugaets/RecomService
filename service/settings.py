from functools import lru_cache
from typing import List

from pydantic import BaseSettings


class Config(BaseSettings):

    class Config:
        case_sensitive = False
        env_file = ".env.example", ".env"


class LogConfig(Config):
    level: str = "INFO"
    datetime_format: str = "%Y-%m-%d %H:%M:%S"

    class Config:
        case_sensitive = False
        fields = {
            "level": {
                "env": ["log_level"]
            },
        }


class ServiceConfig(Config):
    service_name: str = "reco_service"
    k_recs: int = 10
    api_key: str
    model_dir: str = "models"
    models_to_load: List[str] = ["user_knn"]

    log_config: LogConfig


@lru_cache
def get_config() -> ServiceConfig:
    return ServiceConfig(
        log_config=LogConfig(),
    )
