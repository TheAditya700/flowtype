from pydantic_settings import BaseSettings, SettingsConfigDict
import os


class Settings(BaseSettings):
    # Environment
    env: str = "dev"  # dev, stage, or prod

    # Database
    database_url: str = (
        "postgresql://flowtype_dev:flowtype_dev@postgres_dev:5432/flowtype_dev"
    )

    # FAISS (S3-backed)
    # Object keys within the chosen data bucket for the current env
    faiss_index_key: str = "faiss_index.bin"
    snippet_metadata_key: str = "snippet_metadata.json"

    # ML
    embedding_dim: int = 16

    # API
    cors_origins: list[str] = ["*"]

    # Auth
    secret_key: str = "dev-secret-change-in-production"
    access_token_expire_minutes: int = 30  # Default to 30 minutes

    # Object storage (MinIO/S3 compatible)
    minio_endpoint: str = "http://minio-dev:9000"
    minio_dev_endpoint: str = "http://localhost:9000"
    minio_stage_endpoint: str = "http://localhost:9002"
    minio_prod_endpoint: str = "http://localhost:9004"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_region: str = "us-east-1"

    # Data buckets by environment (offline, dev, stage, prod)
    data_bucket_offline: str = "flowtype-offline"
    data_bucket_dev: str = "flowtype-dev"
    data_bucket_stage: str = "flowtype-stage"
    data_bucket_prod: str = "flowtype-prod"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    def model_post_init(self, __context):
        legacy_bucket = os.getenv("MINIO_BUCKET")
        if legacy_bucket:
            self.data_bucket_dev = legacy_bucket


settings = Settings()
