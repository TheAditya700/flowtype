from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # Environment
    env: str = "dev"  # dev, stage, or prod
    
    # Database
    database_url: str = "postgresql://flowtype_dev:flowtype_dev@postgres_dev:5432/flowtype_dev"

    # FAISS
    faiss_index_path: str = "data/dev/faiss_index.bin"
    snippet_metadata_path: str = "data/dev/snippet_metadata.json"

    # ML
    embedding_dim: int = 16

    # API
    cors_origins: list[str] = ["*"]

    # Auth
    secret_key: str = "dev-secret-change-in-production"
    access_token_expire_minutes: int = 30  # Default to 30 minutes

    model_config = SettingsConfigDict(env_file=".env")


settings = Settings()
