from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str = "postgresql://user:password@db:5432/flowtype_db"
    
    # FAISS
    faiss_index_path: str = "data/faiss_index.bin"
    snippet_metadata_path: str = "data/snippet_metadata.json"
    
    # ML
    embedding_dim: int = 16
    
    # API
    cors_origins: list[str] = ["*"]

    # Auth
    secret_key: str
    access_token_expire_minutes: int = 30 # Default to 30 minutes
    
    class Config:
        env_file = ".env"

settings = Settings()
