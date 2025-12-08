from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Database
    database_url: str
    
    # FAISS
    faiss_index_path: str = "data/faiss_index.bin"
    snippet_metadata_path: str = "data/snippet_metadata.json"
    
    # ML
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 30
    
    # API
    cors_origins: list[str] = ["*"]

    # Auth
    secret_key: str
    access_token_expire_minutes: int = 30 # Default to 30 minutes
    
    class Config:
        env_file = ".env"

settings = Settings()
