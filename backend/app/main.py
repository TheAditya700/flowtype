from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.database import engine, Base
from app.routers import health, snippets, sessions, users
from app.routers import telemetry
from app.ml.vector_store import VectorStore
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="FlowType API",
    description="Adaptive typing practice with two-tower retrieval",
    version="0.1.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup: Initialize DB and load FAISS index
@app.on_event("startup")
async def startup_event():
    logger.info("Initializing database...")
    Base.metadata.create_all(bind=engine)
    
    logger.info("Loading FAISS index...")
    app.state.vector_store = VectorStore()
    logger.info("Startup complete!")

# Include routers
app.include_router(health.router, tags=["health"])
app.include_router(snippets.router, prefix="/api/snippets", tags=["snippets"])
app.include_router(sessions.router, prefix="/api/sessions", tags=["sessions"])
app.include_router(users.router, prefix="/api/users", tags=["users"])
app.include_router(telemetry.router, prefix="/api/telemetry", tags=["telemetry"])

@app.get("/")
def root():
    return {
        "message": "FlowType API",
        "version": "0.1.0",
        "docs": "/docs"
    }
