import asyncio
from app.database import engine, Base
from app.models.db_models import User, Snippet, TypingSession, SnippetUsage

async def init_models():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)

if __name__ == "__main__":
    print("Initializing database...")
    # In Python 3.8+ and SQLAlchemy 1.4+, you can use asyncio.run()
    # For simplicity in this script, we'll use the synchronous API for creation
    Base.metadata.create_all(bind=engine)
    print("Database initialized.")
