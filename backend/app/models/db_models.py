import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Index, JSON, UUID, TypeDecorator
from sqlalchemy.sql import func
from app.database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type that uses CHAR(32) on SQLite and UUID on PostgreSQL."""
    impl = String(36)
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return str(value)
        return str(value) if value else None

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        if isinstance(value, uuid.UUID):
            return value
        try:
            return uuid.UUID(value)
        except (ValueError, AttributeError):
            # If it's not a valid UUID, return as-is (e.g., user-001, sess-001)
            return value

class User(Base):
    __tablename__ = "users"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now(), onupdate=func.now())

class Snippet(Base):
    __tablename__ = "snippets"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    words = Column(String, nullable=False)
    word_count = Column(Integer, nullable=False)
    difficulty_score = Column(Float, nullable=False)
    avg_word_length = Column(Float)
    punctuation_density = Column(Float)
    rare_letter_count = Column(Integer)
    bigram_rarity = Column(Float)
    created_at = Column(DateTime, default=func.now())
    
    __table_args__ = (
        Index('idx_difficulty', 'difficulty_score'),
    )

class TypingSession(Base):
    __tablename__ = "typing_sessions"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    started_at = Column(DateTime, default=func.now())
    duration_seconds = Column(Float, nullable=False)
    words_typed = Column(Integer, nullable=False)
    characters_typed = Column(Integer, nullable=False)
    errors = Column(Integer, nullable=False)
    backspaces = Column(Integer, nullable=False)
    final_wpm = Column(Float, nullable=False)
    avg_wpm = Column(Float, nullable=False)
    peak_wpm = Column(Float, nullable=False)
    accuracy = Column(Float, nullable=False)
    starting_difficulty = Column(Float, nullable=False)
    ending_difficulty = Column(Float, nullable=False)
    avg_difficulty = Column(Float, nullable=False)
    keystroke_events = Column(JSON)
    flow_score = Column(Float)

class SnippetUsage(Base):
    __tablename__ = "snippet_usage"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    session_id = Column(GUID(), ForeignKey("typing_sessions.id"))
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    snippet_id = Column(GUID(), ForeignKey("snippets.id"))
    started_at = Column(DateTime, default=func.now())
    completed_at = Column(DateTime)
    user_wpm = Column(Float)
    user_accuracy = Column(Float)
    snippet_position = Column(Integer)
    difficulty_snapshot = Column(Float)


class TelemetrySnippetRaw(Base):
    __tablename__ = "telemetry_snippet_raw"
    id = Column(GUID(), primary_key=True, default=uuid.uuid4)
    received_at = Column(DateTime, default=func.now())
    # store the full raw payload for later processing
    payload = Column(JSON, nullable=False)
    user_id = Column(GUID(), ForeignKey("users.id"), nullable=True)
    session_id = Column(GUID(), ForeignKey("typing_sessions.id"), nullable=True)
    source = Column(String, nullable=True)
