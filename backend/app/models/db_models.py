import uuid
from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey, Index, JSON, UUID
from sqlalchemy.sql import func
from app.database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    created_at = Column(DateTime, default=func.now())
    last_active = Column(DateTime, default=func.now(), onupdate=func.now())

class Snippet(Base):
    __tablename__ = "snippets"
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
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
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(UUID(as_uuid=True), ForeignKey("users.id"), nullable=True)
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
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(UUID(as_uuid=True), ForeignKey("typing_sessions.id"))
    snippet_id = Column(UUID(as_uuid=True), ForeignKey("snippets.id"))
    presented_at = Column(DateTime, default=func.now())
    user_wpm = Column(Float)
    user_accuracy = Column(Float)
    snippet_position = Column(Integer)
