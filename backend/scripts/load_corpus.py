import json
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import Snippet
from app.ml.difficulty import calculate_difficulty
from app.utils.preprocessing import clean_text
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
CORPUS_FILE = DATA_DIR / "google-10000-english.txt"
SNIPPET_LENGTH = 7 # Number of words per snippet

def create_snippets_from_corpus():
    """
    Reads the corpus, creates snippets, calculates difficulty, and saves to DB.
    """
    if not CORPUS_FILE.exists():
        logger.error(f"Corpus file not found at {CORPUS_FILE}")
        return

    with open(CORPUS_FILE, 'r') as f:
        words = [line.strip() for line in f.readlines()]

    Session = sessionmaker(bind=engine)
    session = Session()
    
    # Clear existing snippets
    session.query(Snippet).delete()
    
    snippets_to_add = []
    for i in range(0, len(words) - SNIPPET_LENGTH + 1, SNIPPET_LENGTH):
        snippet_words = words[i:i+SNIPPET_LENGTH]
        snippet_text = " ".join(snippet_words)
        cleaned_text = clean_text(snippet_text)
        
        difficulty_features = calculate_difficulty(cleaned_text)
        
        new_snippet = Snippet(
            words=cleaned_text,
            word_count=len(snippet_words),
            **difficulty_features
        )
        snippets_to_add.append(new_snippet)

    try:
        session.bulk_save_objects(snippets_to_add)
        session.commit()
        logger.info(f"Successfully loaded {len(snippets_to_add)} snippets into the database.")
    except Exception as e:
        session.rollback()
        logger.error(f"Failed to load snippets: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    create_snippets_from_corpus()
