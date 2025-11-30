import json
from pathlib import Path
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import Snippet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).resolve().parent / "data"
SNIPPET_FILE = DATA_DIR / "snippets.json"


def populate_snippet_database(clear_existing=True):
    """
    Loads generated snippets.json and inserts them into the Snippet DB table.
    """

    if not SNIPPET_FILE.exists():
        logger.error(f"snippets.json not found at {SNIPPET_FILE}")
        return

    logger.info("Loading snippets.json ...")
    snippets = json.loads(SNIPPET_FILE.read_text())

    Session = sessionmaker(bind=engine)
    session = Session()

    if clear_existing:
        logger.info("Clearing existing snippet table ...")
        session.query(Snippet).delete()
        session.commit()

    db_objects = []

    for s in snippets:
        text = s["text"]
        words = s["words"]
        features = s["features"]

        obj = Snippet(
            text=text,
            words=words,
            word_count=len(words),
            features=features,
            difficulty_score=None  # will be learned later
        )

        db_objects.append(obj)

    logger.info(f"Inserting {len(db_objects)} snippets into database ...")
    try:
        session.bulk_save_objects(db_objects)
        session.commit()
        logger.info("Snippet table populated successfully.")

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to populate snippets: {e}")

    finally:
        session.close()


if __name__ == "__main__":
    populate_snippet_database()
