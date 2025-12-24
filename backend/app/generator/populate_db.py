# app/generator/populate_snippets.py

import json
from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import Snippet
import logging
from . import config
from app.utils.s3_data import read_json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


SNIPPET_BUCKET = config.OFFLINE_BUCKET
SNIPPET_KEY = config.SNIPPETS_KEY


def populate_snippet_database(clear_existing=True):
    """
    Loads generated snippets.json and inserts them into the Snippet DB.
    """

    logger.info("Loading snippets.json from S3 ...")
    try:
        snippets = read_json(SNIPPET_BUCKET, SNIPPET_KEY, env="offline")
    except Exception as e:
        logger.error(f"Failed to read snippets from S3: {e}")
        return

    Session = sessionmaker(bind=engine)
    session = Session()

    if clear_existing:
        logger.info("Clearing existing snippet table ...")
        session.query(Snippet).delete()
        session.commit()

    db_objects = []

    for s in snippets:
        obj = Snippet(
            text=s["text"],
            words=s["words"],
            word_count=len(s["words"]),
            features=s["features"],
            difficulty_score=None,  # Will be learned later
        )

        db_objects.append(obj)

    logger.info(f"Inserting {len(db_objects)} snippets into database ...")

    try:
        session.bulk_save_objects(db_objects)
        session.commit()
        logger.info("✔ Snippet table populated successfully.")

    except Exception as e:
        session.rollback()
        logger.error(f"Failed to populate snippets: {e}")

    finally:
        session.close()


if __name__ == "__main__":
    populate_snippet_database()
