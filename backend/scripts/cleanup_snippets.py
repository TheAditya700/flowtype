import sys
import os

# Add parent directory to path to allow importing app modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.database import SessionLocal
from app.models.db_models import Snippet
from sqlalchemy import func

def cleanup():
    db = SessionLocal()
    try:
        # Find snippets with length > 60
        query = db.query(Snippet).filter(func.length(Snippet.text) > 60)
        count = query.count()
        print(f"Found {count} snippets with length > 60.")
        
        if count > 0:
            deleted = query.delete(synchronize_session=False)
            db.commit()
            print(f"Deleted {deleted} snippets.")
        else:
            print("No snippets to delete.")
            
    except Exception as e:
        print(f"Error: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    cleanup()
