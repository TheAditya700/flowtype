import sys
import os
from sqlalchemy.orm import sessionmaker

# Add backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import engine
from app.models.db_models import Snippet

def check_snippet_difficulty():
    Session = sessionmaker(bind=engine)
    session = Session()
    
    # ID from your log
    target_id = "ec82dc40-8581-4712-a8b6-454523c35459"
    
    try:
        snippet = session.query(Snippet).get(target_id)
        if snippet:
            print(f"Snippet {target_id}:")
            print(f"  Difficulty Score: {snippet.difficulty_score}")
            print(f"  Processed Embedding: {'Yes' if snippet.processed_embedding else 'No'}")
            print(f"  Normalized Features: {'Yes' if snippet.normalized_features else 'No'}")
        else:
            print(f"Snippet {target_id} not found in DB.")
            
        # Check overall stats
        total = session.query(Snippet).count()
        with_diff = session.query(Snippet).filter(Snippet.difficulty_score.isnot(None)).count()
        print(f"\nTotal Snippets: {total}")
        print(f"Snippets with Difficulty: {with_diff}")
        
    finally:
        session.close()

if __name__ == "__main__":
    check_snippet_difficulty()
