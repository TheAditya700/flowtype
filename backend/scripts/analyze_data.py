import sys
import os
from sqlalchemy import func
from sqlalchemy.orm import sessionmaker
from collections import Counter

# Add backend to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.database import engine
from app.models.db_models import SnippetUsage, User, TypingSession, Snippet

def analyze_sessions():
    print("=== Analyzing Session Data ===")
    Session = sessionmaker(bind=engine)
    session = Session()

    try:
        # 1. Check Total Sessions
        total_sessions = session.query(TypingSession).count()
        print(f"Total Typing Sessions: {total_sessions}")

        # 2. Check Snippet Usage (Repeats)
        usages = session.query(SnippetUsage).order_by(SnippetUsage.created_at).all()
        print(f"Total Snippet Usages: {len(usages)}")
        
        if not usages:
            print("No usage data found.")
            return

        snippet_counts = Counter([u.snippet_id for u in usages])
        most_common = snippet_counts.most_common(5)
        print("\nMost Frequent Snippets (Overall):")
        for snip_id, count in most_common:
            print(f"  Snippet {snip_id}: {count} times")

        # Check for immediate repeats
        print("\nSequence of Snippets (Last 15):")
        last_15 = usages[-15:]
        for i, u in enumerate(last_15):
            snip = session.get(Snippet, u.snippet_id)
            text_preview = snip.text[:30] + "..." if snip else "Unknown"
            print(f"  {i+1}. [{u.created_at}] ID: {u.snippet_id} | Diff: {u.difficulty_snapshot:.2f} | WPM: {u.user_wpm:.1f} | Text: {text_preview}")
            
            if i > 0:
                prev_id = last_15[i-1].snippet_id
                if prev_id == u.snippet_id:
                    print("     WARNING: IMMEDIATE REPEAT DETECTED!")

        # 3. Check User Features
        # Dynamically find the most recent user ID from the last session
        user_id_to_check = "test_user_default"
        if last_15 and last_15[-1].session:
             user_id_to_check = last_15[-1].session.user_id or "test_user_default"
             
        user = session.get(User, user_id_to_check)
        if user and user.features:
            print(f"\nUser Stats for {user_id_to_check}:")
            feats = user.features
            print(f"  Session Count: {feats.get('session_count')}")
            print(f"  Short Term History Length: {len(feats.get('short_term_history', []))}")
            print(f"  WPM History (Last 5): {feats.get('wpm_history', [])[-5:]}")
            print(f"  Accuracy History (Last 5): {feats.get('accuracy_history', [])[-5:]}")
        else:
            print(f"\nUser features for '{user_id_to_check}' not found or empty.")
    
    except Exception as e:
        print(f"Error analyzing sessions: {e}")
    finally:
        session.close()

if __name__ == "__main__":
    analyze_sessions()
