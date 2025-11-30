from sqlalchemy.orm import sessionmaker
from app.database import engine
from app.models.db_models import User
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def seed():
    """
    Seeds the database with sample data.
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    # Create a sample user
    if not session.query(User).first():
        user = User()
        session.add(user)
        session.commit()
        logger.info(f"Created sample user with ID: {user.id}")
    else:
        logger.info("User already exists, skipping seeding.")
        
    session.close()

if __name__ == "__main__":
    seed()
