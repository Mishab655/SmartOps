from sqlalchemy import create_engine
from backend.core.config import DB_URI

engine = create_engine(DB_URI)

def get_db_engine():
    """Returns a standalone engine for backwards compatibility (models, decision_engine)"""
    return create_engine(DB_URI)

def get_engine():
    """Returns the shared engine for chatbot"""
    return engine
