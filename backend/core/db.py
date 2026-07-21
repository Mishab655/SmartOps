from sqlalchemy import create_engine
from backend.core.config import DB_URI

# Added connect_timeout to prevent infinite hanging when DB is unreachable (e.g. paused or IPv6 blocked)
engine = create_engine(DB_URI, connect_args={"connect_timeout": 10})

def get_db_engine():
    """Returns a standalone engine for backwards compatibility (models, decision_engine)"""
    return create_engine(DB_URI, connect_args={"connect_timeout": 10})

def get_engine():
    """Returns the shared engine for chatbot"""
    return engine
