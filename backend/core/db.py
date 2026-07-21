from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool
from backend.core.config import DB_URI

# Added connect_timeout and NullPool to prevent infinite hanging when DB is unreachable or when using Supabase PgBouncer
engine = create_engine(DB_URI, connect_args={"connect_timeout": 10}, poolclass=NullPool)

def get_db_engine():
    """Returns a standalone engine for backwards compatibility (models, decision_engine)"""
    return create_engine(DB_URI, connect_args={"connect_timeout": 10}, poolclass=NullPool)

def get_engine():
    """Returns the shared engine for chatbot"""
    return engine
