from sqlalchemy import inspect
from core.db import get_engine

def get_db_schema():

    engine = get_engine()
    inspector = inspect(engine)

    schema = ""

    for table in inspector.get_table_names():

        schema += f"\nTable: {table}\n"

        columns = inspector.get_columns(table)

        for col in columns:
            schema += f"{col['name']} ({col['type']})\n"

    return schema