from sqlalchemy import text
from backend.core.db import get_engine

class ChurnAgent:
    def __init__(self):
        self.engine = get_engine()

    def run(self, entity):
        if not entity: return "No customer provided for churn analysis."
        query = "SELECT * FROM customer_churn_prediction WHERE customer_unique_id = :entity"
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"entity": entity})
            rows = result.fetchall()
        if not rows: return []
        return [dict(zip(result.keys(), row)) for row in rows]
