from sqlalchemy import text
from backend.core.db import get_engine

class ForecastAgent:
    def __init__(self):
        self.engine = get_engine()

    def run(self, entity):
        if not entity: return "No category provided for forecast."
        query = "SELECT * FROM category_forecast WHERE product_category = :entity ORDER BY forecast_date DESC LIMIT 30"
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"entity": entity})
            rows = result.fetchall()
        if not rows: return []
        return [dict(zip(result.keys(), row)) for row in rows]
