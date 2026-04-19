from sqlalchemy import text
from core.db import get_engine

class SentimentAgent:
    def __init__(self):
        self.engine = get_engine()

    def run(self, entity):
        if not entity: return "No category provided for sentiment."
        query = "SELECT * FROM category_sentiment_summary WHERE product_category = :entity"
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"entity": entity})
            rows = result.fetchall()
        if not rows: return []
        return [dict(zip(result.keys(), row)) for row in rows]
