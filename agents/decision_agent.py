from sqlalchemy import text
from core.db import get_engine

class DecisionAgent:
    def __init__(self):
        self.engine = get_engine()

    def run(self, entity, forecast_data, sentiment_data):
        if not entity: return "No category or customer provided for decisions."
        query = "SELECT * FROM decision_action_log WHERE target_entity_id = :entity ORDER BY priority ASC LIMIT 5"
        with self.engine.connect() as conn:
            result = conn.execute(text(query), {"entity": entity})
            rows = result.fetchall()
            
        if rows:
            return [dict(zip(result.keys(), row)) for row in rows]
            
        if not forecast_data or not sentiment_data:
            return [{"fallback_decision": "Not enough data to generate rule-based fallback decision."}]
            
        avg_sentiment = float(sentiment_data[0].get('avg_review_score', 0))
        total_forecasted_sales = sum([float(f.get('predicted_sales', 0)) for f in forecast_data])
        
        fallback_rules = []
        if total_forecasted_sales > 1000 and avg_sentiment < 3.5:
             fallback_rules.append({"action_type": "Critical QA Fallback", "action_description": f"High demand ({total_forecasted_sales} expected) but poor sentiment ({avg_sentiment}). Audit immediately.", "priority": 1})
        elif total_forecasted_sales > 1000 and avg_sentiment >= 3.5:
             fallback_rules.append({"action_type": "Scale Fallback", "action_description": f"Strong demand and healthy sentiment ({avg_sentiment}). Ensure stock.", "priority": 2})
        elif total_forecasted_sales < 1000 and avg_sentiment >= 3.5:
             fallback_rules.append({"action_type": "Marketing Push Fallback", "action_description": f"Low demand but high satisfaction. Needs better visibility.", "priority": 2})
        else:
             fallback_rules.append({"action_type": "Sunset Warning Fallback", "action_description": "Low demand and poor sentiment. Consider discontinuing.", "priority": 3})
        return fallback_rules
