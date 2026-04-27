import re
from sqlalchemy import text
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.core.db import get_engine
from backend.core.schema_loader import get_db_schema

class RetrievalAgent:
    def __init__(self, llm):
        self.engine = get_engine()
        self.llm = llm

    def generate_sql(self, question):
        schema = get_db_schema()
        prompt = f"""
You are a data analyst.
Convert the user question into a SQL query.
Database Schema:
{schema}
Rules:
- ONLY generate valid SELECT queries.
- DO NOT hallucinate columns. Only use columns strictly defined in the schema.
- MANDATORY: You must append `LIMIT 50` to the end of every query to protect against heavy data loads.
- Only run JOINs if absolutely necessary.
- Return ONLY the raw SQL query without text or markdown.
User Question:
{question}
"""
        response = self.llm.invoke(prompt)
        sql_query = response.content.strip()
        match = re.search(r"```(?:sql)?\s*(.*?)\s*```", sql_query, re.DOTALL | re.IGNORECASE)
        if match:
            sql_query = match.group(1).strip()
        return sql_query

    def run(self, question):
        sql_query = self.generate_sql(question)
            
        if not sql_query.upper().startswith("SELECT"):
            return "Error: Could not generate a safe SELECT query."
            
        try:
            with self.engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
            if not rows: return "No data found."
            return [dict(zip(result.keys(), row)) for row in rows]
        except Exception as e:
            return f"SQL Error: {str(e)}"
