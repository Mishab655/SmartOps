import sys
import os
import pandas as pd
from sqlalchemy import text

# Add backend to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.core.db import get_engine
from backend.core.config import GROQ_API_KEY
from langchain_groq import ChatGroq
from backend.agents.retrieval_agent import RetrievalAgent

def compare_results(expected_res, generated_res):
    if isinstance(expected_res, str) or isinstance(generated_res, str):
        return expected_res == generated_res
        
    if not isinstance(expected_res, list) or not isinstance(generated_res, list):
        return False
        
    if len(expected_res) != len(generated_res):
        return False
        
    for i in range(len(expected_res)):
        # Loose comparison: convert values to strings and ignore keys
        # This handles cases where generated SQL uses different column aliases
        vals1 = [str(v).lower() for v in expected_res[i].values()]
        vals2 = [str(v).lower() for v in generated_res[i].values()]
        
        if vals1 != vals2:
            if sorted(vals1) != sorted(vals2):
                return False
    return True

def run_query(engine, query):
    if not query.upper().startswith("SELECT"):
        return "Invalid query"
    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
        if not rows:
            return []
        return [dict(zip(result.keys(), row)) for row in rows]
    except Exception as e:
        return f"Error: {str(e)}"

def evaluate():
    print("Starting SQL Generation Evaluation (Phase 1)...")
    
    csv_path = os.path.join(os.path.dirname(__file__), "sql_eval_dataset.csv")
    df = pd.read_csv(csv_path)
    
    # Initialize the same LLM the chatbot uses
    llm = ChatGroq(temperature=0, api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    agent = RetrievalAgent(llm)
    engine = get_engine()
    
    total = len(df)
    exact_matches = 0
    execution_matches = 0
    
    for index, row in df.iterrows():
        question = row['question']
        expected_sql = row['expected_sql']
        
        print(f"\n[{index+1}/{total}] Question: {question}")
        
        try:
            generated_sql = agent.generate_sql(question)
        except Exception as e:
            generated_sql = f"Error generating: {str(e)}"
            
        print(f"Expected : {expected_sql}")
        print(f"Generated: {generated_sql}")
        
        # 1. Exact Match Check
        norm_expected = expected_sql.strip().rstrip(';').lower()
        norm_generated = generated_sql.strip().rstrip(';').lower()
        
        # Ignore LIMIT 50 which the agent is prompted to add
        if norm_generated.endswith("limit 50") and not norm_expected.endswith("limit 50"):
            norm_generated = norm_generated[:-8].strip()
            
        is_exact = (norm_expected == norm_generated)
        if is_exact:
            exact_matches += 1
            print("Exact Match: YES")
        else:
            print("Exact Match: NO")
            
        # 2. Execution Match Check
        expected_data = run_query(engine, expected_sql)
        generated_data = run_query(engine, generated_sql)
        
        if isinstance(expected_data, str) and expected_data.startswith("Error"):
            print(f"Warning: Expected SQL failed! {expected_data}")
            is_exec_match = False
        elif isinstance(generated_data, str) and generated_data.startswith("Error"):
            print(f"Execution failed: {generated_data}")
            is_exec_match = False
        else:
            is_exec_match = compare_results(expected_data, generated_data)
            if is_exec_match:
                execution_matches += 1
                print("Execution Match: YES")
            else:
                print("Execution Match: NO")
        
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Total Test Cases: {total}")
    print(f"Exact Match Accuracy:     {(exact_matches/total)*100:.2f}% ({exact_matches}/{total})")
    print(f"Execution Match Accuracy: {(execution_matches/total)*100:.2f}% ({execution_matches}/{total})")
    print("="*50)
    
if __name__ == "__main__":
    evaluate()
