import sys
import os
import json
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.chatbot.executor import chatbot_answer
from backend.core.config import GROQ_API_KEY
from langchain_groq import ChatGroq

def evaluate_llm_judge(llm, question, context, answer):
    if not answer or not context:
        return 0, 0
        
    prompt = f"""
You are an expert evaluator for an AI assistant.
Score the following Answer based on the Question and Retrieved Context.

Question: {question}
Retrieved Context: {context}
Generated Answer: {answer}

Provide a score from 1 to 5 for Relevance and Faithfulness:
- Relevance: Does the generated answer directly address the user's question? (1=Not relevant, 5=Perfectly answers)
- Faithfulness: Is the generated answer STRICTLY grounded in the Retrieved Context? (1=Hallucination/Not in context, 5=Fully grounded). If the answer includes information NOT present in the context, score low.

Return ONLY a valid JSON object in this exact format, with no markdown formatting or extra text:
{{"relevance": 5, "faithfulness": 5}}
"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        if content.startswith("```"):
             content = content.replace("```", "").strip()
             
        scores = json.loads(content)
        return scores.get("relevance", 0), scores.get("faithfulness", 0)
    except Exception as e:
        print(f"Error grading with LLM: {e}")
        return 0, 0

def evaluate():
    print("Starting RAG Evaluation (Phase 3)...")
    
    csv_path = os.path.join(os.path.dirname(__file__), "rag_eval_dataset.csv")
    df = pd.read_csv(csv_path)
    
    llm_judge = ChatGroq(temperature=0, api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    total = len(df)
    retrieval_success_count = 0
    total_relevance = 0
    total_faithfulness = 0
    
    results = []
    
    for index, row in df.iterrows():
        question = row['question']
        expected_keywords_str = str(row['expected_answer_keywords'])
        expected_keywords = [k.strip().lower() for k in expected_keywords_str.split(',')]
        
        print(f"\n[{index+1}/{total}] Question: {question}")
        
        try:
            # Get chatbot answer
            response = chatbot_answer(question)
            answer = response.get("answer", "")
            raw_results = response.get("raw_agent_results", {})
            context = raw_results.get("general", "")
            
            # Retrieval Evaluation: Check Top-K (are keywords in the raw retrieved context?)
            context_lower = context.lower()
            keywords_found = sum(1 for kw in expected_keywords if kw in context_lower)
            retrieval_score = keywords_found / len(expected_keywords) if expected_keywords else 0
            
            is_retrieval_success = retrieval_score >= 0.5 # At least half keywords found
            if is_retrieval_success:
                retrieval_success_count += 1
                
            # LLM-as-a-judge Scoring
            relevance, faithfulness = evaluate_llm_judge(llm_judge, question, context, answer)
            total_relevance += relevance
            total_faithfulness += faithfulness
            
            print(f"Retrieval Success: {is_retrieval_success} ({keywords_found}/{len(expected_keywords)} keywords)")
            print(f"Relevance Score: {relevance}/5")
            print(f"Faithfulness Score: {faithfulness}/5")
            
            results.append({
                "Question": question,
                "Expected Keywords": expected_keywords_str,
                "Retrieved Context": context,
                "Generated Answer": answer,
                "Retrieval Success": is_retrieval_success,
                "Relevance (1-5)": relevance,
                "Faithfulness (1-5)": faithfulness
            })
            
        except Exception as e:
            print(f"Error executing evaluation for question: {e}")
            results.append({
                "Question": question,
                "Expected Keywords": expected_keywords_str,
                "Retrieved Context": "ERROR",
                "Generated Answer": "ERROR",
                "Retrieval Success": False,
                "Relevance (1-5)": 0,
                "Faithfulness (1-5)": 0
            })
            
    # Save the scoring sheet
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "rag_scoring_sheet.csv")
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("PHASE 3: RAG EVALUATION RESULTS")
    print("="*50)
    print(f"Total Test Cases: {total}")
    print(f"Retrieval Accuracy (Top-K Keyword Match): {(retrieval_success_count/total)*100:.2f}% ({retrieval_success_count}/{total})")
    print(f"Average Relevance (LLM Judge):            {total_relevance/total:.2f} / 5.0")
    print(f"Average Faithfulness (LLM Judge):         {total_faithfulness/total:.2f} / 5.0")
    print("="*50)
    print(f"\nSaved detailed manual scoring sheet to: {output_path}")

if __name__ == "__main__":
    evaluate()
