import sys
import os
import json
import time
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.chatbot.executor import chatbot_answer
from backend.core.config import GROQ_API_KEY
from langchain_groq import ChatGroq

def evaluate_e2e_llm_judge(llm, question, raw_data, answer):
    if not answer or "error" in answer.lower():
        return 0, 0, 0
        
    prompt = f"""
You are an expert evaluator for an AI e-commerce assistant.
Score the following Final Answer based on the User Question.

Question: {question}
Agent Raw Data Used: {str(raw_data)[:2000]}  # Truncate to prevent context window overflow
Generated Answer: {answer}

Provide a score from 1 to 5 for Correctness, Relevance, and Clarity:
- Correctness (1-5): Is the data presented in the answer accurate based on the Agent Raw Data Used? (1=Hallucinated/Wrong, 5=Perfectly accurate). If no raw data is needed (e.g. general questions), score 5.
- Relevance (1-5): Does the answer fully address the multi-part user query? (1=Not relevant/Missed parts, 5=Fully addressed)
- Clarity (1-5): Is the tone conversational, structured (e.g., uses bullet points if needed), and easy to read? (1=Wall of text/confusing, 5=Excellent formatting and tone)

Return ONLY a valid JSON object in this exact format, with no markdown formatting or extra text:
{{"correctness": 5, "relevance": 5, "clarity": 5}}
"""
    try:
        response = llm.invoke(prompt)
        content = response.content.strip()
        if content.startswith("```json"):
            content = content.replace("```json", "").replace("```", "").strip()
        if content.startswith("```"):
             content = content.replace("```", "").strip()
             
        scores = json.loads(content)
        return scores.get("correctness", 0), scores.get("relevance", 0), scores.get("clarity", 0)
    except Exception as e:
        print(f"Error grading with LLM: {e}")
        return 0, 0, 0

def evaluate():
    print("Starting End-to-End Evaluation (Phase 4)...")
    
    csv_path = os.path.join(os.path.dirname(__file__), "e2e_eval_dataset.csv")
    df = pd.read_csv(csv_path)
    
    llm_judge = ChatGroq(temperature=0, api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")
    
    total = len(df)
    total_correctness = 0
    total_relevance = 0
    total_clarity = 0
    total_latency = 0
    
    results = []
    
    for index, row in df.iterrows():
        question = row['question']
        
        print(f"\n[{index+1}/{total}] Question: {question}")
        
        try:
            start_time = time.time()
            
            # Get chatbot answer (End to End)
            response = chatbot_answer(question)
            
            end_time = time.time()
            latency = end_time - start_time
            
            answer = response.get("answer", "")
            raw_data = response.get("raw_agent_results", {})
            
            # LLM-as-a-judge Scoring
            correctness, relevance, clarity = evaluate_e2e_llm_judge(llm_judge, question, raw_data, answer)
            
            total_correctness += correctness
            total_relevance += relevance
            total_clarity += clarity
            total_latency += latency
            
            print(f"Latency: {latency:.2f}s")
            print(f"Correctness Score: {correctness}/5")
            print(f"Relevance Score:   {relevance}/5")
            print(f"Clarity Score:     {clarity}/5")
            
            results.append({
                "Question": question,
                "Latency (s)": round(latency, 2),
                "Generated Answer": answer,
                "Correctness (1-5)": correctness,
                "Relevance (1-5)": relevance,
                "Clarity (1-5)": clarity
            })
            
        except Exception as e:
            print(f"Error executing evaluation for question: {e}")
            results.append({
                "Question": question,
                "Latency (s)": 0,
                "Generated Answer": "ERROR",
                "Correctness (1-5)": 0,
                "Relevance (1-5)": 0,
                "Clarity (1-5)": 0
            })
            
    # Save the scoring sheet
    results_df = pd.DataFrame(results)
    output_path = os.path.join(os.path.dirname(__file__), "e2e_scoring_sheet.csv")
    results_df.to_csv(output_path, index=False)
    
    print("\n" + "="*50)
    print("PHASE 4: END-TO-END EVALUATION RESULTS")
    print("="*50)
    print(f"Total Test Cases: {total}")
    print(f"Average Latency:                  {total_latency/total:.2f} seconds")
    print(f"Average Correctness (LLM Judge):  {total_correctness/total:.2f} / 5.0")
    print(f"Average Relevance (LLM Judge):    {total_relevance/total:.2f} / 5.0")
    print(f"Average Clarity (LLM Judge):      {total_clarity/total:.2f} / 5.0")
    print("="*50)
    print(f"\nSaved detailed manual scoring sheet to: {output_path}")

if __name__ == "__main__":
    evaluate()
