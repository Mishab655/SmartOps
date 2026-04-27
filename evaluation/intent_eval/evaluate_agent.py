import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.chatbot.executor import chatbot_answer

def evaluate():
    print("Starting Agent System Evaluation (Phase 2)...")
    
    csv_path = os.path.join(os.path.dirname(__file__), "agent_eval_dataset.csv")
    df = pd.read_csv(csv_path)
    
    total = len(df)
    intent_correct_count = 0
    agent_correct_count = 0
    task_success_count = 0
    
    for index, row in df.iterrows():
        question = row['question']
        expected_intent = row['expected_intent']
        expected_agent = row['expected_agent']
        
        print(f"\n[{index+1}/{total}] Question: {question}")
        print(f"Expected Intent: {expected_intent} | Expected Agent: {expected_agent}")
        
        try:
            # Get chatbot answer
            response = chatbot_answer(question)
            
            # Extract predictions
            parsed_query = response.get("parsed_query", {})
            predicted_intent = parsed_query.get("intent", "UNKNOWN")
            
            actions_taken = response.get("actions_taken", [])
            predicted_agent = actions_taken[0] if actions_taken else "NONE"
            
            answer = response.get("answer", "")
            error = response.get("error", None)
            
            # Task success: answer is not empty and error is None
            task_success = bool(answer) and not error
            
        except Exception as e:
            print(f"Error executing chatbot: {e}")
            predicted_intent = "ERROR"
            predicted_agent = "ERROR"
            task_success = False
            
        print(f"Predicted Intent: {predicted_intent} | Predicted Agent: {predicted_agent}")
        
        # Metrics
        is_intent_correct = (predicted_intent == expected_intent)
        is_agent_correct = (predicted_agent == expected_agent)
        
        if is_intent_correct:
            intent_correct_count += 1
            
        if is_agent_correct:
            agent_correct_count += 1
            
        if task_success:
            task_success_count += 1
            
        # Insights
        if is_intent_correct and not is_agent_correct:
            print("WARNING: Planner Inconsistency Detected: Intent correct but agent correct is False")
        elif not is_agent_correct:
            print("WARNING: Planner Issue: Agent correct is False")
            
    print("\n" + "="*50)
    print("PHASE 2: AGENT SYSTEM EVALUATION RESULTS")
    print("="*50)
    print(f"Total Test Cases: {total}")
    print(f"Intent Accuracy:      {(intent_correct_count/total)*100:.2f}% ({intent_correct_count}/{total})")
    print(f"Agent Accuracy:       {(agent_correct_count/total)*100:.2f}% ({agent_correct_count}/{total})")
    print(f"Task Completion Rate: {(task_success_count/total)*100:.2f}% ({task_success_count}/{total})")
    print("="*50)

if __name__ == "__main__":
    evaluate()
