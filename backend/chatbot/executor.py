from langchain_groq import ChatGroq
from backend.core.config import GROQ_API_KEY
from backend.chatbot.router import parse_query
from backend.chatbot.planner import plan_execution
from backend.chatbot.generator import generate_final_response

from backend.agents.forecast_agent import ForecastAgent
from backend.agents.sentiment_agent import SentimentAgent
from backend.agents.churn_agent import ChurnAgent
from backend.agents.decision_agent import DecisionAgent
from backend.agents.retrieval_agent import RetrievalAgent
from backend.agents.rag_agent import RagAgent


llm = ChatGroq(temperature=0, api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

forecast_agent = ForecastAgent()
sentiment_agent = SentimentAgent()
churn_agent = ChurnAgent()
decision_agent = DecisionAgent()
retrieval_agent = RetrievalAgent(llm)
rag_agent = RagAgent()


def execute_plan(plan, parsed_query, original_question):
    results = {}
    entity = parsed_query.get("entity", "")
    for step in plan:
        if step == "forecast":
            results["forecast"] = forecast_agent.run(entity)
        elif step == "sentiment":
            results["sentiment"] = sentiment_agent.run(entity)
        elif step == "churn":
            results["churn"] = churn_agent.run(entity)
        elif step == "decision":
            f_data = results.get("forecast", [])
            s_data = results.get("sentiment", [])
            results["decision"] = decision_agent.run(entity, f_data, s_data)
        elif step == "retrieval":
            results["retrieval"] = retrieval_agent.run(original_question)
        elif step == "general":
            results["general"] = rag_agent.run(original_question)
    return results

def chatbot_answer(question):
    try:
        parsed_query = parse_query(llm, question)
        intent = parsed_query.get("intent", "retrieval")
        entity = parsed_query.get("entity", "")
        if intent in ["forecast", "sentiment", "strategy"] and not entity:
            return {
                "user_query": question, "parsed_query": parsed_query, "plan": [],
                "actions_taken": [], "raw_agent_results": {},
                "answer": "Which product category or specific customer are you referring to? Please provide a bit more detail.",
                "error": None
            }
        
        plan = plan_execution(parsed_query, question)
        results = execute_plan(plan, parsed_query, question)
        answer = generate_final_response(llm, question, parsed_query, plan, results)
        
        return {
            "user_query": question, "parsed_query": parsed_query, "plan": plan,
            "actions_taken": list(results.keys()), "raw_agent_results": results,
            "answer": answer, "error": None
        }
    except Exception as e:
        return {"user_query": question, "answer": f"Sorry, I encountered an error: {str(e)}", "error": str(e)}
