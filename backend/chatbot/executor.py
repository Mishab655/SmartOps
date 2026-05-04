from langchain_groq import ChatGroq
from backend.core.config import GROQ_API_KEY
from backend.chatbot.router import parse_query
from backend.chatbot.planner import plan_execution
from backend.chatbot.generator import generate_final_response


llm = ChatGroq(temperature=0, api_key=GROQ_API_KEY, model_name="llama-3.1-8b-instant")

# Lazy-loaded agents dictionary (Fixes Cloud Run startup timeout)
_agents = {}

def get_agent(agent_name):
    if agent_name not in _agents:
        if agent_name == "forecast":
            from backend.agents.forecast_agent import ForecastAgent
            _agents["forecast"] = ForecastAgent()
        elif agent_name == "sentiment":
            from backend.agents.sentiment_agent import SentimentAgent
            _agents["sentiment"] = SentimentAgent()
        elif agent_name == "churn":
            from backend.agents.churn_agent import ChurnAgent
            _agents["churn"] = ChurnAgent()
        elif agent_name == "decision":
            from backend.agents.decision_agent import DecisionAgent
            _agents["decision"] = DecisionAgent()
        elif agent_name == "retrieval":
            from backend.agents.retrieval_agent import RetrievalAgent
            _agents["retrieval"] = RetrievalAgent(llm)
        elif agent_name == "rag":
            from backend.agents.rag_agent import RagAgent
            _agents["rag"] = RagAgent()
    return _agents[agent_name]


def execute_plan(plan, parsed_query, original_question):
    results = {}
    entity = parsed_query.get("entity", "")
    for step in plan:
        if step == "forecast":
            results["forecast"] = get_agent("forecast").run(entity)
        elif step == "sentiment":
            results["sentiment"] = get_agent("sentiment").run(entity)
        elif step == "churn":
            results["churn"] = get_agent("churn").run(entity)
        elif step == "decision":
            f_data = results.get("forecast", [])
            s_data = results.get("sentiment", [])
            results["decision"] = get_agent("decision").run(entity, f_data, s_data)
        elif step == "retrieval":
            results["retrieval"] = get_agent("retrieval").run(original_question)
        elif step == "general":
            results["general"] = get_agent("rag").run(original_question)
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
