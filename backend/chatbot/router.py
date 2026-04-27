import json

def parse_query(llm, question):
    prompt = f"""
You are a router for an e-commerce assistant.
Extract the user's intent from the following question into a strict JSON object.
Schema:
{{
    "intent": "forecast" | "sentiment" | "churn" | "strategy" | "retrieval" | "general",
    "entity": "a specific product category or customer name (or empty string)",
    "time_range": "a specific time range (or empty string)",
    "requires_decision": true or false
}}
Rules:
- If the user asks a meta-question about how the chatbot works, its architecture, or the datasets it is built on, YOU MUST classify the intent as 'general'.

User Query: "{question}"
Return ONLY valid JSON.
"""
    response = llm.invoke(prompt)
    content = response.content.strip()
    if content.startswith("```json"):
        content = content.replace("```json", "").replace("```", "").strip()
    if content.startswith("```"):
         content = content.replace("```", "").strip()
         
    try:
        parsed = json.loads(content)
        if "requires_decision" not in parsed:
            parsed["requires_decision"] = False
        if "intent" not in parsed:
            parsed["intent"] = "retrieval"
        return parsed
    except json.JSONDecodeError:
        return {"intent": "retrieval", "entity": "", "time_range": "", "requires_decision": False}
