def plan_execution(parsed_query, question):
    intent = parsed_query.get("intent", "retrieval").lower()
    requires_decision = parsed_query.get("requires_decision", False)
    entity = parsed_query.get("entity", "")
    
    plan = []
    
    if intent == "strategy" or requires_decision:
        if entity:
            plan.append("forecast")
            plan.append("sentiment")
        if "customer" in question.lower() or "churn" in question.lower() or "retention" in question.lower():
            plan.append("churn")
        plan.append("decision")
        return plan
    elif intent == "forecast":
        return ["forecast"]
    elif intent == "sentiment":
        return ["sentiment"]
    elif intent == "churn":
        return ["churn"]
    elif intent == "general":
        return ["general"]
    else:
        return ["retrieval"]
