import json

def summarize_results(results):
    summary = ""
    if "forecast" in results and isinstance(results["forecast"], list) and results["forecast"]:
        total_sales = sum([float(f.get('predicted_sales', 0)) for f in results["forecast"]])
        summary += f"- Forecasted Sales: ~{total_sales:.0f} units expected over the timeframe.\n"
    if "sentiment" in results and isinstance(results["sentiment"], list) and results["sentiment"]:
        avg_score = results["sentiment"][0].get('avg_review_score', "N/A")
        summary += f"- Sentiment Score: {avg_score} / 5.0\n"
    if "churn" in results and isinstance(results["churn"], list) and results["churn"]:
        risk = results["churn"][0].get('churn_risk', "N/A")
        summary += f"- Customer Churn Risk: {risk}\n"
    if "decision" in results and isinstance(results["decision"], list) and results["decision"]:
        summary += "- Internal Decision Rules Triggered:\n"
        for d in results["decision"]:
            summary += f"  > Action: {d.get('action_type', 'N/A')}. Details: {d.get('action_description', 'N/A')}\n"
    if "retrieval" in results:
        summary += f"- Raw Database Query Results: {results['retrieval']}\n"
    if "general" in results:
        summary += f"- Knowledge Base Results:\n{results['general']}\n"
    return summary if summary else "No actionable data was extracted."

def generate_final_response(llm, question, parsed_query, plan, results):
    summary = summarize_results(results)
    prompt = f"""
You are a strategic e-commerce business assistant.
The user asked: "{question}"

Our internal agents produced the following execution plan: {plan}
The parsed intent was: {json.dumps(parsed_query)}

Here is the summarized data returned by the agents:
{summary}

Explain the result in simple, professional business language. 
IMPORTANT: ALWAYS structure your final answer to include a clear "Reason" or "Reasoning" section.
"""
    response = llm.invoke(prompt)
    return response.content
