import streamlit as st
import requests

st.title("AI Business Advisor Chatbot")

# Initialize chat history (tuples of: role, message, sql_query)
if "messages" not in st.session_state:
    st.session_state.messages = []

# 1. Render existing chat history first
for role, msg, sql in st.session_state.messages:
    if role == "user":
        st.chat_message("user").write(msg)
    else:
        with st.chat_message("assistant"):
            st.write(msg)
            if sql:
                with st.expander("Show SQL Details"):
                    st.code(sql, language="sql")

# 2. Get new user input
user_input = st.chat_input("Ask about sales, forecast, churn...")

if user_input:
    # Render user message immediately
    st.chat_message("user").write(user_input)
    
    with st.spinner("Thinking..."):
        try:
            response = requests.post(
                "http://localhost:8000/chat",
                json={"question": user_input}
            )
            data = response.json()
            answer = data.get("answer", "No answer provided")
            sql_query = data.get("sql")
        except Exception as e:
            answer = f"Error connecting to backend: {str(e)}"
            sql_query = None

    # Render bot response immediately
    with st.chat_message("assistant"):
        st.write(answer)
        if sql_query:
            with st.expander("Show SQL Details"):
                st.code(sql_query, language="sql")
    
    # Save both to history
    st.session_state.messages.append(("user", user_input, None))
    st.session_state.messages.append(("bot", answer, sql_query))