import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.executor import chatbot_answer

app = FastAPI()


class Query(BaseModel):
    question: str


@app.post("/chat")

def chat(query: Query):

    result = chatbot_answer(query.question)

    return result