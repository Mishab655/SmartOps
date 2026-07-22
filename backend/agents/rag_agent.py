import os
import requests
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

class CustomHFEmbeddings(Embeddings):
    def __init__(self, api_key: str, model_name: str):
        self.api_url = f"https://router.huggingface.co/hf-inference/pipeline/feature-extraction/{model_name}"
        self.headers = {"Authorization": f"Bearer {api_key}"}

    def _embed(self, texts):
        response = requests.post(
            self.api_url, 
            headers=self.headers, 
            json={"inputs": texts, "options": {"wait_for_model": True}}
        )
        if response.status_code != 200:
            raise ValueError(f"HuggingFace API Error ({response.status_code}): {response.text}")
        return response.json()

    def embed_documents(self, texts):
        res = self._embed(texts)
        if isinstance(res, dict) and "error" in res:
            raise ValueError(f"HuggingFace API Error: {res['error']}")
        return res

    def embed_query(self, text):
        res = self._embed(text)
        if isinstance(res, dict) and "error" in res:
            raise ValueError(f"HuggingFace API Error: {res['error']}")
        if isinstance(res, list) and len(res) > 0 and isinstance(res[0], list):
            return res[0]
        return res

class RagAgent:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_dir = os.path.join(base_dir, "../../data/chroma_db")
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set. Required for RAG embeddings via API.")
            
        self.embeddings = CustomHFEmbeddings(
            api_key=hf_token, 
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        
    def run(self, question):
        if not os.path.exists(self.db_dir):
            return "Knowledge base not initialized. Run ingest.py first."
            
        try:
            vectorstore = Chroma(
                persist_directory=self.db_dir,
                embedding_function=self.embeddings
            )
            
            # Retrieve top 5 chunks
            docs = vectorstore.similarity_search(question, k=5)
            
            if not docs:
                return "No relevant information found in the knowledge base."
                
            # Combine retrieved texts
            context = "\n\n".join([f"Source ({doc.metadata.get('source', 'Unknown')}):\n{doc.page_content}" for doc in docs])
            return context
            
        except Exception as e:
            return f"RAG Search Error: {str(e)}"
                                    