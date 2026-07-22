import os
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_chroma import Chroma

class RagAgent:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_dir = os.path.join(base_dir, "../../data/chroma_db")
        
        hf_token = os.environ.get("HF_TOKEN")
        if not hf_token:
            raise ValueError("HF_TOKEN environment variable is not set. Required for RAG embeddings via API.")
            
        self.embeddings = HuggingFaceInferenceAPIEmbeddings(
            api_key=hf_token, 
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            api_url="https://router.huggingface.co/hf-inference/models/sentence-transformers/all-MiniLM-L6-v2"
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
                                    