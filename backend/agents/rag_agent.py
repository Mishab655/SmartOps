import os
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

class RagAgent:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.db_dir = os.path.join(base_dir, "../../data/chroma_db")
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
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
                                    