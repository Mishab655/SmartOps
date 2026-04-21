import os
from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def count_tokens(text):
    return len(text.split())

def ingest_knowledge_base():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    kb_dir = os.path.join(base_dir, "../../data/knowledge_base")
    db_dir = os.path.join(base_dir, "../../data/chroma_db")
    
    if not os.path.exists(kb_dir):
        print(f"Directory '{kb_dir}' not found. Please create it and add documents.")
        return
        
    print(f"Loading documents from {kb_dir}...")
    
    # Load txt and pdfs
    txt_loader = DirectoryLoader(kb_dir, glob="**/*.txt", loader_cls=TextLoader, loader_kwargs={'encoding': 'utf-8'})
    pdf_loader = DirectoryLoader(kb_dir, glob="**/*.pdf", loader_cls=PyPDFLoader)
    
    docs = []
    try:
        txt_docs = txt_loader.load()
        docs.extend(txt_docs)
    except Exception as e:
        print(f"Error loading txts: {e}")
        
    try:
        pdf_docs = pdf_loader.load()
        docs.extend(pdf_docs)
    except Exception as e:
        print(f"Error loading pdfs: {e}")
        
    if not docs:
        print("No documents found to ingest.")
        return
        
    print(f"Loaded {len(docs)} documents. Splitting text...")
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=200,
        length_function=count_tokens
    )
    chunks = text_splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks.")
    
    # Embedding
    print("Initializing embedding model (all-MiniLM-L6-v2) - this may download models on first run...")
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create / update vector store
    print(f"Creating vector store at '{db_dir}'...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_dir
    )
    
    print("Ingestion completed successfully!")

if __name__ == "__main__":
    ingest_knowledge_base()
