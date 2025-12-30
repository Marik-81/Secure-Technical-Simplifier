import os
import sys

# --- 1. CONFIGURATION (Must be first to fix the hang) ---
# We set these BEFORE importing the heavy libraries (Chroma/LangChain)
# This prevents the code from waiting for a server that doesn't exist.
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# --- 2. NOW WE IMPORT THE REST ---
from dotenv import load_dotenv
from os import getenv
from pathlib import Path

# Print immediately so you know it's alive (flush=True forces it to show)
print("--- DEBUG: System is initializing... (Please wait 30s) ---", flush=True)

from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_chroma import Chroma
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Fall back to sensible defaults if environment variables are not set
OLLAMA_BASE_URL = getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = getenv("LLM_MODEL", "llama3:8b") 
CHROMA_PATH = getenv("CHROMA_PATH", "chroma_db")
DATA_PATH = "data"

# --- 1. CORE RAG FUNCTIONS ---

# 1. Load Documents
def load_documents():
    """Loads documents from the 'data/' directory using PyPDF and Docx2txt loaders."""
    documents = []
    # Ensure data directory exists
    if not Path(DATA_PATH).exists():
        os.makedirs(DATA_PATH)
        print(f"[WARN] Created missing '{DATA_PATH}' folder. Put your files there!")
        return []
        
    data_dir = Path(DATA_PATH)
    
    # Load PDF files
    for pdf_file in data_dir.glob("*.pdf"):
        print(f"Loading PDF: {pdf_file.name}")
        documents.extend(PyPDFLoader(str(pdf_file)).load())
    
    # Load DOCX files
    for docx_file in data_dir.glob("*.docx"):
        print(f"Loading DOCX: {docx_file.name}")
        documents.extend(Docx2txtLoader(str(docx_file)).load())
        
    return documents

# 2. Split Documents
def split_documents(documents):
    """Splits documents into smaller chunks for the vector store."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)

# 3. Create or Load Vector Store (Embedding)
def setup_vector_store(chunks):
    """
    Sets up the Chroma vector store. If data exists, loads it.
    Otherwise, creates embeddings and saves them to disk.
    """
    try:
        # Initialize the Ollama Embeddings model
        embeddings = OllamaEmbeddings(model=LLM_MODEL, base_url=OLLAMA_BASE_URL)
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize Ollama Embeddings: {e}")
        print("Ensure the Ollama server is running and the model is pulled.")
        return None

    # Check if vector store already exists on disk
    if Path(CHROMA_PATH).exists():
        print("\n[INFO] Loading existing vector store from disk...")
        vector_store = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)
        return vector_store

    # If no store exists, create a new one
    if chunks:
        print(f"\n[INFO] Creating new vector store and saving to {CHROMA_PATH}...")
        vector_store = Chroma.from_documents(
            chunks, embeddings, persist_directory=CHROMA_PATH
        )
        print("[SUCCESS] Vector store created successfully!")
        return vector_store
    else:
        print("\n[ERROR] No documents found in the 'data' folder. Cannot create vector store.")
        return None

# Streaming callback to print tokens as they arrive
class StdOutCallbackHandler(BaseCallbackHandler):
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        print(token, end="", flush=True)

# 4. Define and Run RAG Chain (Runnable pipeline)
def setup_rag_chain(vector_store):
    """Builds a retrieval + generation pipeline using Runnables with streaming."""
    callbacks = [StdOutCallbackHandler()]
    llm = OllamaLLM(model=LLM_MODEL, base_url=OLLAMA_BASE_URL, callbacks=callbacks)

    custom_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="""
        You are a Private & Secure Technical Simplifier. Your role is to translate 
        complex technical information from the provided context into simple, 
        non-technical language suitable for a layperson.
        
        Answer the question based ONLY on the following context.
        If the context does not contain the answer, state that you cannot find the
        information in the documents. Do not use outside knowledge.
        
        CONTEXT: {context}
        QUESTION: {question}
        
        SIMPLIFIED ANSWER:
        """
    )

    retriever = vector_store.as_retriever()

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | custom_prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain

# --- 2. COMMAND LINE INTERFACE (CLI) ---

def cli_main():
    print("----------------------------------------------------------------------")
    print("🔒 Private & Secure Technical Simplifier (CLI RAG System)")
    print(f"Model: {LLM_MODEL} | Ollama Server: {OLLAMA_BASE_URL}")
    print("----------------------------------------------------------------------")
    print("[SETUP] Initializing RAG System...")

    try:
        documents = load_documents()
        if not documents:
            print(f"[ERROR] No documents found in the '{DATA_PATH}' folder.")
            print("Please add PDF or DOCX files to the data folder and restart.")
            sys.exit(1)
        
        chunks = split_documents(documents)
        print(f"[INFO] Loaded {len(documents)} documents and split into {len(chunks)} chunks.")
    except Exception as e:
        print(f"\n[ERROR] An error occurred during document loading/splitting: {e}")
        sys.exit(1)

    qa_chain = None
    try:
        vector_store = setup_vector_store(chunks)
        if vector_store:
            qa_chain = setup_rag_chain(vector_store)
            print("[SUCCESS] RAG System Initialized! Ready to simplify.")
    except Exception as e:
        print(f"\n[CRITICAL ERROR] Failed to initialize RAG chain: {e}")
        print("Please check your Ollama server status and model name.")
        sys.exit(1)

    if qa_chain:
        while True:
            print("\n----------------------------------------------------------------------")
            query = input("Your Technical Query (or type 'quit' to exit): \n> ")
            
            if query.lower() in ["quit", "exit"]:
                print("Exiting RAG System. Goodbye!")
                break
            
            if not query.strip():
                continue

            print("\n[INFO] Generating Simplified Answer (streaming)...\n")

            try:
                # Invoke the runnable pipeline; tokens stream via callback
                _ = qa_chain.invoke(query)
                print("\n\n✅ SIMPLIFIED ANSWER (complete above).")

                # Show sources
                print("\n--- SOURCES USED ---")
                for i, doc in enumerate(vector_store.as_retriever().invoke(query)):
                    source_name = Path(doc.metadata.get('source', 'N/A')).name
                    page_num = doc.metadata.get('page', 'N/A')
                    print(f"\n[Source {i+1}] File: {source_name} | Page: {page_num}")
                print("--------------------")

            except Exception as e:
                print(f"\n[ERROR] An error occurred during chain execution: {e}")
                print("Check if the LLM model is still loaded and reachable.")

if __name__ == "__main__":
    cli_main()