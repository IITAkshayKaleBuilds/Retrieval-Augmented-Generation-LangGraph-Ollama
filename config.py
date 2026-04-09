# config.py
import os

LLM_MODEL = "gemma4:26b"
EMBEDDING_MODEL = 'qwen3-embedding:4b'
BASE_URL = 'http://127.0.0.1:11434'

DATA_DIR = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/data"
CHROMA_DIR = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/chroma_financial_db"
COLLECTION_NAME = "financial_docs"
DEBUG_PATH = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/debug_logs"

def setup_env():
    import os
    try:
        #

        LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")
    except:
        print("Not running in Colab or secret not found")

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com/"
    os.environ["LANGCHAIN_PROJECT"] = "Retrieval-Augmented-Generation-LangGraph-Ollama"
