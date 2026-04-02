# config.py
import os

LLM_MODEL = "mistral:7b"
EMBEDDING_MODEL = 'embeddinggemma:300m'
BASE_URL = 'http://127.0.0.1:11434'

DATA_DIR = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/data"
CHROMA_DIR = "/content/drive/MyDrive/chroma_financial_db"
COLLECTION_NAME = "financial_docs"
DEBUG_PATH = "/content/drive/MyDrive/debug_logs"

def setup_env():
    import os
    try:
        from google.colab import userdata

        os.environ["LANGSMITH_API_KEY"] = userdata.get("LANGSMITH_API_KEY")
    except:
        print("Not running in Colab or secret not found")

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com/"
    os.environ["LANGCHAIN_PROJECT"] = "Retrieval-Augmented-Generation-LangGraph-Ollama"
