# retrieve_docs
# web_search
import os
#

# Load API key from Colab secrets
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

# Non-sensitive configs
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com/"
os.environ["LANGCHAIN_PROJECT"] = "Retrieval-Augmented-Generation-LangGraph-Ollama"

# Configuration
DATA_DIR = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/data"
CHROMA_DIR = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/chroma_financial_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = 'qwen3-embedding:0.6b'
BASE_URL = 'http://127.0.0.1:11434'
LLM_MODEL = "qwen3.5:9b"
DEBUG_PATH = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/debug_logs"

from langchain_core.tools import tool
from RAG_Applications.scripts import utils


@tool
def retrieve_docs(query:str, k=5):
    """
    Retrieve relevant financial documents from ChromaDB.
    Extracts filters from query and retrieves matching documents.

    Args:
        query: The search query (e.g., "What was Amazon's revenue in Q2 2025?")
        k: Number of documents to retrieve. generally prefer 5 docs

    Returns:
        Retrieved documents with metadata as formatted string
    """
    print(f"\n[TOOL] retrieve_docs called")
    print(f"[QUERY] {query}")

    filters = utils.extract_filters(query)
    ranking_keywords = utils.generate_ranking_keywords(query)
    
    # fetch more docs than needed for better re-ranking
    results = utils.search_docs(query, filters, ranking_keywords, k=10*k)

    # rank retrieved docs
    docs = utils.rank_documents_by_keywords(results, ranking_keywords, k=k)

    print(f"[RETRIEVED] {len(docs)} documents")

    # format extracted docs or chunks
    if len(docs)==0:
        return f"No ducuments found for the query: '{query}'. Try rephrasing query or use different filter."
    
    # final format
    # --- Document {i} ---
    retrieved_text = []
    for i, doc in enumerate(docs, 1):
        doc_text = [f"--- Document {i} ---"]

        # add all metadata
        for key, value in doc.metadata.items():
            doc_text.append(f"{key}: {value}")

        # add content
        doc_text.append(f"\nContent:\n{doc.page_content}")

        text = "\n".join(doc_text)
        retrieved_text.append(text)

    retrieved_text = "\n".join(retrieved_text)

    os.makedirs(DEBUG_PATH, exist_ok=True)
    with open(f"{DEBUG_PATH}/retrieved_reranked_docs.md", "w", encoding='utf-8') as f:
        f.write(retrieved_text)

    return retrieved_text

# tavily search dev serper, ddgs
from ddgs import DDGS

@tool
def web_search(query:str, num_results: int = 10) -> str:
    """Use this tool whenever you need to access realtime or latest information.
        Search the web using DuckDuckGo.
    
    Args:
        query: Search query string
        num_results: Number of results to return (default: 10)
    
    Returns:
        Formatted search results with titles, descriptions, and URLs
    """

    results = DDGS().text(query=query, max_results=num_results, region='us-en')

    if not results:
        return f"No results found for '{query}'"
    
    formatted_results = [f"Search results for search query: '{query}'"]
    for i, result in enumerate(results, 1):
        title = result.get('title', 'No title')
        href = result.get('href', '')
        body = result.get('body', 'No description available')
        
        text = f"{i}. **{title}**\n   {body}\n   {href}"
        
        formatted_results.append(text)

    return "\n\n".join(formatted_results)

