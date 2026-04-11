#
from urllib import response

from langchain_core import documents
from pytube import query
from typing_extensions import TypedDict, Annotated
from typing import List
import os
import operator
from scripts.utils import robust_json_parser

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from RAG_Applications.scripts import my_tools

# Load API key from Colab secrets
LANGSMITH_API_KEY = os.environ.get("LANGSMITH_API_KEY")

# Non-sensitive configs
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com/"
os.environ["LANGCHAIN_PROJECT"] = "Retrieval-Augmented-Generation-LangGraph-Ollama"

# =============================================================================
# Configuration
# =============================================================================
DATA_DIR = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/data"
CHROMA_DIR = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/chroma_financial_db"
COLLECTION_NAME = "financial_docs"
EMBEDDING_MODEL = 'qwen3-embedding:4b'
BASE_URL = 'http://127.0.0.1:11434'
LLM_MODEL = "qwen3:14b"
DEBUG_PATH = "/content/Retrieval-Augmented-Generation-LangGraph-Ollama/RAG_Applications/debug_logs"

llm = ChatOllama(model=LLM_MODEL, base_url=BASE_URL, reasoning=True)

# =============================================================================
# Pydantic Schemas for Structured Outputs
# =============================================================================
class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(description="Documents are relevant to the query, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded with the facts for the query, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    """Binary score to assess answer addresses query."""
    binary_score: str = Field(description="Answer addresses the query, 'yes' or 'no'")


class SearchQueries(BaseModel):
    """Search queries for retrieving missing information."""
    search_queries: list[str] = Field(description="1-3 search queries to retrieve the missing information.")


# =============================================================================
# Helper Function
# =============================================================================
def get_latest_user_query(messages:list):

    for message in reversed(messages):
        if isinstance(message, HumanMessage):
            return message.content
        
    return messages[0].content if messages else ''


# =============================================================================
# LangGraph Nodes
# =============================================================================

# Retrieve documents based on user query
def retrieve_node(state):

    print("[RETRIEVE] fetching documents...")

    query = get_latest_user_query(state['messages'])

    rewritten_queries = state.get('rewritten_queries', [])

    # use rewriten queries if present
    queries_to_search = rewritten_queries if rewritten_queries else [query]

    all_results = []
    for idx, search_query in enumerate(queries_to_search, 1):
        print(f"[RETRIEVE] Query {idx}: {search_query}")

        # 3(Reranking) -> 3*10(Retrieval) -> 3*10*20 (MMR)
        result = my_tools.retrieve_docs.invoke({'query': search_query, 'k': 3}) or ""

        text = f"## Query {idx}: {search_query}\n\n### Retrieved Documents:\n{result}"
        all_results.append(text)

    combined_result = "\n\n".join(all_results)

    os.makedirs(DEBUG_PATH, exist_ok=True)
    with open(f"{DEBUG_PATH}/self_rag.md", "w", encoding='utf-8') as f:
        f.write(combined_result)

    return {
        'retrieved_docs': combined_result
    }


# Grade document relevance and filter out irrelevant ones
def grade_documents_node(state):

    print("[GRADE] Evaluating document relevance")

    query = get_latest_user_query(state['messages'])
    documents = state.get('retrieved_docs', 'No document available!')

    system_prompt = """You are a grader assessing relevance of retrieved documents to a user query.

                It does not need to be a stringent test. The goal is to filter out erroneous retrievals.

                If the document contains keyword(s) or semantic meaning related to the user query, grade it as relevant.

                Respond in JSON format.

                Format:
                {
                    "binary_score": "yes" OR "no"
                } 
                to indicate whether the document is relevant to the query."""
    
    system_msg = SystemMessage(system_prompt)

    messages = [system_msg, HumanMessage(f"Retrieved Document: {documents}\n\nUser query: {query}")]

    response = llm.invoke(messages)
    print("RAW GRADE OUTPUT:", response.content)
    parsed = robust_json_parser(response.content)

    if parsed and "binary_score" in parsed:
        score = parsed["binary_score"]
    else:
        print("[GRADE] JSON parsing failed — using fallback heuristic")

        content_lower = response.content.lower()

        if "yes" in content_lower:
            score = "yes"
        elif "no" in content_lower:
            score = "no"
        else:
            score = "yes"   # SAFE fallback

    score = str(score).lower().strip()

    if score == "yes":
        print("[GRADE] Documents are relevant")
        return {'retrieved_docs': documents}

    elif score == "no":
        print("[GRADE] Documents marked NOT relevant — but keeping them for safety")
        return {'retrieved_docs': documents}

    else:
        print("[GRADE] Parsing failed — keeping documents")
        return {'retrieved_docs': documents}

# Generate answer based on retrieved documents
def generate_node(state):
    print("[GENERATE] Creating Answer")

    query = get_latest_user_query(state['messages'])
    documents = state.get('retrieved_docs', '')

    print(f"[GENERATE] Query: {query}")
    print(f"[GENERATE] Raw documents length: {len(documents)}")

    documents = documents[:3000]
    print(f"[GENERATE] Trimmed documents length: {len(documents)}")

    if not documents or documents.strip() == "":
        print("[GENERATE] No documents available for generation")
        return {
            "messages": [AIMessage(content="No relevant information found in the documents.")]
        }

    system_prompt = """You are a helpful financial document analyst.

                STRICT RULES:
                - Keep answer concise and factual
                - Answer ONLY using the provided documents
                - Do NOT use prior knowledge
                - Do NOT assume or infer missing information
                - If answer is found → answer clearly
                - If partially found → answer what is available
                - If NOT found → say: "I could not find this information in the provided documents."
                - DO NOT make up or hallucinate information

                OUTPUT FORMAT:
                Write a comprehensive answer (200-300 words) in MARKDOWN format:
                - Use ## headings for sections
                - Use **bold** for emphasis
                - Use bullet points or numbered lists
                - Include inline citations like [1], [2] where applicable

                GUIDELINES:
                - Base your answer ONLY on the provided documents
                - Be specific with numbers, dates, and metrics
                - If information is missing, acknowledge it
                - Use proper financial terminology

                CITATIONS:
                At the end, list references in this format:
                **References:**
                1. Company: x, Year: y, Quarter: z, Page: n
                
                DO NOT:
                - hallucinate
                - guess
                - add external knowledge
                """
    
    query_prompt = f"Retrieved Document: {documents}\n\nUser query: {query}"

    system_msg = SystemMessage(system_prompt)
    user_msg = HumanMessage(query_prompt)

    messages = [system_msg, user_msg]
    print("[GENERATE] Sending request to LLM...")
    response = llm.invoke(messages)

    os.makedirs(DEBUG_PATH, exist_ok=True)
    with open(f"{DEBUG_PATH}/self_rag_answer.md", "w", encoding='utf-8') as f:
        f.write(f"Query: {query}")
        f.write(response.content)

    if not response or not response.content:
        print("[GENERATE] Empty response from LLM")
        return {
            "messages": [AIMessage(content="Unable to generate answer.")]
        }

    print(f"[GENERATE] Response received (first 200 chars): {response.content[:200]}")

    return {
        "messages": [AIMessage(content=response.content)]
    }

# Transform the query to produce better search queries
def transform_query_node(state):

    print("[TRANSFORM] Rewriting query")
    query = get_latest_user_query(state['messages'])
    print(f"[TRANSFORM] Original query: {query}")
    rewritten_queries = state.get('rewritten_queries', [])

    system_prompt = """You are a query re-writer that decomposes complex queries into focused search queries optimized for vectorstore retrieval.

                DECOMPOSITION STRATEGY:
                Break down the original query into 1-3 specific, focused queries where each query targets:
                - A single company (e.g., "Amazon revenue 2023" vs "Google revenue 2023")
                - A specific time period (e.g., "Q1 2024" vs "Q2 2024")
                - A specific metric or aspect (e.g., "revenue" vs "net income")
                - A specific document section (e.g., "risk factors" vs "business overview")

                GUIDELINES:
                - Expand abbreviations (e.g., "rev" -> "revenue", "GOOGL" -> "Google")
                - Add financial context if missing
                - Make each query self-contained and specific
                - Keep queries concise but clear (5-10 words each)
                - Avoid repeating previously tried queries

                EXAMPLES:
                - "Compare Apple and Google revenue in 2024 Q1" → 
                ["Apple total revenue Q1 2024", "Google total revenue Q1 2024"]
                
                - "Amazon's revenue growth from 2022 to 2024" →
                ["Amazon revenue 2022", "Amazon revenue 2023", "Amazon revenue 2024"]
                
                - "What were the main risks for Microsoft in 2023?" →
                ["Microsoft risk factors 2023", "Microsoft business challenges 2023"]
                Respond in JSON format.

                Format:
                {
                    "search_queries": ["q1", "q2", "q3"]
                }
                """
                

    query_context = f"Original Query: {query}"
    if rewritten_queries:
        query_context = query_context + f"\n\n These queries have been already generated. do not generate same queries again.\n"
        for idx, prev_query in enumerate(rewritten_queries, 1):
            query_context = query_context + f"Query {idx}: {prev_query}\n\n"

    query_context = query_context + "\n\nGenerate 1-3 focused search queries that decompose the original query. Each query should target a specific aspect."

    system_msg = SystemMessage(system_prompt)
    user_msg = HumanMessage(query_context)

    messages = [system_msg, user_msg]
    
    response = llm.invoke(messages)
    print(f"[TRANSFORM] Raw LLM output: {response.content}")

    parsed = robust_json_parser(response.content)

    if parsed and "search_queries" in parsed:
        queries = parsed["search_queries"]

        if not isinstance(queries, list):
            print("search_queries not list, fixing")
            queries = [str(queries)]

    else:
        print("Using fallback query")
        queries = [query]

    print(f"[TRANSFORM] Final queries: {queries}")
    return {
        "rewritten_queries": rewritten_queries + queries,
        "retry_count": state.get("retry_count", 0) + 1
    }

# ### Router Logic

# =============================================================================
# Router Logic
# =============================================================================

# Decide whether to generate answer or transform query
def should_generate(state):
    print("[ROUTER] Assess graded documents")

    retrieved_docs = state.get('retrieved_docs', '')
    retry_count = state.get("retry_count", 0)
    
    if not retrieved_docs or retrieved_docs.strip() == '':

        if retry_count >= 2:
            print("Max retries reached. Forcing final answer generation.")
            return 'generate'   # fallback (prevents infinite loop)

        print(f"[ROUTER] Retry {retry_count + 1} → transforming query")
        return 'transform_query'

    else:
        print('[ROUTER] Have relevant documents - generating answer')
        return 'generate'

# Check for hallucinations and whether answer addresses query
def check_answer_quality(state):
    retry_count = state.get("retry_count", 0)
    query = get_latest_user_query(state['messages'])
    documents = state.get('retrieved_docs', '')
    generation = state['messages'][-1].content

    if "could not find" in generation.lower() or "not found" in generation.lower():
        print("[ROUTER] Valid 'not found' answer — stopping loop")
        return END

    hallucination_prompt = """You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
                            Respond in JSON format.

                            Format:
                            {
                                "binary_score": "yes" OR "no"
                            } 
                            'yes' means that the answer is grounded in / supported by the set of facts."""

    system_msg = SystemMessage(hallucination_prompt)
    user_msg = HumanMessage(f"Set of facts:\n\n{documents}\n\nLLM Generation: {generation}")

    messages = [system_msg, user_msg]
    response = llm.invoke(messages)
    parsed = robust_json_parser(response.content)

    if parsed and "binary_score" in parsed:
        hallucination_grade = parsed["binary_score"]
    else:
        print("[HALLUCINATION] JSON parsing failed — using fallback")
        
        if "yes" in response.content.lower():
            hallucination_grade = "yes"
        elif "no" in response.content.lower():
            hallucination_grade = "no"
        else:
            hallucination_grade = "yes"

    hallucination_grade = str(hallucination_grade).lower().strip()

    # if result is grounded into the facts or retrieved docs
    if hallucination_grade == 'yes':
        # now check answer quality
        print("[ROUTER] Generation is gounded in documents")

        print("[ROUTER] Checking answer quality")
        llm_answer = llm

        answer_prompt = """You are a grader assessing whether an answer addresses / resolves a query.
                        Respond in JSON format.
                        Format:
                        {
                            "binary_score": "yes" OR "no"
                        } 
                        'Yes' means that the answer resolves the query."""

        system_msg = SystemMessage(answer_prompt)

        user_msg = HumanMessage(f"User Query: {query}\n\n LLM Generation: {generation}")

        messages = [system_msg, user_msg]

        answer_response = llm_answer.invoke(messages)
        parsed = robust_json_parser(answer_response.content)

        if parsed and "binary_score" in parsed:
            answer_grade = parsed["binary_score"]
        else:
            print("[ANSWER] JSON parsing failed — using fallback")

            content_lower = answer_response.content.lower()

            if "yes" in content_lower:
                answer_grade = "yes"
            elif "no" in content_lower:
                answer_grade = "no"
            else:
                answer_grade = "yes"

        answer_grade = str(answer_grade).lower().strip()

        if answer_grade=='yes':
            print('[ROUTER] generation is good. - USEFUL')
            return END
        else:
            print("[ROUTER] Generation does not address the query - NOT USEFUL")
            return "transform_query"

    else:
        print("[ROUTER] Generation NOT grounded — stopping to avoid loop")
        return END

