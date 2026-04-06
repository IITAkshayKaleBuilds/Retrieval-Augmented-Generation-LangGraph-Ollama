# ## Advanced RAG Concept - Data Ingestion Pipeline
# ### Page-wise Document Processing with LLM Metadata Extraction
# 
# Learning Objectives:
# - Extract text from PDFs page by page
# - Extract metadata using LLM (structured output)
# - Store in ChromaDB with rich metadata

import os

import hashlib
from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document

from docling.document_converter import DocumentConverter

# ### Metadata Extraction
def extract_metadata_from_filename(filename: str) -> dict:

    name = filename.replace('.pdf', '')   
    parts = name.split()

    metadata = {}
    if len(parts) == 4:
        metadata['fiscal_quarter'] = parts[2]
        metadata['fiscal_year'] = int(parts[3])

    else:
        metadata['fiscal_quarter'] = None
        metadata['fiscal_year'] = int(parts[2])

    metadata['company_name'] = parts[0]
    metadata['doc_type'] = parts[1]

    return metadata

# ### Extract text from each page of PDF
def extract_pdf_pages(pdf_path):

    converter = DocumentConverter()

    result = converter.convert(pdf_path)

    page_break = "<!-- page break -->"

    markdown_text = result.document.export_to_markdown(page_break_placeholder=page_break)

    pages = markdown_text.split(page_break)

    return pages

# """Compute SHA-256 hash of file content."""
def compute_file_hash(file_path: str) -> str:
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

# ### Documents Ingestion in Vector DB

def ingest_docs_in_vectordb(pdf_path):
    print(f"Processing: {pdf_path.name}")

    file_hash = compute_file_hash(pdf_path)
    if file_hash in processed_hashes:
        print(f"[SKIP] already processed: {pdf_path}")
        return 

    pages = extract_pdf_pages(pdf_path)

    print(f"Total pages in {pdf_path.name}: {len(pages)}")

    file_metadata = extract_metadata_from_filename(pdf_path.name)

    processed_pages = []

    for page_num, page_text in enumerate(pages, start=1):
        metadata_dict = file_metadata.copy()
        metadata_dict['page'] = page_num
        metadata_dict['file_hash'] = file_hash
        metadata_dict['source_file'] = pdf_path.name

        doc = Document(page_content=page_text, metadata=metadata_dict)

        processed_pages.append(doc)

    
    vector_store.add_documents(documents=processed_pages)
