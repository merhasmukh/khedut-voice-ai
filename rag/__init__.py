"""
RAG (Retrieval-Augmented Generation) package for Khedut Voice AI.
Uses Qdrant vector database and Google Gemini embeddings.
"""

from .embeddings import get_embedding, get_embeddings_batch
from .qdrant_client import (
    get_qdrant_client,
    init_qdrant_collection,
    is_qdrant_available,
    search_knowledge,
    upsert_documents,
    get_collection_stats,
    COLLECTION_NAME,
)
from .retriever import retrieve_relevant_knowledge, build_rag_context
from .ingestion import ingest_file, ingest_knowledge_base_directory

__all__ = [
    "get_embedding",
    "get_embeddings_batch",
    "get_qdrant_client",
    "init_qdrant_collection",
    "is_qdrant_available",
    "search_knowledge",
    "upsert_documents",
    "get_collection_stats",
    "COLLECTION_NAME",
    "retrieve_relevant_knowledge",
    "build_rag_context",
    "ingest_file",
    "ingest_knowledge_base_directory",
]
