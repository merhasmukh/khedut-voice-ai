"""
Integration test: Ingest knowledge_base directory into Qdrant and test semantic search in Gujarati.
"""

import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from rag.ingestion import ingest_knowledge_base_directory
from rag.retriever import retrieve_relevant_knowledge, build_rag_context
from rag.qdrant_client import get_collection_stats


async def main():
    print("🌱 1. Ingesting knowledge_base/ documents into Qdrant...")
    results = await ingest_knowledge_base_directory("knowledge_base")
    print(f"Ingestion results by file: {results}")

    stats = await get_collection_stats()
    print(f"\n📊 2. Qdrant Collection Stats: {stats}")

    print("\n🔍 3. Testing Semantic Search 1: 'જીવામૃત બનાવવાની સાચી રીત'")
    matches1 = await retrieve_relevant_knowledge("જીવામૃત બનાવવાની સાચી રીત", limit=2)
    for i, m in enumerate(matches1):
        print(f"  Match {i+1} [Score: {m['score']:.3f}] - {m['title']}")
        print(f"    Excerpt: {m['text'][:120]}...")

    print("\n🔍 4. Testing Semantic Search 2: 'કપાસમાં ગુલાબી ઈયળ આવી ગઈ છે શું છાંટવું?'")
    matches2 = await retrieve_relevant_knowledge("કપાસમાં ગુલાબી ઈયળ આવી ગઈ છે શું છાંટવું?", limit=2)
    for i, m in enumerate(matches2):
        print(f"  Match {i+1} [Score: {m['score']:.3f}] - {m['title']}")
        print(f"    Excerpt: {m['text'][:120]}...")

    print("\n📝 5. Testing Prompt Context Builder:")
    context = await build_rag_context(query="કપાસમાં ગુલાબી ઈયળ માટે દવા", crops=["કપાસ"])
    print("--- Formatted RAG Context for Gemini Prompt ---")
    print(context)
    print("-----------------------------------------------")

    assert len(matches1) > 0, "Should find Jeevamrut guide"
    assert len(matches2) > 0, "Should find Pink Bollworm guide"
    assert "પ્રમાણિત કૃષિ માર્ગદર્શિકા" in context, "Context should contain header"
    print("\n🎉 End-to-End RAG Test with Qdrant and Gemini Embeddings PASSED!")


if __name__ == "__main__":
    asyncio.run(main())
