import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
from dotenv import load_dotenv

load_dotenv()

from rag.embeddings import get_embedding, get_embeddings_batch, VECTOR_SIZE
from rag.ingestion import split_text_into_chunks, load_json_documents, ingest_file
from rag.retriever import build_rag_context, retrieve_relevant_knowledge
from rag.qdrant_client import is_qdrant_available, init_qdrant_collection, get_collection_stats


@pytest.mark.asyncio
async def test_text_chunking():
    sample_text = "આ એક નાનો ફકરો છે.\n\nઆ બીજો મોટો ફકરો છે જેમાં પ્રાકૃતિક ખેતી વિશે ઘણી બધી માહિતી આપવામાં આવી છે."
    chunks = split_text_into_chunks(sample_text, chunk_size=50, chunk_overlap=10)
    assert len(chunks) >= 1
    assert all(isinstance(c, str) and len(c) > 0 for c in chunks)


@pytest.mark.asyncio
async def test_json_loading():
    json_path = Path("knowledge_base/jeevamrut_and_fertilizers.json")
    assert json_path.exists(), "Seed JSON file should exist"
    docs = load_json_documents(json_path)
    assert len(docs) >= 3
    assert any("જીવામૃત" in d["title"] for d in docs)
    assert all("text" in d and "title" in d for d in docs)


@pytest.mark.asyncio
async def test_gemini_embeddings():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        pytest.skip("GEMINI_API_KEY not set, skipping live embedding test")

    gujarati_query = "કપાસમાં ગુલાબી ઈયળ માટે પ્રાકૃતિક દવા"
    vector = await get_embedding(gujarati_query)
    assert len(vector) == VECTOR_SIZE
    assert isinstance(vector[0], float)

    batch_queries = ["જીવામૃત", "બીજામૃત", "દશપર્ણી અર્ક"]
    batch_vectors = await get_embeddings_batch(batch_queries)
    assert len(batch_vectors) == 3
    assert all(len(v) == VECTOR_SIZE for v in batch_vectors)


@pytest.mark.asyncio
async def test_rag_fallback_when_offline():
    # Context builder should gracefully return empty string or non-failing result if Qdrant is offline
    context = await build_rag_context(query="જીવામૃત કેવી રીતે બનાવવું?", crops=["કપાસ"])
    assert isinstance(context, str)


@pytest.mark.asyncio
async def test_qdrant_if_running():
    available = await is_qdrant_available()
    print(f"\nQdrant Available: {available}")
    if not available:
        print("ℹ️ Qdrant container is not running currently. Skipping live Qdrant test.")
        return

    # If Qdrant is running, test collection init & stats
    initialized = await init_qdrant_collection()
    assert initialized is True

    stats = await get_collection_stats()
    assert stats["status"] == "online"
    assert stats["vector_size"] == VECTOR_SIZE


async def main():
    print("Testing text chunking...")
    await test_text_chunking()
    print("Testing JSON loading...")
    await test_json_loading()
    print("Testing Gemini embeddings...")
    await test_gemini_embeddings()
    print("Testing RAG fallback...")
    await test_rag_fallback_when_offline()
    print("Testing live Qdrant...")
    await test_qdrant_if_running()
    print("\n✅ All RAG tests completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
