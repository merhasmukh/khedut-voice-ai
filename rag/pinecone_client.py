"""
Pinecone Cloud Vector Database Client for Khedut Voice AI.
=========================================================
Uses Pinecone as the STORAGE layer with Gemini embeddings (768-dim).
This gives the best accuracy for Gujarati text because:
  - Gemini text-embedding-001 has native Gujarati/Indic language support
  - Pinecone provides scalable cloud ANN search (no Docker needed on Pi)

Drop-in alternative to qdrant_client.py. Exposes the same interface:
  - is_pinecone_available()       -> bool
  - upsert_documents_pinecone()   -> int   (takes docs + pre-computed vectors)
  - search_knowledge_pinecone()   -> List  (takes pre-computed query vector)
  - get_pinecone_stats()          -> Dict

Activate by setting in .env:
  VECTOR_STORE=pinecone
  PINECONE_API_KEY=your_key_here
  PINECONE_INDEX=khedut-knowledge
  PINECONE_CLOUD=aws
  PINECONE_REGION=us-east-1
  PINECONE_NAMESPACE=khedut

Index dimension: 768  (matches Gemini text-embedding-001)
Distance metric: cosine
"""

import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Config from env -----------------------------------------------------------
PINECONE_API_KEY   = os.environ.get("PINECONE_API_KEY", "")
PINECONE_INDEX     = os.environ.get("PINECONE_INDEX",   "khedut-knowledge")
PINECONE_CLOUD     = os.environ.get("PINECONE_CLOUD",   "aws")
PINECONE_REGION    = os.environ.get("PINECONE_REGION",  "us-east-1")
PINECONE_NAMESPACE = os.environ.get("PINECONE_NAMESPACE", "khedut")

# Must match Gemini text-embedding-001 output dimension
VECTOR_DIMENSION   = 768
DISTANCE_METRIC    = "cosine"

_pc_client = None
_pc_index  = None


def _get_pinecone():
    """Lazy-init Pinecone client + index (singleton). Auto-creates index if missing."""
    global _pc_client, _pc_index

    if _pc_client is not None and _pc_index is not None:
        return _pc_client, _pc_index

    if not PINECONE_API_KEY:
        raise ValueError(
            "PINECONE_API_KEY is not set. "
            "Add it to your .env: PINECONE_API_KEY=your_key_here"
        )

    from pinecone import Pinecone, ServerlessSpec
    import time

    _pc_client = Pinecone(api_key=PINECONE_API_KEY)

    # Check if index exists and has correct dimension
    recreate = False
    if _pc_client.has_index(PINECONE_INDEX):
        try:
            desc = _pc_client.describe_index(PINECONE_INDEX)
            if desc.dimension != VECTOR_DIMENSION:
                print(f"⚠️ Existing index '{PINECONE_INDEX}' has dimension {desc.dimension} (expected {VECTOR_DIMENSION} for Gemini). Re-creating index...")
                _pc_client.delete_index(PINECONE_INDEX)
                recreate = True
        except Exception as e:
            print(f"⚠️ Error checking index dimension: {e}")

    if not _pc_client.has_index(PINECONE_INDEX) or recreate:
        print(f"📦 Creating Pinecone index '{PINECONE_INDEX}' "
              f"({PINECONE_CLOUD}/{PINECONE_REGION}) "
              f"dim={VECTOR_DIMENSION}, metric={DISTANCE_METRIC}...")
        _pc_client.create_index(
            name=PINECONE_INDEX,
            dimension=VECTOR_DIMENSION,
            metric=DISTANCE_METRIC,
            spec=ServerlessSpec(
                cloud=PINECONE_CLOUD,
                region=PINECONE_REGION,
            ),
        )
        # Wait for index to become ready
        while not _pc_client.describe_index(PINECONE_INDEX).status.state.lower() in ("ready", "available"):
            time.sleep(1)
        print(f"✅ Pinecone index '{PINECONE_INDEX}' created and ready.")

    _pc_index = _pc_client.Index(PINECONE_INDEX)
    return _pc_client, _pc_index


def is_pinecone_available() -> bool:
    """Check whether Pinecone is configured and reachable."""
    if not PINECONE_API_KEY:
        return False
    try:
        _, idx = _get_pinecone()
        idx.describe_index_stats()
        return True
    except Exception:
        return False


def upsert_documents_pinecone(
    documents: List[Dict[str, Any]],
    embeddings: List[List[float]],
) -> int:
    """
    Upsert document chunks with pre-computed Gemini embeddings into Pinecone.
    Uses standard Pinecone upsert (not integrated embedding) to preserve
    full Gujarati text accuracy from Gemini text-embedding-001.

    Args:
        documents:  List of dicts with keys: id, text, title, category,
                    crop, source_file, farmer_name, district, audio_url, ...
        embeddings: Parallel list of 768-dim float vectors from Gemini.

    Returns:
        Number of vectors successfully upserted.
    """
    if not documents or not embeddings or len(documents) != len(embeddings):
        return 0

    try:
        _, idx = _get_pinecone()

        vectors = []
        for doc, emb in zip(documents, embeddings):
            meta = doc.get("metadata", {})
            vectors.append({
                "id":     doc.get("id", ""),
                "values": emb,
                "metadata": {
                    # store everything as metadata for retrieval
                    "text":             doc.get("text", ""),
                    "title":            doc.get("title", ""),
                    "category":         doc.get("category", ""),
                    "crop":             doc.get("crop", "બધા"),
                    "source_file":      doc.get("source_file", ""),
                    "language":         doc.get("language", "gu"),
                    "farmer_name":      meta.get("farmer_name", ""),
                    "district":         meta.get("district", ""),
                    "experience_years": meta.get("experience_years", ""),
                    "audio_url":        meta.get("audio_url", ""),
                    "is_experience":    meta.get("is_experience", False),
                },
            })

        # Upsert in batches of 100 (Pinecone recommended)
        BATCH = 100
        total = 0
        for i in range(0, len(vectors), BATCH):
            batch = vectors[i : i + BATCH]
            idx.upsert(vectors=batch, namespace=PINECONE_NAMESPACE)
            total += len(batch)
            print(f"  ↑ Pinecone upsert batch {i // BATCH + 1}: {len(batch)} vectors")

        return total

    except Exception as exc:
        print(f"⚠️  Pinecone upsert error: {exc}")
        return 0


def search_knowledge_pinecone(
    query_vector: List[float],
    limit: int = 4,
    score_threshold: float = 0.45,
    crop_filter: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Search Pinecone using a pre-computed Gemini query embedding.
    Returns results in the same schema as qdrant_client.search_knowledge().

    Args:
        query_vector:    768-dim Gemini embedding for the user query.
        limit:           Max results to return.
        score_threshold: Minimum cosine similarity score (0–1).
        crop_filter:     Optional Gujarati crop name to filter by.

    Returns:
        List of result dicts: id, score, text, title, category, crop,
        source_file, farmer_name, district, experience_years, audio_url,
        is_experience.
    """
    try:
        _, idx = _get_pinecone()

        # Build optional metadata filter
        query_filter: Optional[Dict] = None
        if crop_filter:
            query_filter = {
                "$or": [
                    {"crop": {"$eq": crop_filter}},
                    {"crop": {"$eq": "બધા"}},
                ]
            }

        kwargs: Dict[str, Any] = {
            "vector":           query_vector,
            "top_k":            limit,
            "namespace":        PINECONE_NAMESPACE,
            "include_metadata": True,
        }
        if query_filter:
            kwargs["filter"] = query_filter

        results = idx.query(**kwargs)

        matches = []
        for hit in results.get("matches", []):
            score = float(hit.get("score", 0.0))
            if score < score_threshold:
                continue
            meta = hit.get("metadata", {})
            matches.append({
                "id":               hit.get("id", ""),
                "score":            round(score, 4),
                "text":             meta.get("text", ""),
                "title":            meta.get("title", ""),
                "category":         meta.get("category", ""),
                "crop":             meta.get("crop", ""),
                "source_file":      meta.get("source_file", ""),
                "farmer_name":      meta.get("farmer_name", ""),
                "district":         meta.get("district", ""),
                "experience_years": meta.get("experience_years", ""),
                "audio_url":        meta.get("audio_url", ""),
                "is_experience":    meta.get("is_experience", False),
            })

        return matches

    except Exception as exc:
        print(f"⚠️  Pinecone search error: {exc}")
        return []


def get_pinecone_stats() -> Dict[str, Any]:
    """Return index statistics matching the same schema as qdrant get_collection_stats."""
    try:
        _, idx = _get_pinecone()
        stats = idx.describe_index_stats()
        ns    = stats.get("namespaces", {}).get(PINECONE_NAMESPACE, {})
        return {
            "status":        "online",
            "index_name":    PINECONE_INDEX,
            "cloud":         PINECONE_CLOUD,
            "region":        PINECONE_REGION,
            "namespace":     PINECONE_NAMESPACE,
            "embed_model":   "gemini-embedding-001 (Gujarati-native)",
            "vector_size":   VECTOR_DIMENSION,
            "distance":      DISTANCE_METRIC,
            "points_count":  ns.get("vector_count", 0),
            "vectors_count": ns.get("vector_count", 0),
            "total_vectors": stats.get("total_vector_count", 0),
        }
    except Exception as exc:
        return {"status": "error", "error": str(exc), "points_count": 0}
