"""
RAG Retriever for Khedut Voice AI.
Supports two vector backends — select via .env:
  VECTOR_STORE=qdrant    (default) → Gemini embeddings + local Qdrant
  VECTOR_STORE=pinecone            → Gemini embeddings + Pinecone cloud

Both backends use Gemini text-embedding-001 for full Gujarati accuracy.
Pinecone saves the vectors in the cloud (no Docker / local disk needed).

Falls back to local JSON keyword search if the active backend is unavailable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# Gemini embedding is always used (both Qdrant and Pinecone paths)
from .embeddings import get_embedding

# ── Select active vector backend ──────────────────────────────────────────────
VECTOR_STORE  = os.environ.get("VECTOR_STORE", "qdrant").strip().lower()
_USE_PINECONE = VECTOR_STORE == "pinecone"

if _USE_PINECONE:
    from .pinecone_client import (
        is_pinecone_available     as _is_pinecone_available_sync,
        search_knowledge_pinecone as _search_pinecone,
    )
else:
    from .qdrant_client import is_qdrant_available, search_knowledge



def _search_local_json_knowledge(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    """
    Fallback local search across knowledge_base JSON files when Qdrant is offline.
    Matches keywords, titles, summaries, and content.
    """
    kb_dir = Path("knowledge_base")
    if not kb_dir.exists():
        return []

    q_tokens = [t.lower().strip() for t in query.split() if len(t.strip()) > 1]
    if not q_tokens:
        return []

    matches = []
    for json_file in kb_dir.glob("*.json"):
        if json_file.name.startswith("."):
            continue
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data if isinstance(data, list) else data.get("items", [data])
            for item in items:
                title = item.get("title") or item.get("name") or ""
                summary = item.get("summary") or ""
                content = item.get("content") or ""
                keywords = item.get("keywords") or []
                kw_text = " ".join(keywords) if isinstance(keywords, list) else str(keywords)
                farmer_name = item.get("farmer_name") or ""
                district = item.get("district") or ""
                audio_url = item.get("audio_url") or item.get("audio_file") or ""
                category = item.get("category") or "પ્રાકૃતિક ખેતી"

                full_item_text = f"{title} {summary} {content} {kw_text} {farmer_name} {district} {category}".lower()
                clean_q_lower = query.lower()
                score = 0

                # Full phrase or exact keyword match boost
                for kw in keywords:
                    if str(kw).lower() in clean_q_lower or clean_q_lower in str(kw).lower():
                        score += 10

                if "અનુભવ" in clean_q_lower and ("અનુભવ" in title or "અનુભવ" in category or kw_text):
                    score += 6

                for token in q_tokens:
                    if token in full_item_text:
                        score += 1
                        if token in title.lower() or token in farmer_name.lower() or token in district.lower():
                            score += 3
                        if token in kw_text.lower():
                            score += 4

                if score > 0:
                    text_display = f"### {title} ({category})\n"
                    if farmer_name:
                        text_display += f"ખેડૂતનું નામ: {farmer_name} (જિલ્લો: {district})\n"
                    if summary:
                        text_display += f"સારાંશ: {summary}\n"
                    if content:
                        text_display += f"{content}\n"
                    if audio_url:
                        text_display += f"ઓડિયો રેકોર્ડિંગ URL: {audio_url}\n"

                    matches.append({
                        "id": item.get("id") or str(json_file.name),
                        "score": round(min(0.99, 0.5 + score * 0.15), 4),
                        "text": text_display.strip(),
                        "title": title,
                        "category": category,
                        "crop": item.get("crop", "બધા"),
                        "audio_url": audio_url,
                        "farmer_name": farmer_name,
                        "district": district,
                        "source_file": json_file.name,
                    })
        except Exception:
            continue

    matches.sort(key=lambda x: x.get("score", 0), reverse=True)
    return matches[:limit]


async def retrieve_relevant_knowledge(
    query: str,
    crop_filter: Optional[str] = None,
    limit: int = 3,
    score_threshold: float = 0.50,
) -> List[Dict[str, Any]]:
    """
    Search the active vector store for agricultural knowledge.
    Backend is selected by VECTOR_STORE env var (qdrant | pinecone).
    Falls back to local JSON search if the backend is unavailable.
    """
    clean_query = query.strip()
    if not clean_query:
        return []

    # ── Pinecone backend ──────────────────────────────────────────────────────
    if _USE_PINECONE:
        try:
            if not _is_pinecone_available_sync():
                print("⚠️  Pinecone unavailable — falling back to local JSON search")
                return _search_local_json_knowledge(clean_query, limit=limit)
            query_vector = await get_embedding(clean_query)
            matches = _search_pinecone(
                query_vector=query_vector,
                limit=limit,
                score_threshold=score_threshold,
                crop_filter=crop_filter,
            )
            if not matches:
                return _search_local_json_knowledge(clean_query, limit=limit)
            return matches
        except Exception as exc:
            print(f"⚠️  Pinecone retrieval error — falling back to local search: {exc}")
            return _search_local_json_knowledge(clean_query, limit=limit)

    # ── Qdrant backend (default) ──────────────────────────────────────────────
    if not await is_qdrant_available():
        return _search_local_json_knowledge(clean_query, limit=limit)

    try:
        query_vector = await get_embedding(clean_query)
        matches = await search_knowledge(
            query_vector=query_vector,
            limit=limit,
            score_threshold=score_threshold,
            crop_filter=crop_filter,
        )
        if not matches:
            return _search_local_json_knowledge(clean_query, limit=limit)
        return matches
    except Exception as exc:
        print(f"⚠️  Vector retrieval fallback to local search: {exc}")
        return _search_local_json_knowledge(clean_query, limit=limit)


async def build_rag_context(
    query: str,
    crops: Optional[List[str]] = None,
    max_chunks: int = 3,
) -> str:
    """
    Builds a concise Gujarati knowledge context string from Qdrant vectors.
    Returns an empty string if no relevant knowledge is found or Qdrant is offline.
    """
    if not query.strip() and not crops:
        return ""

    search_query = query.strip()
    if not search_query and crops:
        search_query = " ".join(crops) + " પ્રાકૃતિક ખેતી ઉપાયો ખાતર કીટ નિયંત્રણ"

    # Search with primary crop if available
    crop_filter = crops[0] if crops and len(crops) > 0 else None

    matches = await retrieve_relevant_knowledge(
        query=search_query,
        crop_filter=crop_filter,
        limit=max_chunks,
        score_threshold=0.50,
    )

    if not matches:
        return ""

    context_lines = [
        "### પ્રમાણિત કૃષિ માર્ગદર્શિકા (Verified Agricultural Knowledge Base):"
    ]

    for match in matches:
        title = match.get("title", "")
        text = match.get("text", "").strip()
        category = match.get("category", "")
        header = f"• [{title}]" if title else "•"
        if category and category != "સામાન્ય":
            header += f" ({category})"
        context_lines.append(f"{header}:\n{text}")

    context_lines.append(
        "આ આધારભૂત માહિતીનો ઉપયોગ કરીને ખેડૂતને એકદમ સાચો અને સચોટ જવાબ આપો."
    )

    return "\n\n".join(context_lines)
