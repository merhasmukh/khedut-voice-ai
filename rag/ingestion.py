"""
Knowledge Ingestion Pipeline for Khedut Voice AI.
Supports loading and chunking from:
- JSON (.json): Structured Gujarati farming guides, formulas, recipes
- PDF (.pdf): Official agricultural university guides, organic reports
- Markdown (.md) & Text (.txt): Documentation and notes

Backend is selected by VECTOR_STORE env var:
  VECTOR_STORE=qdrant    → Gemini embeddings + Qdrant upsert (default)
  VECTOR_STORE=pinecone  → Pinecone integrated embedding upsert (no local GPU)

Includes automatic incremental file change detection (SHA-256 caching).
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

# ── Select active vector backend ──────────────────────────────────────────────
VECTOR_STORE  = os.environ.get("VECTOR_STORE", "qdrant").strip().lower()
_USE_PINECONE = VECTOR_STORE == "pinecone"

from .embeddings import get_embeddings_batch

if _USE_PINECONE:
    from .pinecone_client import upsert_documents_pinecone
else:
    from .qdrant_client import upsert_documents


def get_file_sha256(file_path: Path) -> str:
    """Computes SHA-256 checksum for a file to track changes."""
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def _get_cache_path(folder: Path) -> Path:
    backend = os.environ.get("VECTOR_STORE", "qdrant").strip().lower()
    return folder / f".index_cache_{backend}.json"


def load_index_cache(cache_path: Path) -> Dict[str, str]:
    if cache_path.exists():
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_index_cache(cache_path: Path, cache: Dict[str, str]):
    try:
        with open(cache_path, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        print(f"⚠️ Could not save index cache: {e}")


def split_text_into_chunks(text: str, chunk_size: int = 500, chunk_overlap: int = 80) -> List[str]:
    """
    Split text into overlapping chunks respecting sentence/paragraph boundaries.
    """
    clean_text = text.strip()
    if not clean_text:
        return []

    # If text fits inside one chunk, return directly
    if len(clean_text) <= chunk_size:
        return [clean_text]

    # Split by paragraphs or double newlines first
    paragraphs = [p.strip() for p in clean_text.split("\n\n") if p.strip()]
    chunks = []
    current_chunk = ""

    for p in paragraphs:
        if len(current_chunk) + len(p) + 2 <= chunk_size:
            current_chunk = f"{current_chunk}\n\n{p}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            # If paragraph itself is larger than chunk_size, split by lines/sentences
            if len(p) > chunk_size:
                start = 0
                while start < len(p):
                    end = start + chunk_size
                    chunks.append(p[start:end].strip())
                    start += chunk_size - chunk_overlap
                current_chunk = ""
            else:
                current_chunk = p

    if current_chunk:
        chunks.append(current_chunk)

    return [c for c in chunks if c.strip()]


def load_json_documents(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load structured agricultural knowledge from a JSON file.
    Expected schema:
    [
      {
        "title": "...",
        "category": "...",
        "crop": "...",
        "content": "...",
        "ingredients": [...],
        "method": "...",
        "benefits": "..."
      }
    ]
    or { "items": [...] }
    """
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    items = data if isinstance(data, list) else data.get("items", [data])
    docs = []

    for item_idx, item in enumerate(items):
        title = item.get("title") or item.get("name") or "કૃષિ માર્ગદર્શિકા"
        category = item.get("category") or "પ્રાકૃતિક ખેતી"
        crop = item.get("crop") or "બધા"
        item_id = item.get("id") or f"item_{item_idx}"

        # Build full content string from fields
        content_parts = []
        if item.get("farmer_name"):
            exp_yr = f" (અનુભવ: {item.get('experience_years')})" if item.get("experience_years") else ""
            dist = f" (જિલ્લો: {item.get('district')})" if item.get("district") else ""
            content_parts.append(f"ખેડૂતનું નામ: {item['farmer_name']}{dist}{exp_yr}")
        if item.get("summary"):
            content_parts.append(f"સારાંશ: {item['summary']}")
        if item.get("content"):
            content_parts.append(item["content"])
        if item.get("ingredients"):
            ingr = item["ingredients"]
            ingr_str = ", ".join(ingr) if isinstance(ingr, list) else str(ingr)
            content_parts.append(f"જરૂરી સામગ્રી / ઘટકો: {ingr_str}")
        if item.get("preparation") or item.get("method"):
            content_parts.append(f"બનાવવાની રીત / પદ્ધતિ: {item.get('preparation') or item.get('method')}")
        if item.get("usage") or item.get("application"):
            content_parts.append(f"વાપરવાની રીત / છંટકાવ: {item.get('usage') or item.get('application')}")
        if item.get("benefits"):
            content_parts.append(f"ફાયદા: {item['benefits']}")
        if item.get("advice"):
            content_parts.append(f"ખેડૂતની સલાહ: {item['advice']}")
        if item.get("precautions"):
            content_parts.append(f"સાવચેતી: {item['precautions']}")
        if item.get("audio_url") or item.get("audio_file"):
            content_parts.append(f"ઓડિયો રેકોર્ડિંગ ઉપલબ્ધ: {item.get('audio_url') or item.get('audio_file')}")
        if item.get("pdf_url"):
            content_parts.append(f"સત્તાવાર પુસ્તક PDF ડાઉનલોડ લિંક: {item['pdf_url']}")
        if item.get("video_url"):
            content_parts.append(f"સત્તાવાર YouTube વિડીયો લિંક: {item['video_url']}")
        if item.get("keywords"):
            kw = item["keywords"]
            kw_str = ", ".join(kw) if isinstance(kw, list) else str(kw)
            content_parts.append(f"મુખ્ય વિષયો: {kw_str}")

        full_text = f"### {title} ({category})\n" + "\n\n".join(content_parts)

        # Chunk if needed
        chunks = split_text_into_chunks(full_text, chunk_size=600, chunk_overlap=80)
        for idx, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"khedut:{file_path.name}:{item_id}:{idx}:{chunk[:30]}"))
            docs.append({
                "id": doc_id,
                "text": chunk,
                "title": f"{title} (ભાગ {idx+1})" if len(chunks) > 1 else title,
                "category": category,
                "crop": crop,
                "source_file": file_path.name,
                "language": "gu",
                "metadata": {
                    "raw_item": str(item_id),
                    "farmer_name": item.get("farmer_name", ""),
                    "district": item.get("district", ""),
                    "experience_years": item.get("experience_years", ""),
                    "audio_url": item.get("audio_url") or item.get("audio_file", ""),
                    "is_experience": bool(item.get("audio_url") or item.get("audio_file") or category == "ખેડૂત જાત-અનુભવ"),
                },
            })

    return docs


def load_pdf_documents(file_path: Path) -> List[Dict[str, Any]]:
    """
    Extract text and metadata from PDF files using pypdf.
    """
    reader = PdfReader(str(file_path))
    docs = []
    filename = file_path.name

    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text or not text.strip():
            continue

        chunks = split_text_into_chunks(text, chunk_size=550, chunk_overlap=80)
        for idx, chunk in enumerate(chunks):
            doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"khedut:{filename}:p_{page_num+1}:{idx}:{chunk[:30]}"))
            docs.append({
                "id": doc_id,
                "text": chunk,
                "title": f"{filename} - પેજ {page_num + 1}",
                "category": "પીડીએફ દસ્તાવેજ",
                "crop": "બધા",
                "source_file": filename,
                "language": "gu",
                "metadata": {
                    "page_number": page_num + 1,
                    "chunk_index": idx,
                },
            })

    return docs


def load_text_documents(file_path: Path) -> List[Dict[str, Any]]:
    """
    Load markdown or plain text files.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        text = f.read()

    filename = file_path.name
    chunks = split_text_into_chunks(text, chunk_size=550, chunk_overlap=80)
    docs = []

    for idx, chunk in enumerate(chunks):
        doc_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"khedut:{filename}:{idx}:{chunk[:30]}"))
        docs.append({
            "id": doc_id,
            "text": chunk,
            "title": f"{filename} (ભાગ {idx+1})" if len(chunks) > 1 else filename,
            "category": "દસ્તાવેજ",
            "crop": "બધા",
            "source_file": filename,
            "language": "gu",
            "metadata": {"chunk_index": idx},
        })

    return docs


async def ingest_file(file_path: Path | str) -> int:
    """
    Ingest a single file (.json, .pdf, .md, .txt) into the active vector store.
    - Pinecone: upserts records directly (Pinecone generates embeddings server-side)
    - Qdrant:   generates Gemini embeddings locally, then upserts to Qdrant
    Returns the number of chunks successfully stored.
    """
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"File not found: {p}")

    ext = p.suffix.lower()
    docs: List[Dict[str, Any]] = []

    if ext == ".json":
        docs = load_json_documents(p)
    elif ext == ".pdf":
        docs = load_pdf_documents(p)
    elif ext in (".md", ".txt"):
        docs = load_text_documents(p)
    else:
        print(f"⚠️ Unsupported file type: {ext} (Skipping {p.name})")
        return 0

    if not docs:
        print(f"ℹ️ No text extracted from {p.name}")
        return 0

    backend = "Pinecone" if _USE_PINECONE else "Qdrant"
    print(f"📖 Extracted {len(docs)} chunks from {p.name} → generating Gemini embeddings...")
    texts = [d["text"] for d in docs]
    embeddings = await get_embeddings_batch(texts, batch_size=32)

    if _USE_PINECONE:
        upserted = upsert_documents_pinecone(docs, embeddings)
        print(f"✅ Successfully indexed {upserted} chunks into Pinecone from {p.name}.")
    else:
        upserted = await upsert_documents(docs, embeddings)
        print(f"✅ Successfully indexed {upserted} chunks into Qdrant from {p.name}.")

    return upserted


async def ingest_knowledge_base_directory(dir_path: Path | str = "knowledge_base", force: bool = False) -> Dict[str, int]:
    """
    Scans the knowledge_base directory and incrementally ingests new or modified files.
    - If force=False: Uses SHA-256 caching to skip unchanged files automatically.
    - If force=True: Re-indexes all files regardless of cache.
    """
    folder = Path(dir_path)
    if not folder.exists():
        folder.mkdir(parents=True, exist_ok=True)
        print(f"📁 Created directory: {folder.absolute()}")
        return {}

    cache_file = _get_cache_path(folder)
    cache = {} if force else load_index_cache(cache_file)
    results = {}
    supported_exts = {".json", ".pdf", ".md", ".txt"}

    for file_path in folder.iterdir():
        if file_path.is_file() and not file_path.name.startswith(".") and file_path.suffix.lower() in supported_exts:
            file_name = file_path.name
            current_hash = get_file_sha256(file_path)

            # Check if file has already been indexed with the same content
            if not force and cache.get(file_name) == current_hash:
                results[file_name] = 0  # 0 means unchanged / skipped
                continue

            try:
                print(f"🆕 Found new or modified knowledge file: {file_name}")
                count = await ingest_file(file_path)
                results[file_name] = count
                if count >= 0:
                    cache[file_name] = current_hash
            except Exception as e:
                print(f"❌ Error ingesting {file_name}: {e}")
                results[file_name] = -1

    # Save updated cache
    save_index_cache(cache_file, cache)
    return results
