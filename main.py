from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
import os

from database.connection import init_db, get_db
from database import crud
from ai_services.gemini_api import handle_gemini_live_session
from rag.ingestion import ingest_knowledge_base_directory
from rag.retriever import retrieve_relevant_knowledge

load_dotenv()

# ── Select active vector backend (Qdrant or Pinecone) ─────────────────────────
_VECTOR_STORE = os.environ.get("VECTOR_STORE", "qdrant").strip().lower()

if _VECTOR_STORE == "pinecone":
    from rag.pinecone_client import (
        is_pinecone_available  as _is_store_available_sync,
        get_pinecone_stats     as _get_store_stats,
    )
    async def _is_store_available():
        return _is_store_available_sync()
    async def _get_vector_stats():
        return _get_store_stats()
    async def _init_store():
        pass   # Pinecone index is auto-created on first upsert
else:
    from rag.qdrant_client import (
        init_qdrant_collection as _init_store,
        is_qdrant_available    as _is_store_available,
        get_collection_stats   as _get_vector_stats,
    )


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database schema
    await init_db()

    # Initialize active vector store and sync knowledge base
    backend_label = "Pinecone ☁️" if _VECTOR_STORE == "pinecone" else "Qdrant 🗄️"
    if await _is_store_available():
        print(f"🧠 {backend_label} Vector Database detected.")
        await _init_store()
        # Automatically sync knowledge base (ingests any new or modified files)
        sync_res = await ingest_knowledge_base_directory("knowledge_base")
        new_files = [f for f, count in sync_res.items() if count > 0]
        stats = await _get_vector_stats()
        if new_files:
            print(f"🌱 Ingested {len(new_files)} new/updated file(s): {', '.join(new_files)}")
        else:
            print("✨ Knowledge base is up to date.")
        print(f"📚 {backend_label} collection ready with {stats.get('points_count', 0)} knowledge vectors.")
    else:
        if _VECTOR_STORE == "pinecone":
            print("⚠️  Pinecone unavailable. Check PINECONE_API_KEY in .env.")
        else:
            print("ℹ️  Qdrant is offline. Start Qdrant Docker (docker compose up -d) for RAG features.")

    print("🚀 Khedut Voice AI backend started.")
    yield
    print("🛑 Khedut Voice AI backend shutting down.")


app = FastAPI(title="Khedut Voice AI", lifespan=lifespan)


# ─── Pydantic Schemas ─────────────────────────────────────────────────────────
class ProfileUpdateRequest(BaseModel):
    name: Optional[str] = None
    village: Optional[str] = None
    district: Optional[str] = None
    land_acres: Optional[float] = None
    crops: Optional[List[str]] = None
    soil_type: Optional[str] = None
    farming_type: Optional[str] = None
    notes: Optional[str] = None


class NewConversationRequest(BaseModel):
    title: Optional[str] = None


# ─── REST Endpoints ───────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/profile")
async def get_profile(db: AsyncSession = Depends(get_db)):
    profile = await crud.get_or_create_default_profile(db)
    return {
        "id": profile.id,
        "name": profile.name,
        "village": profile.village,
        "district": profile.district,
        "land_acres": profile.land_acres,
        "crops": profile.crops,
        "soil_type": profile.soil_type,
        "farming_type": profile.farming_type,
        "notes": profile.notes,
        "updated_at": profile.updated_at.isoformat() if profile.updated_at else None,
    }


@app.post("/api/profile")
async def update_profile(req: ProfileUpdateRequest, db: AsyncSession = Depends(get_db)):
    profile = await crud.get_or_create_default_profile(db)
    data = req.model_dump(exclude_unset=True)
    updated = await crud.update_farmer_profile(db, profile.id, data)
    return {"status": "updated", "profile_id": updated.id}


@app.get("/api/conversations")
async def list_conversations(db: AsyncSession = Depends(get_db)):
    convs = await crud.list_conversations(db, limit=40)
    return [
        {
            "id": c.id,
            "title": c.title,
            "summary": c.summary,
            "message_count": len(c.messages),
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        }
        for c in convs
    ]


@app.post("/api/conversations")
async def create_conversation(req: Optional[NewConversationRequest] = None, db: AsyncSession = Depends(get_db)):
    title = req.title if req else None
    conv = await crud.get_or_create_conversation(db, title=title)
    return {
        "id": conv.id,
        "title": conv.title,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
    }


@app.get("/api/conversations/{conversation_id}")
async def get_conversation_history(conversation_id: str, db: AsyncSession = Depends(get_db)):
    conv = await crud.get_conversation(db, conversation_id)
    if not conv:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {
        "id": conv.id,
        "title": conv.title,
        "summary": conv.summary,
        "created_at": conv.created_at.isoformat() if conv.created_at else None,
        "updated_at": conv.updated_at.isoformat() if conv.updated_at else None,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            for m in conv.messages
        ],
    }


@app.delete("/api/conversations/{conversation_id}")
async def delete_conversation(conversation_id: str, db: AsyncSession = Depends(get_db)):
    deleted = await crud.delete_conversation(db, conversation_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"status": "deleted", "id": conversation_id}


# ─── RAG & Vector Database Endpoints ──────────────────────────────────────────
class RagSearchRequest(BaseModel):
    query: str
    crop: Optional[str] = None
    limit: Optional[int] = 3


@app.get("/api/rag/status")
async def rag_status():
    """Returns active vector store backend and collection stats."""
    stats = await _get_vector_stats()
    stats["vector_store"] = _VECTOR_STORE
    return stats


@app.post("/api/rag/ingest")
async def rag_ingest():
    """
    Trigger re-ingestion of all files in knowledge_base/ into the active vector store.
    Works with both Qdrant and Pinecone — no restart needed.
    """
    if not await _is_store_available():
        store_name = "Pinecone" if _VECTOR_STORE == "pinecone" else "Qdrant"
        raise HTTPException(
            status_code=503,
            detail=f"{store_name} is unavailable. Check your configuration."
        )
    results = await ingest_knowledge_base_directory("knowledge_base", force=True)
    stats   = await _get_vector_stats()
    return {
        "status":           "ingested",
        "vector_store":     _VECTOR_STORE,
        "files":            results,
        "collection_stats": stats,
    }


@app.post("/api/rag/search")
async def rag_search(req: RagSearchRequest):
    """Semantic vector search against the active knowledge base."""
    if not await _is_store_available():
        raise HTTPException(status_code=503, detail="Vector store is unavailable.")
    matches = await retrieve_relevant_knowledge(
        query=req.query,
        crop_filter=req.crop,
        limit=req.limit or 3,
    )
    return {
        "query":         req.query,
        "vector_store":  _VECTOR_STORE,
        "matches_count": len(matches),
        "results":       matches,
    }


# ─── Experience Audio Endpoint ────────────────────────────────────────────────
@app.get("/api/experiences/audio/{filename}")
async def get_experience_audio(filename: str):
    """Serves recorded farmer experience audio files."""
    audio_dir = Path("audio_experiences")
    audio_path = audio_dir / filename
    if not audio_path.exists():
        # Check if .wav exists if .mp3 requested or vice versa
        alt_wav = audio_dir / filename.replace(".mp3", ".wav")
        if alt_wav.exists():
            return FileResponse(alt_wav, media_type="audio/wav")
        raise HTTPException(status_code=404, detail=f"Audio file '{filename}' not found in audio_experiences directory.")
    media_type = "audio/mpeg" if audio_path.suffix.lower() == ".mp3" else "audio/wav"
    return FileResponse(audio_path, media_type=media_type)


# ─── Avatar Endpoints ─────────────────────────────────────────────────────────
@app.get("/farmer.webp")
async def get_farmer_image():
    """Serves the farmer portrait image."""
    img_path = Path("farmer.webp")
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="farmer.webp not found")
    return FileResponse(img_path, media_type="image/webp")


@app.get("/api/avatar-video")
async def get_avatar_video():
    """Serves the farmer avatar video."""
    video_path = Path("V1.mp4")
    if not video_path.exists():
        raise HTTPException(status_code=404, detail="Avatar video not found")
    return FileResponse(video_path, media_type="video/mp4")


# ─── WebSocket Endpoint ───────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(browser_ws: WebSocket, conversation_id: Optional[str] = Query(None)):
    await browser_ws.accept()
    print(f"🔌 Browser connected (requested session: {conversation_id})")
    await handle_gemini_live_session(browser_ws, conversation_id=conversation_id)


# ─── HTML Frontend UI ─────────────────────────────────────────────────────────
HTML = """
<!DOCTYPE html>
<html lang="gu">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no, viewport-fit=cover"/>
  <meta name="theme-color" content="#2e7d32"/>
  <meta name="apple-mobile-web-app-capable" content="yes"/>
  <meta name="apple-mobile-web-app-status-bar-style" content="default"/>
  <title>ખેડૂત Voice AI — પ્રાકૃતિક ખેતી સહાયક</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; -webkit-tap-highlight-color: transparent; }
    
    :root {
      --primary-green: #2e7d32;
      --primary-dark: #1b5e20;
      --accent-green: #43a047;
      --light-green: #e8f5e9;
      --bg-gradient: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
      --card-bg: #ffffff;
      --text-dark: #1f2d20;
      --text-muted: #558b2f;
      --border-color: #dcedc8;
      --safe-bottom: env(safe-area-inset-bottom, 16px);
    }

    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      background: var(--bg-gradient);
      min-height: 100vh;
      min-height: 100dvh;
      display: flex;
      flex-direction: row;
      color: var(--text-dark);
      overflow-x: hidden;
    }

    /* ── Mobile Top Navbar (Hidden on Desktop) ── */
    #mobileNavbar {
      display: none;
      position: fixed;
      top: 0;
      left: 0;
      right: 0;
      height: 58px;
      background: #ffffff;
      border-bottom: 1px solid var(--border-color);
      box-shadow: 0 2px 10px rgba(0,0,0,0.05);
      z-index: 900;
      align-items: center;
      justify-content: space-between;
      padding: 0 14px;
    }
    .nav-btn {
      background: #f1f8e9;
      border: 1px solid var(--border-color);
      color: var(--primary-dark);
      width: 40px;
      height: 40px;
      border-radius: 10px;
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 1.2rem;
      cursor: pointer;
      transition: all 0.2s;
    }
    .nav-btn:active {
      background: #c8e6c9;
      transform: scale(0.95);
    }
    .nav-title {
      font-weight: 700;
      font-size: 1.05rem;
      color: var(--primary-dark);
      display: flex;
      align-items: center;
      gap: 6px;
    }
    .nav-new-btn {
      background: var(--primary-green);
      color: #fff;
      border: none;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 0.82rem;
      font-weight: 600;
      display: flex;
      align-items: center;
      gap: 4px;
      cursor: pointer;
    }

    /* ── Sidebar Overlay for Mobile Drawer ── */
    #sidebarOverlay {
      display: none;
      position: fixed;
      inset: 0;
      background: rgba(0, 0, 0, 0.45);
      backdrop-filter: blur(3px);
      z-index: 998;
      opacity: 0;
      transition: opacity 0.3s ease;
    }
    #sidebarOverlay.active {
      display: block;
      opacity: 1;
    }

    /* ── Sidebar Drawer ── */
    #sidebar {
      width: 290px;
      background: #ffffff;
      border-right: 1px solid var(--border-color);
      display: flex;
      flex-direction: column;
      height: 100vh;
      height: 100dvh;
      box-shadow: 2px 0 14px rgba(0,0,0,0.03);
      transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
      z-index: 1000;
      flex-shrink: 0;
    }
    .sidebar-header {
      padding: 18px 16px;
      border-bottom: 1px solid #e8f5e9;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .sidebar-header-top {
      display: flex;
      align-items: center;
      justify-content: space-between;
    }
    .sidebar-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      color: var(--primary-dark);
      font-size: 1.08rem;
    }
    .sidebar-close-btn {
      display: none;
      background: transparent;
      border: none;
      font-size: 1.3rem;
      color: #666;
      cursor: pointer;
      padding: 4px 8px;
      border-radius: 6px;
    }
    .sidebar-close-btn:active {
      background: #f0f0f0;
    }
    .new-chat-btn {
      background: var(--primary-green);
      color: white;
      border: none;
      padding: 10px 14px;
      border-radius: 10px;
      cursor: pointer;
      font-weight: 600;
      font-size: 0.9rem;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      transition: background 0.2s, transform 0.1s;
    }
    .new-chat-btn:hover { background: var(--primary-dark); }
    .new-chat-btn:active { transform: scale(0.98); }
    
    .conv-list {
      flex: 1;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      padding: 12px 10px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .conv-item {
      padding: 10px 12px;
      border-radius: 8px;
      cursor: pointer;
      font-size: 0.88rem;
      color: #33691e;
      background: #f9fbf9;
      border: 1px solid transparent;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: all 0.2s;
    }
    .conv-item:hover {
      background: #e8f5e9;
      border-color: #c8e6c9;
    }
    .conv-item:active {
      background: #c8e6c9;
    }
    .conv-item.active {
      background: #c8e6c9;
      color: var(--primary-dark);
      font-weight: 600;
    }
    .conv-title {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 200px;
    }

    /* ── Main Content Area ── */
    #main-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px 20px;
      min-height: 100vh;
      min-height: 100dvh;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      position: relative;
    }
    .card {
      background: var(--card-bg);
      border-radius: 24px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.06);
      padding: 28px 34px;
      text-align: center;
      max-width: 520px;
      width: 100%;
      border: 1px solid rgba(46, 125, 50, 0.12);
      margin: auto 0;
      transition: padding 0.3s ease;
    }
    .avatar-wrapper {
      display: flex;
      justify-content: center;
      margin-bottom: 14px;
    }
    .avatar-box {
      position: relative;
      width: 230px;
      height: 300px;
      border-radius: 20px;
      overflow: hidden;
      border: 3px solid #81c784;
      background: #112211;
      box-shadow: 0 8px 26px rgba(0, 0, 0, 0.12);
      transition: all 0.3s ease;
    }
    .avatar-box video {
      width: 100%;
      height: 100%;
      object-fit: cover;
      object-position: center top;
      display: block;
      border-radius: 17px;
    }
    .avatar-box.speaking {
      border-color: var(--primary-green);
      box-shadow: 0 0 28px rgba(46, 125, 50, 0.7), 0 0 10px rgba(46, 125, 50, 0.35);
      animation: speakingPulse 1.2s infinite alternate ease-in-out;
    }
    .avatar-box.listening {
      border-color: #e53935;
      box-shadow: 0 0 20px rgba(229, 57, 53, 0.45);
    }
    @keyframes speakingPulse {
      0% {
        box-shadow: 0 0 14px rgba(46, 125, 50, 0.4);
        border-color: #43a047;
      }
      100% {
        box-shadow: 0 0 32px rgba(46, 125, 50, 0.85), 0 0 12px rgba(76, 175, 80, 0.6);
        border-color: var(--primary-dark);
      }
    }
    .avatar-sound-waves {
      position: absolute;
      top: 10px;
      right: 10px;
      display: none;
      align-items: flex-end;
      gap: 3px;
      height: 18px;
      background: rgba(0, 0, 0, 0.65);
      padding: 4px 8px;
      border-radius: 12px;
      backdrop-filter: blur(4px);
      z-index: 2;
    }
    .avatar-box.speaking .avatar-sound-waves {
      display: flex;
    }
    .avatar-sound-waves span {
      width: 3px;
      background: #66bb6a;
      border-radius: 2px;
      animation: waveAnim 0.6s infinite ease-in-out alternate;
    }
    .avatar-sound-waves span:nth-child(1) { height: 6px; animation-delay: 0.1s; }
    .avatar-sound-waves span:nth-child(2) { height: 14px; animation-delay: 0.25s; }
    .avatar-sound-waves span:nth-child(3) { height: 10px; animation-delay: 0.15s; }
    .avatar-sound-waves span:nth-child(4) { height: 16px; animation-delay: 0.35s; }
    @keyframes waveAnim {
      0% { height: 4px; }
      100% { height: 16px; }
    }
    .avatar-indicator {
      position: absolute;
      bottom: 10px;
      left: 50%;
      transform: translateX(-50%);
      background: rgba(0, 0, 0, 0.75);
      color: #ffffff;
      font-size: 0.76rem;
      padding: 4px 12px;
      border-radius: 20px;
      backdrop-filter: blur(6px);
      white-space: nowrap;
      font-weight: 600;
      letter-spacing: 0.2px;
      border: 1px solid rgba(255, 255, 255, 0.2);
      pointer-events: none;
      z-index: 2;
    }
    h1 { color: var(--primary-dark); font-size: 1.45rem; font-weight: 700; margin-bottom: 2px; }
    .subtitle { color: var(--text-muted); font-size: 0.88rem; margin-bottom: 12px; }
    
    .status-row {
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin-bottom: 14px;
      background: #f4fbf5;
      padding: 6px 14px;
      border-radius: 50px;
      border: 1px solid #e0ede0;
      max-width: 100%;
    }
    .dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #bbb;
      flex-shrink: 0;
      transition: all 0.3s ease;
    }
    .dot.connected { background: #43a047; box-shadow: 0 0 8px #43a047; animation: pulse 1.4s infinite; }
    .dot.listening { background: #e53935; box-shadow: 0 0 10px #e53935; animation: pulse 0.8s infinite; }
    .dot.speaking  { background: #1e88e5; box-shadow: 0 0 10px #1e88e5; animation: pulse 0.9s infinite; }
    
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.35); opacity: 0.6; }
    }
    #statusText { color: var(--primary-green); font-size: 0.85rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }

    canvas#waveform {
      width: 100%;
      height: 48px;
      border-radius: 12px;
      background: #f9fbf9;
      margin-bottom: 16px;
      display: block;
      border: 1px solid #e0ede0;
    }

    .btn-row {
      display: flex;
      gap: 10px;
      justify-content: center;
      width: 100%;
    }
    button.action-btn {
      flex: 1;
      padding: 13px 20px;
      font-size: 0.96rem;
      border: none;
      border-radius: 50px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 6px;
      min-height: 48px;
    }
    #startBtn {
      background: var(--primary-green);
      color: #fff;
      box-shadow: 0 4px 16px rgba(46, 125, 50, 0.3);
    }
    #startBtn:hover:not(:disabled) {
      background: var(--primary-dark);
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(46, 125, 50, 0.4);
    }
    #startBtn:active:not(:disabled) {
      transform: scale(0.98);
    }
    #stopBtn {
      background: #e53935;
      color: #fff;
      box-shadow: 0 4px 16px rgba(229, 57, 53, 0.25);
    }
    #stopBtn:hover:not(:disabled) {
      background: #c62828;
      transform: translateY(-2px);
    }
    #stopBtn:active:not(:disabled) {
      transform: scale(0.98);
    }
    button:disabled {
      background: #e0e0e0;
      color: #9e9e9e;
      box-shadow: none;
      cursor: not-allowed;
      transform: none;
    }

    #transcriptBox {
      margin-top: 16px;
      max-height: 160px;
      overflow-y: auto;
      -webkit-overflow-scrolling: touch;
      text-align: left;
      background: #fdfefd;
      border: 1px solid #c8e6c9;
      border-radius: 14px;
      padding: 12px 14px;
      font-size: 0.88rem;
      color: var(--primary-dark);
      line-height: 1.55;
    }
    .rag-badge {
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 4px 12px;
      border-radius: 20px;
      font-size: 0.76rem;
      font-weight: 500;
      background: #e8f5e9;
      color: var(--primary-green);
      border: 1px solid #c8e6c9;
      margin-bottom: 10px;
    }
    .rag-badge.offline {
      background: #fff3e0;
      color: #e65100;
      border-color: #ffe0b2;
    }
    .transcript-placeholder { color: #81c784; font-style: italic; }
    .msg-user { color: #1565c0; font-weight: 500; margin-bottom: 6px; }
    .msg-ai { color: var(--primary-green); font-weight: 500; margin-bottom: 6px; }
    .err { color: #c62828; font-size: 0.82rem; margin-top: 8px; }

    /* ── Experience Audio Banner ── */
    #experienceAudioBanner {
      display: none;
      background: linear-gradient(135deg, #e8f5e9 0%, #dcedc8 100%);
      border: 2px solid #81c784;
      border-radius: 16px;
      padding: 12px 14px;
      margin-bottom: 14px;
      text-align: left;
      animation: expFadeIn 0.3s ease;
      box-shadow: 0 4px 14px rgba(46, 125, 50, 0.12);
    }
    @keyframes expFadeIn {
      from { opacity: 0; transform: translateY(-6px); }
      to { opacity: 1; transform: translateY(0); }
    }
    .exp-badge {
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: var(--primary-dark);
      color: #ffffff;
      font-size: 0.72rem;
      font-weight: 700;
      padding: 3px 10px;
      border-radius: 12px;
      margin-bottom: 6px;
    }
    .exp-title {
      font-size: 0.95rem;
      font-weight: 700;
      color: var(--primary-dark);
      margin-bottom: 2px;
    }
    .exp-desc {
      font-size: 0.8rem;
      color: #33691e;
      margin-bottom: 8px;
      line-height: 1.35;
    }
    .exp-player-row {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 8px;
    }
    .exp-stop-btn {
      background: #e53935;
      color: white;
      border: none;
      padding: 7px 14px;
      border-radius: 20px;
      font-size: 0.8rem;
      font-weight: 600;
      cursor: pointer;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      transition: background 0.2s;
    }
    .exp-stop-btn:hover { background: #c62828; }

    /* ── Responsive Breakpoints ── */
    @media (max-width: 768px) {
      body {
        flex-direction: column;
      }
      #mobileNavbar {
        display: flex;
      }
      #sidebar {
        position: fixed;
        top: 0;
        left: 0;
        height: 100vh;
        height: 100dvh;
        width: min(300px, 84vw);
        transform: translateX(-100%);
        box-shadow: 4px 0 24px rgba(0, 0, 0, 0.2);
      }
      #sidebar.open {
        transform: translateX(0);
      }
      .sidebar-close-btn {
        display: block;
      }
      #main-container {
        padding: 72px 14px calc(var(--safe-bottom) + 14px) 14px;
        min-height: 100vh;
        min-height: 100dvh;
        justify-content: flex-start;
      }
      .card {
        padding: 20px 16px;
        border-radius: 20px;
        margin: 0 auto;
        box-shadow: 0 6px 24px rgba(0,0,0,0.05);
      }
      .avatar-box {
        width: min(200px, 52vw);
        height: min(260px, 68vw);
        border-radius: 18px;
      }
      .avatar-box video {
        border-radius: 15px;
      }
      h1 {
        font-size: 1.3rem;
      }
      .subtitle {
        font-size: 0.82rem;
        margin-bottom: 10px;
      }
      #transcriptBox {
        max-height: 140px;
        font-size: 0.85rem;
      }
    }

    @media (max-width: 380px) {
      .card {
        padding: 16px 12px;
      }
      .avatar-box {
        width: 170px;
        height: 220px;
      }
      button.action-btn {
        padding: 11px 14px;
        font-size: 0.9rem;
      }
    }
  </style>
</head>
<body>

  <!-- Mobile Top Navbar -->
  <header id="mobileNavbar">
    <button class="nav-btn" id="menuToggleBtn" aria-label="ઓપન મેનૂ" title="વાતચીત ઇતિહાસ">☰</button>
    <div class="nav-title">🌱 ખેડૂત Voice AI</div>
    <button class="nav-new-btn" id="mobileNewChatBtn">➕ નવી</button>
  </header>

  <!-- Sidebar Overlay -->
  <div id="sidebarOverlay"></div>

  <!-- Sidebar (Desktop Fixed / Mobile Drawer) -->
  <aside id="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-header-top">
        <div class="sidebar-title">🌱 ખેડૂત AI સત્રો</div>
        <button class="sidebar-close-btn" id="sidebarCloseBtn" aria-label="બંધ કરો">✕</button>
      </div>
      <button class="new-chat-btn" id="newChatBtn">➕ નવી વાતચીત શરૂ કરો</button>
    </div>
    <div class="conv-list" id="convList">
      <!-- Conversation history dynamically loaded here -->
    </div>
  </aside>

  <!-- Main Container -->
  <main id="main-container">
    <div class="card">
      <div class="avatar-wrapper">
        <div class="avatar-box" id="avatarBox">
          <video id="farmerVideo" src="/api/avatar-video" playsinline muted preload="auto"></video>
          <div class="avatar-sound-waves" id="soundWaves">
            <span></span><span></span><span></span><span></span>
          </div>
          <div class="avatar-indicator" id="avatarIndicator">🌾 કિસાન મિત્ર</div>
        </div>
      </div>
      <h1 id="headerTitle">ખેડૂત Voice AI</h1>
      <p class="subtitle">પ્રાકૃતિક ખેતી માટે તમારો ડિજિટલ સાથી</p>
      <div id="ragBadge" class="rag-badge">🧠 RAG: તપાસી રહ્યું છે...</div>
      
      <div class="status-row">
        <div class="dot" id="dot"></div>
        <span id="statusText">વાતચીત શરૂ કરવા 'Start' દબાવો</span>
      </div>
      
      <!-- Farmer Experience Audio Banner -->
      <div id="experienceAudioBanner">
        <span class="exp-badge">🎙️ ખેડૂત જાત-અનુભવ (Recorded Audio)</span>
        <div class="exp-title" id="expFarmerTitle">હસમુખભાઈ પટેલ (વલસાડ)</div>
        <div class="exp-desc" id="expFarmerDesc">૫ વર્ષથી પ્રાકૃતિક/ઓર્ગેનિક ખેતીનો સફળ જાત-અનુભવ</div>
        <div class="exp-player-row">
          <audio id="expAudioElement" preload="auto"></audio>
          <button type="button" class="exp-stop-btn" id="expStopBtn">■ ઓડિયો બંધ કરો</button>
        </div>
      </div>
      
      <canvas id="waveform" width="440" height="48"></canvas>
      
      <div class="btn-row">
        <button class="action-btn" id="startBtn">▶ Start વાતચીત</button>
        <button class="action-btn" id="stopBtn" disabled>■ Stop</button>
      </div>
      
      <div id="transcriptBox">
        <div id="transcript" class="transcript-placeholder">AI નો જવાબ અહીં દેખાશે... તમે ગુજરાતીમાં બોલી શકો છો!</div>
      </div>
      <div class="err" id="errMsg"></div>
    </div>
  </main>

<script>
const SEND_RATE    = 16000;
const RECEIVE_RATE = 24000;

const startBtn          = document.getElementById('startBtn');
const stopBtn           = document.getElementById('stopBtn');
const statusText        = document.getElementById('statusText');
const dot               = document.getElementById('dot');
const transcript        = document.getElementById('transcript');
const transcriptBox     = document.getElementById('transcriptBox');
const errMsg            = document.getElementById('errMsg');
const canvas            = document.getElementById('waveform');
const ctx               = canvas.getContext('2d');
const convList          = document.getElementById('convList');
const newChatBtn        = document.getElementById('newChatBtn');
const mobileNewChatBtn  = document.getElementById('mobileNewChatBtn');
const headerTitle       = document.getElementById('headerTitle');
const farmerVideo       = document.getElementById('farmerVideo');
const avatarBox         = document.getElementById('avatarBox');
const avatarIndicator   = document.getElementById('avatarIndicator');
const sidebar           = document.getElementById('sidebar');
const sidebarOverlay    = document.getElementById('sidebarOverlay');
const menuToggleBtn     = document.getElementById('menuToggleBtn');
const sidebarCloseBtn   = document.getElementById('sidebarCloseBtn');
const expBanner         = document.getElementById('experienceAudioBanner');
const expFarmerTitle    = document.getElementById('expFarmerTitle');
const expFarmerDesc     = document.getElementById('expFarmerDesc');
const expAudioElement   = document.getElementById('expAudioElement');
const expStopBtn        = document.getElementById('expStopBtn');
const START_TIME_SEC    = 2.0;  // Start time for farmer video loop

// ── Experience Audio Controller ─────────────────────────────────────────────
let pendingExperienceAudio = null;
let expAudioSafetyTimer = null;

function startExperienceAudio(msg) {
  if (stopped || !msg) return;
  pendingExperienceAudio = null;
  clearTimeout(expAudioSafetyTimer);

  if (expFarmerTitle) expFarmerTitle.textContent = `${msg.farmer_name || 'હસમુખભાઈ પટેલ'} (${msg.district || 'વલસાડ'})`;
  if (expFarmerDesc) expFarmerDesc.textContent = `${msg.experience_years || '૫ વર્ષ'}થી સફળ પ્રાકૃતિક/ઓર્ગેનિક ખેતીનો જાત-અનુભવ`;
  if (expAudioElement) {
    expAudioElement.src = msg.audio_url || '/api/experiences/audio/valsad_asmukhbhai_experience.mp3';
    if (expBanner) expBanner.style.display = 'block';
    expAudioElement.play().catch(e => console.warn('Audio play notice:', e));
  }
  setStatus('🎙️ ખેડૂત અનુભવ વાગી રહ્યો છે...', 'speaking');
  setAvatarState('speaking');
}

function queueOrPlayExperienceAudio(msg) {
  // Check if AI is currently playing or scheduled to play PCM audio
  const isAiSpeaking = activeSources.size > 0 || (recvCtx && recvCtx.currentTime < nextPlay - 0.15);
  if (isAiSpeaking) {
    console.log('⏳ AI is speaking intro speech. Queuing experience audio to play after AI finishes...');
    pendingExperienceAudio = msg;
    if (expFarmerTitle) expFarmerTitle.textContent = `${msg.farmer_name || 'હસમુખભાઈ પટેલ'} (${msg.district || 'વલસાડ'})`;
    if (expFarmerDesc) expFarmerDesc.textContent = `AI નો પરિચય પૂરો થતાં જ ઓડિયો શરૂ થશે...`;
    if (expBanner) expBanner.style.display = 'block';

    clearTimeout(expAudioSafetyTimer);
    const delaySec = Math.max(1.0, (nextPlay - (recvCtx ? recvCtx.currentTime : 0)) + 0.4);
    expAudioSafetyTimer = setTimeout(() => {
      if (pendingExperienceAudio && activeSources.size === 0) {
        startExperienceAudio(pendingExperienceAudio);
      }
    }, delaySec * 1000);
  } else {
    startExperienceAudio(msg);
  }
}

function stopExperienceAudio() {
  pendingExperienceAudio = null;
  clearTimeout(expAudioSafetyTimer);
  if (expAudioElement) {
    try {
      expAudioElement.pause();
      expAudioElement.currentTime = 0;
    } catch(_) {}
  }
  if (expBanner) {
    expBanner.style.display = 'none';
  }
}

if (expStopBtn) {
  expStopBtn.onclick = () => {
    stopExperienceAudio();
    if (ws && ws.readyState === WebSocket.OPEN && !stopped) {
      setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
      setAvatarState('listening');
    }
  };
}

if (expAudioElement) {
  expAudioElement.onended = () => {
    if (expBanner) expBanner.style.display = 'none';
    if (ws && ws.readyState === WebSocket.OPEN && !stopped) {
      setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
      setAvatarState('listening');
    }
  };
}

// ── Mobile Drawer Open/Close Logic ──────────────────────────────────────────
function openSidebar() {
  sidebar.classList.add('open');
  sidebarOverlay.classList.add('active');
}
function closeSidebar() {
  sidebar.classList.remove('open');
  sidebarOverlay.classList.remove('active');
}
if (menuToggleBtn) menuToggleBtn.onclick = openSidebar;
if (sidebarCloseBtn) sidebarCloseBtn.onclick = closeSidebar;
if (sidebarOverlay) sidebarOverlay.onclick = closeSidebar;

// ── Responsive Canvas DPI Scaling ───────────────────────────────────────────
function resizeCanvas() {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;
  ctx.scale(dpr, dpr);
  drawFlat();
}
window.addEventListener('resize', resizeCanvas);

if (farmerVideo) {
  farmerVideo.addEventListener('loadedmetadata', () => {
    farmerVideo.currentTime = START_TIME_SEC;
  });
  farmerVideo.addEventListener('timeupdate', () => {
    if (farmerVideo.currentTime >= farmerVideo.duration - 0.2) {
      farmerVideo.currentTime = START_TIME_SEC;
      if (avatarBox && avatarBox.classList.contains('speaking')) {
        farmerVideo.play().catch(() => {});
      }
    }
  });
  farmerVideo.addEventListener('ended', () => {
    farmerVideo.currentTime = START_TIME_SEC;
    if (avatarBox && avatarBox.classList.contains('speaking')) {
      farmerVideo.play().catch(() => {});
    }
  });
}

let ws, sendCtx, recvCtx, micStream, workletNode, analyser;
let nextPlay = 0, animId, stopped = false;
let currentConversationId = null;
const activeSources = new Set();

function setAvatarState(state) {
  if (!avatarBox || !avatarIndicator) return;
  if (state === 'speaking') {
    avatarBox.className = 'avatar-box speaking';
    avatarIndicator.textContent = '🔊 જવાબ આપે છે...';
    if (farmerVideo) {
      if (farmerVideo.currentTime < START_TIME_SEC || farmerVideo.currentTime >= farmerVideo.duration - 0.2) {
        farmerVideo.currentTime = START_TIME_SEC;
      }
      if (farmerVideo.paused) {
        farmerVideo.play().catch(() => {});
      }
    }
  } else if (state === 'listening') {
    avatarBox.className = 'avatar-box listening';
    avatarIndicator.textContent = '🎤 સાંભળી રહ્યા છે...';
    if (farmerVideo && !farmerVideo.paused) {
      farmerVideo.pause();
    }
  } else {
    avatarBox.className = 'avatar-box';
    avatarIndicator.textContent = '🌾 કિસાન મિત્ર';
    if (farmerVideo) {
      if (!farmerVideo.paused) farmerVideo.pause();
      farmerVideo.currentTime = START_TIME_SEC;
    }
  }
}

function stopAllAudioPlayback() {
  if (activeSources.size > 0) {
    for (const src of activeSources) {
      try {
        src.stop(0);
        src.disconnect();
      } catch(_) {}
    }
    activeSources.clear();
  }
  if (recvCtx) {
    nextPlay = recvCtx.currentTime;
  }
  if (farmerVideo && !farmerVideo.paused) {
    farmerVideo.pause();
  }
  setAvatarState(ws && !stopped ? 'listening' : 'idle');
}

function setStatus(txt, cls){
  statusText.textContent = txt;
  dot.className = 'dot ' + (cls || '');
}
function showErr(msg){
  errMsg.textContent = msg;
  setTimeout(() => errMsg.textContent = '', 6000);
}

// ── Load Conversation History ───────────────────────────────────────────────
async function loadConversations() {
  try {
    const res = await fetch('/api/conversations');
    const data = await res.json();
    convList.innerHTML = '';
    if (data.length === 0) {
      convList.innerHTML = '<div style="color:#888;font-size:0.85rem;padding:10px;">હજુ સુધી કોઈ વાતચીત નથી.</div>';
      return;
    }
    data.forEach(c => {
      const item = document.createElement('div');
      item.className = 'conv-item' + (c.id === currentConversationId ? ' active' : '');
      item.innerHTML = `<div class="conv-title" title="${c.title}">${c.title}</div><span style="font-size:0.75rem;color:#888;">${c.message_count}</span>`;
      item.onclick = () => {
        selectConversation(c.id);
        closeSidebar();
      };
      convList.appendChild(item);
    });
  } catch (err) {
    console.error('Failed to load conversations:', err);
  }
}

async function selectConversation(id) {
  if (ws && ws.readyState < 2) cleanup();
  currentConversationId = id;
  try {
    const res = await fetch(`/api/conversations/${id}`);
    const data = await res.json();
    headerTitle.textContent = data.title || 'ખેડૂત Voice AI';
    transcript.className = '';
    transcript.innerHTML = '';
    if (data.messages && data.messages.length > 0) {
      data.messages.forEach(m => {
        const p = document.createElement('div');
        p.className = m.role === 'user' ? 'msg-user' : 'msg-ai';
        p.textContent = (m.role === 'user' ? '👤 તમે: ' : '🌾 AI: ') + m.content;
        transcript.appendChild(p);
      });
    } else {
      transcript.className = 'transcript-placeholder';
      transcript.textContent = 'નવી વાતચીત શરૂ કરવા Start દબાવો...';
    }
    loadConversations();
  } catch (err) {
    console.error('Error fetching conversation history:', err);
  }
}

async function createNewChat() {
  if (ws && ws.readyState < 2) cleanup();
  closeSidebar();
  try {
    const res = await fetch('/api/conversations', { method: 'POST' });
    const data = await res.json();
    currentConversationId = data.id;
    headerTitle.textContent = 'નવી ખેતી વાતચીત';
    transcript.className = 'transcript-placeholder';
    transcript.textContent = 'AI નો જવાબ અહીં દેખાશે... તમે ગુજરાતીમાં બોલી શકો છો!';
    loadConversations();
  } catch (err) {
    console.error('Failed to create new conversation:', err);
  }
}

newChatBtn.onclick = createNewChat;
if (mobileNewChatBtn) mobileNewChatBtn.onclick = createNewChat;

function cleanup(){
  stopped = true;
  cancelAnimationFrame(animId);
  stopExperienceAudio();
  stopAllAudioPlayback();
  setAvatarState('idle');
  if (farmerVideo) {
    farmerVideo.pause();
    farmerVideo.currentTime = START_TIME_SEC;
  }
  if(workletNode){ workletNode.disconnect(); workletNode = null; }
  if(analyser){ analyser.disconnect(); analyser = null; }
  if(micStream){ micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if(sendCtx && sendCtx.state !== 'closed') sendCtx.close();
  if(recvCtx && recvCtx.state !== 'closed') recvCtx.close();
  if(ws && ws.readyState < 2) ws.close();
  sendCtx = recvCtx = ws = null;
  nextPlay = 0;
  startBtn.disabled = false;
  stopBtn.disabled = true;
  setStatus("વાતચીત શરૂ કરવા 'Start' દબાવો", '');
  drawFlat();
  loadConversations();
}

// ── Waveform Visualizer ─────────────────────────────────────────────────────
function drawFlat(){
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#c8e6c9';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, h/2);
  ctx.lineTo(w, h/2);
  ctx.stroke();
}
function drawWave(){
  if(!analyser || stopped) return;
  animId = requestAnimationFrame(drawWave);
  const d = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(d);
  const w = canvas.getBoundingClientRect().width;
  const h = canvas.getBoundingClientRect().height;
  ctx.clearRect(0, 0, w, h);
  ctx.strokeStyle = '#43a047';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const step = w / d.length;
  d.forEach((v, i) => {
    const x = i * step;
    const y = (v / 128) * (h / 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ── PCM16 Playback (24kHz) with active source tracking & avatar sync ────────
function playPCM16(ab){
  if(!recvCtx || stopped) return;
  setAvatarState('speaking');
  const i16 = new Int16Array(ab);
  const f32 = new Float32Array(i16.length);
  for(let i=0; i<i16.length; i++) f32[i] = i16[i] / 32768.0;
  
  const buf = recvCtx.createBuffer(1, f32.length, RECEIVE_RATE);
  buf.copyToChannel(f32, 0);
  
  const src = recvCtx.createBufferSource();
  src.buffer = buf;
  src.connect(recvCtx.destination);
  
  activeSources.add(src);
  src.onended = () => {
    activeSources.delete(src);
    if (activeSources.size === 0 && !stopped) {
      if (pendingExperienceAudio) {
        console.log('✅ AI voice finished. Starting queued experience audio now!');
        startExperienceAudio(pendingExperienceAudio);
      } else {
        setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
        setAvatarState('listening');
      }
    }
  };
  
  const now = recvCtx.currentTime;
  if(nextPlay < now) nextPlay = now + 0.04;
  src.start(nextPlay);
  nextPlay += buf.duration;
}

// ── Start Conversation ──────────────────────────────────────────────────────
startBtn.onclick = async () => {
  stopped = false;
  errMsg.textContent = '';
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus('કનેક્ટ થઈ રહ્યું છે...', '');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  const wsUrl = `${proto}://${location.host}/ws` + (currentConversationId ? `?conversation_id=${currentConversationId}` : '');
  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = async () => {
    setStatus('કનેક્ટ થઈ ગયું! બોલવાનું શરૂ કરો... 🎤', 'listening');
    setAvatarState('listening');
    sendCtx = new AudioContext({ sampleRate: SEND_RATE });
    recvCtx = new AudioContext({ sampleRate: RECEIVE_RATE });

    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: SEND_RATE,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true
        }
      });
    } catch(e) {
      showErr('માઈક્રોફોન પરમિશન નથી મળી: ' + e.message);
      cleanup();
      return;
    }

    const micSrc = sendCtx.createMediaStreamSource(micStream);
    analyser = sendCtx.createAnalyser();
    analyser.fftSize = 1024;
    micSrc.connect(analyser);

    const workletCode = `
class PCMCapture extends AudioWorkletProcessor {
  constructor() {
    super();
    this.bufferSize = 2048;
    this.buffer = new Int16Array(this.bufferSize);
    this.offset = 0;
  }
  process(inputs) {
    if (inputs.length > 0 && inputs[0].length > 0) {
      const ch = inputs[0][0];
      for (let i = 0; i < ch.length; i++) {
        this.buffer[this.offset++] = Math.max(-32768, Math.min(32767, ch[i] * 32768));
        if (this.offset >= this.bufferSize) {
          this.port.postMessage(this.buffer.buffer.slice(0));
          this.offset = 0;
        }
      }
    }
    return true;
  }
}
registerProcessor('pcm-capture', PCMCapture);`;

    const blob = new Blob([workletCode], { type: 'application/javascript' });
    const blobUrl = URL.createObjectURL(blob);
    await sendCtx.audioWorklet.addModule(blobUrl);
    URL.revokeObjectURL(blobUrl);

    workletNode = new AudioWorkletNode(sendCtx, 'pcm-capture');
    micSrc.connect(workletNode);
    workletNode.connect(sendCtx.destination);

    let speechStreak = 0;
    workletNode.port.onmessage = (e) => {
      if(ws && ws.readyState === WebSocket.OPEN && !stopped){
        ws.send(e.data);
      }

      // ── Voice Interruption Detection while Experience Audio is Playing ──────
      const isExpPlaying = (expAudioElement && !expAudioElement.paused) || pendingExperienceAudio;
      if (isExpPlaying) {
        const samples = new Int16Array(e.data);
        let sumSquares = 0;
        for (let i = 0; i < samples.length; i++) {
          sumSquares += samples[i] * samples[i];
        }
        const rms = Math.sqrt(sumSquares / samples.length);

        if (rms > 850) {
          speechStreak++;
          if (speechStreak >= 2) { // ~250ms of sustained speech
            console.log('⚡ User voice interrupted experience audio (RMS:', Math.round(rms), ')');
            stopExperienceAudio();
            stopAllAudioPlayback();
            setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
            setAvatarState('listening');
            speechStreak = 0;
          }
        } else {
          speechStreak = Math.max(0, speechStreak - 1);
        }
      } else {
        speechStreak = 0;
      }
    };

    drawWave();
  };

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      // If experience audio was playing, stop it immediately as AI started speaking
      if (expAudioElement && !expAudioElement.paused) {
        stopExperienceAudio();
      }
      setStatus('AI બોલી રહ્યા છે... 🔊', 'speaking');
      setAvatarState('speaking');
      playPCM16(ev.data);
      clearTimeout(ws._speakTimer);
      ws._speakTimer = setTimeout(() => {
        if (!stopped && activeSources.size === 0) {
          if (pendingExperienceAudio) {
            console.log('✅ Timer expired. Starting pending experience audio!');
            startExperienceAudio(pendingExperienceAudio);
          } else {
            setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
            setAvatarState('listening');
          }
        }
      }, 1200);
    } else {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.error) {
          showErr(msg.error);
        } else if (msg.type === 'play_experience_audio') {
          console.log('🎙️ Received farmer experience audio event:', msg);
          queueOrPlayExperienceAudio(msg);
          if (!transcript.classList.contains('transcript-placeholder')) {
            const expNotice = document.createElement('div');
            expNotice.style.cssText = 'color:#2e7d32;font-weight:600;margin:6px 0;padding:6px 10px;background:#e8f5e9;border-radius:8px;border-left:3px solid #43a047;font-size:0.85rem;';
            expNotice.textContent = `🎙️ [ઓડિયો]: ${msg.farmer_name || 'હસમુખભાઈ પટેલ'} (${msg.district || 'વલસાડ'}) નો જાત-અનુભવ શરૂ થશે...`;
            transcript.appendChild(expNotice);
            transcriptBox.scrollTop = transcriptBox.scrollHeight;
          }
        } else if (msg.type === 'interrupted') {
          console.log('⚡ Interrupted by user voice');
          stopExperienceAudio();
          stopAllAudioPlayback();
          setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
          setAvatarState('listening');
          if (!transcript.classList.contains('transcript-placeholder') && transcript.textContent.trim()) {
            transcript.textContent += ' ... [અટકાવેલ]\\n\\n🌾 AI: ';
            transcriptBox.scrollTop = transcriptBox.scrollHeight;
          }
        } else if (msg.type === 'text' && msg.text) {
          if (expAudioElement && !expAudioElement.paused) {
            stopExperienceAudio();
          }
          if (transcript.classList.contains('transcript-placeholder')) {
            transcript.className = '';
            transcript.textContent = '';
          }
          transcript.textContent += msg.text;
          transcriptBox.scrollTop = transcriptBox.scrollHeight;
        } else if (msg.type === 'turn_complete') {
          loadConversations();
          if (activeSources.size === 0 && !stopped) {
            if (pendingExperienceAudio) {
              console.log('✅ Turn complete. Starting pending experience audio!');
              startExperienceAudio(pendingExperienceAudio);
            } else {
              setAvatarState('listening');
            }
          }
        }
      } catch(_) {}
    }
  };

  ws.onerror = () => { showErr('WebSocket કનેક્શનમાં ખામી આવી'); cleanup(); };
  ws.onclose = () => { if (!stopped) { showErr('કનેક્શન બંધ થયું'); cleanup(); } };
};

stopBtn.onclick = () => cleanup();

// ── RAG Status Checker ──────────────────────────────────────────────────────
const ragBadge = document.getElementById('ragBadge');
async function checkRagStatus() {
  try {
    const res = await fetch('/api/rag/status');
    const data = await res.json();
    if (data.status === 'online') {
      ragBadge.className = 'rag-badge';
      ragBadge.textContent = `🧠 RAG જ્ઞાનકોશ: સક્રિય (${data.points_count || 0} માર્ગદર્શિકા)`;
    } else {
      ragBadge.className = 'rag-badge offline';
      ragBadge.textContent = '🧠 RAG: ઑફલાઇન (Docker શરૂ કરો)';
    }
  } catch (_) {
    ragBadge.className = 'rag-badge offline';
    ragBadge.textContent = '🧠 RAG: ઑફલાઇન';
  }
}

// Initial load
loadConversations();
checkRagStatus();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(HTML)
