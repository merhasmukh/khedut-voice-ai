"""
Khedut Voice AI — FastAPI Server with Database-Backed Conversation Context
"""

from contextlib import asynccontextmanager
from typing import Optional, List
from pydantic import BaseModel
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database.connection import init_db, get_db
from database import crud
from ai_services.gemini_api import handle_gemini_live_session

load_dotenv()


# ─── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Initialize SQLite database schema
    await init_db()
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
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ખેડૂત Voice AI — ઓર્ગેનિક ખેતી સહાયક</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: system-ui, -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, sans-serif;
      background: linear-gradient(135deg, #f1f8e9 0%, #e8f5e9 100%);
      min-height: 100vh;
      display: flex;
      flex-direction: row;
      color: #2e3d2f;
    }

    /* Sidebar Drawer */
    #sidebar {
      width: 280px;
      background: #ffffff;
      border-right: 1px solid #dcedc8;
      display: flex;
      flex-direction: column;
      height: 100vh;
      box-shadow: 2px 0 12px rgba(0,0,0,0.03);
      transition: all 0.3s ease;
      z-index: 10;
    }
    .sidebar-header {
      padding: 20px;
      border-bottom: 1px solid #e8f5e9;
      display: flex;
      flex-direction: column;
      gap: 12px;
    }
    .sidebar-title {
      display: flex;
      align-items: center;
      gap: 8px;
      font-weight: 700;
      color: #1b5e20;
      font-size: 1.1rem;
    }
    .new-chat-btn {
      background: #2e7d32;
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
      transition: background 0.2s;
    }
    .new-chat-btn:hover { background: #1b5e20; }
    
    .conv-list {
      flex: 1;
      overflow-y: auto;
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
    .conv-item.active {
      background: #c8e6c9;
      color: #1b5e20;
      font-weight: 600;
    }
    .conv-title {
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      max-width: 200px;
    }

    /* Main Content Area */
    #main-container {
      flex: 1;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      padding: 24px;
      height: 100vh;
      overflow-y: auto;
    }
    .card {
      background: #ffffff;
      border-radius: 24px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.06);
      padding: 32px 38px;
      text-align: center;
      max-width: 520px;
      width: 100%;
      border: 1px solid rgba(46, 125, 50, 0.12);
    }
    .emoji { font-size: 48px; margin-bottom: 6px; }
    h1 { color: #1b5e20; font-size: 1.5rem; font-weight: 700; margin-bottom: 2px; }
    .subtitle { color: #558b2f; font-size: 0.9rem; margin-bottom: 16px; }
    
    .status-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      margin-bottom: 16px;
      background: #f4fbf5;
      padding: 8px 16px;
      border-radius: 50px;
      display: inline-flex;
    }
    .dot {
      width: 12px; height: 12px;
      border-radius: 50%;
      background: #bbb;
      transition: all 0.3s ease;
    }
    .dot.connected { background: #43a047; box-shadow: 0 0 8px #43a047; animation: pulse 1.4s infinite; }
    .dot.listening { background: #e53935; box-shadow: 0 0 10px #e53935; animation: pulse 0.8s infinite; }
    .dot.speaking  { background: #1e88e5; box-shadow: 0 0 10px #1e88e5; animation: pulse 0.9s infinite; }
    
    @keyframes pulse {
      0%, 100% { transform: scale(1); opacity: 1; }
      50% { transform: scale(1.35); opacity: 0.6; }
    }
    #statusText { color: #2e7d32; font-size: 0.9rem; font-weight: 600; }

    canvas {
      width: 100%;
      height: 56px;
      border-radius: 12px;
      background: #f9fbf9;
      margin-bottom: 18px;
      display: block;
      border: 1px solid #e0ede0;
    }

    .btn-row { display: flex; gap: 12px; justify-content: center; }
    button {
      padding: 12px 28px;
      font-size: 0.98rem;
      border: none;
      border-radius: 50px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s ease;
      display: flex;
      align-items: center;
      gap: 6px;
    }
    #startBtn {
      background: #2e7d32;
      color: #fff;
      box-shadow: 0 4px 16px rgba(46, 125, 50, 0.35);
    }
    #startBtn:hover:not(:disabled) {
      background: #1b5e20;
      transform: translateY(-2px);
      box-shadow: 0 6px 20px rgba(46, 125, 50, 0.45);
    }
    #stopBtn {
      background: #e53935;
      color: #fff;
      box-shadow: 0 4px 16px rgba(229, 57, 53, 0.3);
    }
    #stopBtn:hover:not(:disabled) {
      background: #c62828;
      transform: translateY(-2px);
    }
    button:disabled {
      background: #e0e0e0;
      color: #9e9e9e;
      box-shadow: none;
      cursor: not-allowed;
      transform: none;
    }

    #transcriptBox {
      margin-top: 18px;
      max-height: 180px;
      overflow-y: auto;
      text-align: left;
      background: #fdfefd;
      border: 1px solid #c8e6c9;
      border-radius: 14px;
      padding: 14px 16px;
      font-size: 0.9rem;
      color: #1b5e20;
      line-height: 1.6;
    }
    .transcript-placeholder { color: #81c784; font-style: italic; }
    .msg-user { color: #1565c0; font-weight: 500; margin-bottom: 6px; }
    .msg-ai { color: #2e7d32; font-weight: 500; margin-bottom: 6px; }
    .err { color: #c62828; font-size: 0.85rem; margin-top: 10px; }
  </style>
</head>
<body>

  <!-- Sidebar -->
  <div id="sidebar">
    <div class="sidebar-header">
      <div class="sidebar-title">🌱 ખેડૂત AI સત્રો</div>
      <button class="new-chat-btn" id="newChatBtn">➕ નવી વાતચીત શરૂ કરો</button>
    </div>
    <div class="conv-list" id="convList">
      <!-- Conversation history dynamically loaded here -->
    </div>
  </div>

  <!-- Main Container -->
  <div id="main-container">
    <div class="card">
      <div class="emoji">🌾</div>
      <h1 id="headerTitle">ખેડૂત Voice AI</h1>
      <p class="subtitle">ઓર્ગેનિક ખેતી માટે તમારો ડિજિટલ સાથી</p>
      
      <div class="status-row">
        <div class="dot" id="dot"></div>
        <span id="statusText">વાતચીત શરૂ કરવા 'Start' દબાવો</span>
      </div>
      
      <canvas id="waveform" width="440" height="56"></canvas>
      
      <div class="btn-row">
        <button id="startBtn">▶ Start વાતચીત</button>
        <button id="stopBtn" disabled>■ Stop</button>
      </div>
      
      <div id="transcriptBox">
        <div id="transcript" class="transcript-placeholder">AI નો જવાબ અહીં દેખાશે... તમે ગુજરાતીમાં બોલી શકો છો!</div>
      </div>
      <div class="err" id="errMsg"></div>
    </div>
  </div>

<script>
const SEND_RATE    = 16000;
const RECEIVE_RATE = 24000;

const startBtn     = document.getElementById('startBtn');
const stopBtn      = document.getElementById('stopBtn');
const statusText   = document.getElementById('statusText');
const dot          = document.getElementById('dot');
const transcript   = document.getElementById('transcript');
const transcriptBox= document.getElementById('transcriptBox');
const errMsg       = document.getElementById('errMsg');
const canvas       = document.getElementById('waveform');
const ctx          = canvas.getContext('2d');
const convList     = document.getElementById('convList');
const newChatBtn   = document.getElementById('newChatBtn');
const headerTitle  = document.getElementById('headerTitle');

let ws, sendCtx, recvCtx, micStream, workletNode, analyser;
let nextPlay = 0, animId, stopped = false;
let currentConversationId = null;
const activeSources = new Set();

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
      item.onclick = () => selectConversation(c.id);
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

newChatBtn.onclick = async () => {
  if (ws && ws.readyState < 2) cleanup();
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
};

function cleanup(){
  stopped = true;
  cancelAnimationFrame(animId);
  stopAllAudioPlayback();
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
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#c8e6c9';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height/2);
  ctx.lineTo(canvas.width, canvas.height/2);
  ctx.stroke();
}
function drawWave(){
  if(!analyser || stopped) return;
  animId = requestAnimationFrame(drawWave);
  const d = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(d);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#43a047';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const step = canvas.width / d.length;
  d.forEach((v, i) => {
    const x = i * step;
    const y = (v / 128) * (canvas.height / 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}
drawFlat();

// ── PCM16 Playback (24kHz) with active source tracking ─────────────────────
function playPCM16(ab){
  if(!recvCtx || stopped) return;
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
      setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
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

    // AudioWorklet for clean Int16 chunking
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

    workletNode.port.onmessage = (e) => {
      if(ws && ws.readyState === WebSocket.OPEN && !stopped){
        ws.send(e.data);
      }
    };

    drawWave();
  };

  ws.onmessage = (ev) => {
    if (ev.data instanceof ArrayBuffer) {
      setStatus('AI બોલી રહ્યા છે... 🔊', 'speaking');
      playPCM16(ev.data);
      clearTimeout(ws._speakTimer);
      ws._speakTimer = setTimeout(() => {
        if (!stopped && activeSources.size === 0) setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
      }, 1200);
    } else {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'session_info') {
          currentConversationId = msg.conversation_id;
          if (msg.title) headerTitle.textContent = msg.title;
          loadConversations();
        } else if (msg.type === 'interrupted') {
          // Instant barge-in: cut off all currently queued audio immediately
          console.log('⚡ Interrupted by user voice');
          stopAllAudioPlayback();
          setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
          if (!transcript.classList.contains('transcript-placeholder') && transcript.textContent.trim()) {
            transcript.textContent += ' ... [અટકાવેલ]\\n\\n🌾 AI: ';
            transcriptBox.scrollTop = transcriptBox.scrollHeight;
          }
        } else if (msg.type === 'text' && msg.text) {
          if (transcript.classList.contains('transcript-placeholder')) {
            transcript.className = '';
            transcript.textContent = '';
          }
          transcript.textContent += msg.text;
          transcriptBox.scrollTop = transcriptBox.scrollHeight;
        } else if (msg.type === 'turn_complete') {
          loadConversations();
        }
      } catch(_) {}
    }
  };

  ws.onerror = () => { showErr('WebSocket કનેક્શનમાં ખામી આવી'); cleanup(); };
  ws.onclose = () => { if (!stopped) { showErr('કનેક્શન બંધ થયું'); cleanup(); } };
};

stopBtn.onclick = () => cleanup();

// Initial load
loadConversations();
</script>
</body>
</html>
"""


@app.get("/")
def index():
    return HTMLResponse(HTML)
