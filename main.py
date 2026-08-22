"""
Khedut Voice AI — FastAPI server
Uses a direct raw WebSocket connection to the Gemini Live API with AudioWorklet and real-time PCM streaming.
"""

import asyncio
import base64
import json
import os

import websockets
from dotenv import load_dotenv
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse

from ai_services.gemini_api import handle_gemini_live_session

load_dotenv()

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="Khedut Voice AI")

# ─── HTML UI ──────────────────────────────────────────────────────────────────
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
      background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 20px;
      padding: 20px;
    }
    .card {
      background: #ffffff;
      border-radius: 24px;
      box-shadow: 0 12px 40px rgba(0,0,0,0.08);
      padding: 36px 40px;
      text-align: center;
      max-width: 500px;
      width: 100%;
      border: 1px solid rgba(46, 125, 50, 0.1);
    }
    .emoji { font-size: 56px; margin-bottom: 8px; }
    h1 { color: #1b5e20; font-size: 1.6rem; font-weight: 700; margin-bottom: 4px; }
    .subtitle { color: #4b6b4e; font-size: 0.95rem; margin-bottom: 20px; }
    
    .status-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 10px;
      margin-bottom: 18px;
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
      height: 60px;
      border-radius: 12px;
      background: #f9fbf9;
      margin-bottom: 20px;
      display: block;
      border: 1px solid #e0ede0;
    }

    .btn-row { display: flex; gap: 14px; justify-content: center; }
    button {
      padding: 12px 30px;
      font-size: 1rem;
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
      margin-top: 20px;
      max-height: 180px;
      overflow-y: auto;
      text-align: left;
      background: #fdfefd;
      border: 1px solid #c8e6c9;
      border-radius: 14px;
      padding: 14px 18px;
      font-size: 0.92rem;
      color: #1b5e20;
      line-height: 1.6;
    }
    .transcript-placeholder { color: #81c784; font-style: italic; }
    .msg-user { color: #1565c0; font-weight: 500; margin-bottom: 6px; }
    .msg-ai { color: #2e7d32; font-weight: 500; margin-bottom: 6px; }
    .err { color: #c62828; font-size: 0.85rem; margin-top: 12px; }
  </style>
</head>
<body>
<div class="card">
  <div class="emoji">🌾</div>
  <h1>ખેડૂત Voice AI</h1>
  <p class="subtitle">ઓર્ગેનિક ખેતી માટે તમારો ડિજિટલ સાથી</p>
  
  <div class="status-row">
    <div class="dot" id="dot"></div>
    <span id="statusText">વાતચીત શરૂ કરવા 'Start' દબાવો</span>
  </div>
  
  <canvas id="waveform" width="420" height="60"></canvas>
  
  <div class="btn-row">
    <button id="startBtn">▶ Start વાતચીત</button>
    <button id="stopBtn" disabled>■ Stop</button>
  </div>
  
  <div id="transcriptBox">
    <div id="transcript" class="transcript-placeholder">AI નો જવાબ અહીં દેખાશે... તમે ગુજરાતીમાં બોલી શકો છો!</div>
  </div>
  <div class="err" id="errMsg"></div>
</div>

<script>
const SEND_RATE    = 16000;
const RECEIVE_RATE = 24000;

const startBtn   = document.getElementById('startBtn');
const stopBtn    = document.getElementById('stopBtn');
const statusText = document.getElementById('statusText');
const dot        = document.getElementById('dot');
const transcript = document.getElementById('transcript');
const errMsg     = document.getElementById('errMsg');
const canvas     = document.getElementById('waveform');
const ctx        = canvas.getContext('2d');

let ws, sendCtx, recvCtx, micStream, workletNode, analyser;
let nextPlay = 0, animId, stopped = false;

function setStatus(txt, cls){
  statusText.textContent = txt;
  dot.className = 'dot ' + (cls || '');
}
function showErr(msg){
  errMsg.textContent = msg;
  setTimeout(() => errMsg.textContent = '', 6000);
}

function cleanup(){
  stopped = true;
  cancelAnimationFrame(animId);
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
}

// Waveform visualizer
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

// PCM16 Audio Playback (24kHz)
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
  
  const now = recvCtx.currentTime;
  if(nextPlay < now) nextPlay = now + 0.04;
  src.start(nextPlay);
  nextPlay += buf.duration;
}

startBtn.onclick = async () => {
  stopped = false;
  errMsg.textContent = '';
  startBtn.disabled = true;
  stopBtn.disabled = false;
  setStatus('કનેક્ટ થઈ રહ્યું છે...', '');

  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
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

    // AudioWorklet for clean Int16 chunking (replaces deprecated ScriptProcessor)
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
        if (!stopped) setStatus('સાંભળી રહ્યા છે... 🎤', 'listening');
      }, 1200);
    } else {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.text) {
          if (transcript.classList.contains('transcript-placeholder')) {
            transcript.className = '';
            transcript.textContent = '';
          }
          transcript.textContent += msg.text;
          const box = document.getElementById('transcriptBox');
          box.scrollTop = box.scrollHeight;
        }
      } catch(_) {}
    }
  };

  ws.onerror = () => { showErr('WebSocket કનેક્શનમાં ખામી આવી'); cleanup(); };
  ws.onclose = () => { if (!stopped) { showErr('કનેક્શન બંધ થયું'); cleanup(); } };
};

stopBtn.onclick = () => cleanup();
</script>
</body>
</html>
"""


# ─── Routes ───────────────────────────────────────────────────────────────────
@app.get("/")
def index():
    return HTMLResponse(HTML)


@app.get("/health")
def health():
    return {"status": "ok"}


# ─── WebSocket proxy: Browser ↔ Gemini Live ─────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(browser_ws: WebSocket):
    await browser_ws.accept()
    print("🔌 Browser connected")
    await handle_gemini_live_session(browser_ws)

