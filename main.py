from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
import os
import asyncio
from google import genai
from google.genai import types

# Load .env BEFORE anything reads os.environ
load_dotenv()

MODEL = "models/gemini-3.5-live-translate-preview"

SYSTEM_INSTRUCTION = """
તમે એક અનુભવી ગુજરાતી ઓર્ગેનિક ખેડૂત છો જેઓ ઓર્ગેનિક ખેતી વિશે ઊંડી સમજ ધરાવો છો.
તમારી ભૂમિકા:
- ખેડૂતોને ઓર્ગેનિક ખેતીની સલાહ આપવી
- કુદરતી ખાતર, જૈવ જંતુનાશક, અને ટકાઉ ખેતી પ્રણાલી વિશે માર્ગદર્શન આપવું
- ગુજરાતની સ્થાનિક ફસલો — જેમ કે કપાસ, મગફળી, ઘઉં, બાજરી, શાકભાજી — વિશે સ્થાનિક જ્ઞાન શેર કરવું
- જમીનની ફળદ્રુપતા, પાણી વ્યવસ્થાપન, અને ઋતુ અનુસાર ખેતી અંગે સૂચનો આપવા

નિયમો:
- હંમેશા માત્ર ગુજરાતી ભાષામાં જ જવાબ આપો
- સરળ અને સ્થાનિક ભાષા વાપરો જે સામાન્ય ખેડૂત સમજી શકે
- ઓર્ગેનિક અને કુદરતી ઉપાયો જ સૂચવો, રાસાયણિક ઉત્પાદનો નહીં
- ગરમ, મૈત્રીપૂર્ણ અને વ્યવહારુ અભિગમ રાખો
"""

LIVE_CONFIG = types.LiveConnectConfig(
    response_modalities=["AUDIO"],
    system_instruction=types.Content(
        parts=[types.Part(text=SYSTEM_INSTRUCTION)],
        role="user",
    ),
    translation_config=types.TranslationConfig(
        target_language_code="gu",
    ),
)

# Global client — initialized in lifespan so env vars are guaranteed loaded.
client: genai.Client = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global client
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add it to your .env file or export it in your shell."
        )
    client = genai.Client(
        http_options={"api_version": "v1beta"},
        api_key=api_key,
    )
    print("✅ Gemini client initialized.")
    yield
    print("🛑 Shutting down.")


app = FastAPI(title="Khedut Voice AI API", lifespan=lifespan)

# ─────────────────────────────────────────────
# HTML UI
# ─────────────────────────────────────────────
html = """
<!DOCTYPE html>
<html lang="gu">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>ખેડૂત Voice AI</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', sans-serif;
      background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%);
      min-height: 100vh;
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      gap: 24px;
      padding: 20px;
    }
    .card {
      background: white;
      border-radius: 20px;
      box-shadow: 0 8px 32px rgba(0,0,0,0.12);
      padding: 40px 48px;
      text-align: center;
      max-width: 460px;
      width: 100%;
    }
    .emoji { font-size: 64px; margin-bottom: 12px; }
    h1 { color: #2e7d32; font-size: 1.6rem; margin-bottom: 6px; }
    .subtitle { color: #666; font-size: 0.95rem; margin-bottom: 28px; }

    /* Status indicator */
    .status-row {
      display: flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      margin-bottom: 28px;
    }
    .dot {
      width: 10px; height: 10px;
      border-radius: 50%;
      background: #bbb;
      transition: background 0.3s;
    }
    .dot.connected { background: #43a047; animation: pulse 1.4s infinite; }
    .dot.listening { background: #e53935; animation: pulse 0.8s infinite; }
    .dot.speaking  { background: #1e88e5; animation: pulse 1s infinite; }
    @keyframes pulse {
      0%,100% { opacity: 1; transform: scale(1); }
      50%      { opacity: 0.5; transform: scale(1.4); }
    }
    #statusText { color: #555; font-size: 0.9rem; font-weight: 500; }

    /* Waveform visualizer */
    canvas {
      width: 100%;
      height: 60px;
      border-radius: 10px;
      background: #f5f5f5;
      margin-bottom: 24px;
      display: block;
    }

    /* Buttons */
    .btn-row { display: flex; gap: 12px; justify-content: center; }
    button {
      padding: 12px 28px;
      font-size: 1rem;
      border: none;
      border-radius: 50px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.2s;
    }
    #startBtn {
      background: #43a047; color: white;
      box-shadow: 0 4px 14px rgba(67,160,71,0.4);
    }
    #startBtn:hover:not(:disabled) { background: #388e3c; transform: translateY(-1px); }
    #stopBtn {
      background: #e53935; color: white;
      box-shadow: 0 4px 14px rgba(229,57,53,0.4);
    }
    #stopBtn:hover:not(:disabled) { background: #c62828; transform: translateY(-1px); }
    button:disabled { background: #ccc; box-shadow: none; cursor: not-allowed; transform: none; }

    /* Transcript */
    #transcript {
      margin-top: 20px;
      max-height: 160px;
      overflow-y: auto;
      text-align: left;
      background: #f9fbe7;
      border-radius: 10px;
      padding: 12px 16px;
      font-size: 0.88rem;
      color: #33691e;
      line-height: 1.6;
      white-space: pre-wrap;
    }
    #transcript:empty::before { content: 'AI નો જવાબ અહીં દેખાશે...'; color: #aaa; }
    .error-msg { color: #c62828; font-size: 0.85rem; margin-top: 10px; }
  </style>
</head>
<body>
  <div class="card">
    <div class="emoji">🌾</div>
    <h1>ખેડૂત Voice AI</h1>
    <p class="subtitle">ઓર્ગેનિક ખેતી માટે ગુજરાતી AI સહાયક</p>

    <div class="status-row">
      <div class="dot" id="dot"></div>
      <span id="statusText">કનેક્ટ થવા માટે Start દબાવો</span>
    </div>

    <canvas id="waveform" width="400" height="60"></canvas>

    <div class="btn-row">
      <button id="startBtn">▶ Start</button>
      <button id="stopBtn" disabled>■ Stop</button>
    </div>

    <div id="transcript"></div>
    <div class="error-msg" id="errorMsg"></div>
  </div>

<script>
// ── Constants ───────────────────────────────────────────────────────────────
const SEND_SAMPLE_RATE    = 16000;   // Mic → Gemini
const RECEIVE_SAMPLE_RATE = 24000;   // Gemini → Speaker
const BUFFER_SIZE         = 4096;

// ── DOM refs ────────────────────────────────────────────────────────────────
const startBtn     = document.getElementById('startBtn');
const stopBtn      = document.getElementById('stopBtn');
const statusText   = document.getElementById('statusText');
const dot          = document.getElementById('dot');
const transcript   = document.getElementById('transcript');
const errorMsg     = document.getElementById('errorMsg');
const canvas       = document.getElementById('waveform');
const ctx          = canvas.getContext('2d');

// ── State ────────────────────────────────────────────────────────────────────
let ws, sendAudioCtx, recvAudioCtx;
let micStream, scriptProcessor, analyser;
let nextPlayTime = 0;
let animFrameId;
let stopped = false;

// ── Helpers ──────────────────────────────────────────────────────────────────
function setStatus(text, state) {
  statusText.textContent = text;
  dot.className = 'dot ' + (state || '');
}

function showError(msg) {
  errorMsg.textContent = msg;
  setTimeout(() => errorMsg.textContent = '', 5000);
}

function cleanup() {
  stopped = true;
  cancelAnimationFrame(animFrameId);
  if (scriptProcessor) { scriptProcessor.disconnect(); scriptProcessor = null; }
  if (analyser)        { analyser.disconnect(); analyser = null; }
  if (micStream)       { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
  if (sendAudioCtx && sendAudioCtx.state !== 'closed') sendAudioCtx.close();
  if (recvAudioCtx && recvAudioCtx.state !== 'closed') recvAudioCtx.close();
  if (ws && ws.readyState < 2) ws.close();
  sendAudioCtx = recvAudioCtx = ws = null;
  nextPlayTime = 0;
  startBtn.disabled = false;
  stopBtn.disabled  = true;
  setStatus('Disconnected', '');
  drawFlatLine();
}

// ── Waveform ─────────────────────────────────────────────────────────────────
function drawFlatLine() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#ccc';
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.moveTo(0, canvas.height / 2);
  ctx.lineTo(canvas.width, canvas.height / 2);
  ctx.stroke();
}

function drawWaveform() {
  if (!analyser || stopped) return;
  animFrameId = requestAnimationFrame(drawWaveform);
  const data = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteTimeDomainData(data);
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = '#43a047';
  ctx.lineWidth = 2;
  ctx.beginPath();
  const step = canvas.width / data.length;
  data.forEach((v, i) => {
    const x = i * step;
    const y = (v / 128) * (canvas.height / 2);
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}
drawFlatLine();

// ── Audio Playback (AAC / any browser-decodable format) ──────────────────────
// Gemini Live API returns audio/aac chunks, NOT raw PCM.
// We use decodeAudioData() which handles AAC, MP3, Ogg, etc. natively.
async function playAudio(arrayBuffer) {
  if (!recvAudioCtx || stopped) return;
  try {
    const audioBuffer = await recvAudioCtx.decodeAudioData(arrayBuffer);
    const src = recvAudioCtx.createBufferSource();
    src.buffer = audioBuffer;
    src.connect(recvAudioCtx.destination);
    const now = recvAudioCtx.currentTime;
    if (nextPlayTime < now) nextPlayTime = now + 0.05;
    src.start(nextPlayTime);
    nextPlayTime += audioBuffer.duration;
  } catch (err) {
    console.warn('decodeAudioData failed:', err);
  }
}

// ── Main start ────────────────────────────────────────────────────────────────
startBtn.onclick = async () => {
  stopped = false;
  errorMsg.textContent = '';
  startBtn.disabled = true;
  stopBtn.disabled  = false;
  setStatus('Connecting...', '');

  // 1. Open WebSocket
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws`);
  ws.binaryType = 'arraybuffer';

  ws.onopen = async () => {
    setStatus('Connected — speak now', 'connected');

    // 2. AudioContexts — send at 16kHz, receive at device default rate
    sendAudioCtx = new AudioContext({ sampleRate: SEND_SAMPLE_RATE });
    recvAudioCtx = new AudioContext();  // let browser pick best rate for AAC decode

    try {
      micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, sampleRate: SEND_SAMPLE_RATE, echoCancellation: true, noiseSuppression: true }
      });
    } catch (err) {
      showError('Microphone access denied: ' + err.message);
      cleanup();
      return;
    }

    const source = sendAudioCtx.createMediaStreamSource(micStream);

    // Analyser for waveform
    analyser = sendAudioCtx.createAnalyser();
    analyser.fftSize = 1024;
    source.connect(analyser);

    // ScriptProcessor to capture PCM16 and send to server
    scriptProcessor = sendAudioCtx.createScriptProcessor(BUFFER_SIZE, 1, 1);
    source.connect(scriptProcessor);
    scriptProcessor.connect(sendAudioCtx.destination);

    scriptProcessor.onaudioprocess = (e) => {
      if (!ws || ws.readyState !== WebSocket.OPEN || stopped) return;
      const float32 = e.inputBuffer.getChannelData(0);
      const int16   = new Int16Array(float32.length);
      for (let i = 0; i < float32.length; i++) {
        int16[i] = Math.max(-32768, Math.min(32767, float32[i] * 32768));
      }
      ws.send(int16.buffer);
    };

    setStatus('Listening... 🎤', 'listening');
    drawWaveform();
  };

  // 3. Receive audio (AAC) from Gemini and play it
  ws.onmessage = async (event) => {
    if (event.data instanceof ArrayBuffer) {
      setStatus('AI Speaking... 🔊', 'speaking');
      // Clone the buffer before passing to decodeAudioData (it transfers ownership)
      await playAudio(event.data.slice(0));
      clearTimeout(ws._speakTimer);
      ws._speakTimer = setTimeout(() => {
        if (!stopped) setStatus('Listening... 🎤', 'listening');
      }, 1200);
    } else if (typeof event.data === 'string') {
      const msg = JSON.parse(event.data);
      if (msg.text) {
        transcript.textContent += msg.text;
        transcript.scrollTop = transcript.scrollHeight;
      }
    }
  };

  ws.onerror = (e) => {
    showError('WebSocket error. Check server logs.');
    cleanup();
  };

  ws.onclose = () => {
    if (!stopped) {
      showError('Connection closed by server.');
      cleanup();
    }
  };
};

stopBtn.onclick = () => {
  cleanup();
};
</script>
</body>
</html>
"""


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────
@app.get("/")
def read_root():
    return HTMLResponse(html)


@app.get("/health")
def health_check():
    return {"status": "ok"}


# ─────────────────────────────────────────────
# WebSocket — Gemini Live API proxy
# ─────────────────────────────────────────────
@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")

    try:
        async with client.aio.live.connect(model=MODEL, config=LIVE_CONFIG) as session:

            async def browser_to_gemini():
                """Read raw PCM16 bytes from browser → forward to Gemini via send_realtime_input."""
                try:
                    while True:
                        data = await websocket.receive_bytes()
                        # Use the dedicated realtime audio method with correct MIME + rate
                        await session.send_realtime_input(
                            audio=types.Blob(data=data, mime_type="audio/pcm;rate=16000")
                        )
                except WebSocketDisconnect:
                    print("🔌 Client disconnected")
                except Exception as e:
                    print(f"browser_to_gemini error: {e}")

            async def gemini_to_browser():
                """Read Gemini responses turn-by-turn and push audio/text to browser."""
                import json
                try:
                    while True:
                        async for response in session.receive():
                            print(f"[DEBUG] raw response: {response}")

                            sc = response.server_content
                            print(f"[DEBUG] server_content: {sc}")

                            if not sc:
                                print("[DEBUG] no server_content, skipping")
                                continue

                            if sc.turn_complete:
                                print("[DEBUG] turn_complete received")
                                continue

                            mt = sc.model_turn
                            if not mt or not mt.parts:
                                print("[DEBUG] no model_turn or parts")
                                continue

                            for i, part in enumerate(mt.parts):
                                print(f"[DEBUG] part[{i}]: inline_data={bool(part.inline_data)}, text={repr(part.text)}")
                                if part.inline_data and part.inline_data.data:
                                    print(f"[DEBUG]   → sending {len(part.inline_data.data)} audio bytes")
                                    await websocket.send_bytes(part.inline_data.data)
                                elif part.text:
                                    print(f"[DEBUG]   → sending text: {part.text[:80]}")
                                    await websocket.send_text(
                                        json.dumps({"text": part.text})
                                    )
                except WebSocketDisconnect:
                    pass
                except Exception as e:
                    import traceback
                    print(f"gemini_to_browser error: {e}")
                    traceback.print_exc()

            # Run both directions concurrently
            await asyncio.gather(browser_to_gemini(), gemini_to_browser())

    except Exception as e:
        print(f"Session error: {e}")
        try:
            await websocket.close()
        except Exception:
            pass
