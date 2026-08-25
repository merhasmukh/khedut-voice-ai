# 🌾 ખેડૂત Voice AI (Khedut Voice AI)

> **Real-Time Bidirectional Voice AI Assistant for Gujarati Organic Farmers**  
> Powered by **Gemini Live API**, **Qdrant Vector Database (RAG)**, and **SQLite Conversation Memory**.

---

## 🌟 Key Features

- **🎙️ Real-Time Voice Streaming**: Ultra-low latency bidirectional audio communication using Gemini Live WebSockets (`gemini-3.1-flash-live-preview`).
- **🗣️ Natural Gujarati Voice**: Configured with the **`Sadaltager`** voice persona for warm, natural, and friendly Gujarati speech.
- **⚡ Instant Barge-In (Interruption)**: Farmers can interrupt the AI at any time. Active audio playback stops with 0ms delay and smoothly transitions to answering the new question.
- **🌱 Conversational Farmer Profile Discovery**: Starts with a clean slate; automatically extracts farmer details (name, village, district, crops, land size, soil type) from natural dialogue using background AI parsing.
- **🧠 Qdrant RAG Knowledge Base**: Integrates Qdrant Vector DB with Gemini multilingual embeddings (`gemini-embedding-001`) to ground answers in verified agricultural recipes (Jeevamrut, Beejamrut, Dashaparni Ark, pest management, crop calendars).
- **📂 Multi-Format Ingestion**: Supports `.json`, `.pdf`, `.md`, and `.txt` files in `knowledge_base/`.
- **💾 Persistent Session Memory**: SQLite database stores conversation history, transcripts, and farmer profiles.
- **🎨 Responsive Web UI**: Audio visualizer (waveform), session sidebar, live transcripts, and RAG status badge.

---

## 🏗️ Architecture

```
[Browser Mic (16kHz PCM)] ──WebSocket──► [FastAPI Backend] ──WebSocket──► [Gemini Live API]
                                                 │                             │
                                         (RAG Retrieval)                       │
                                                 │                             │
                                       [Qdrant Vector DB]                      │
                                       (Docker: port 6333)                     │
                                                 │                             │
[Browser Speaker (24kHz)] ◄──WebSocket── [FastAPI Backend] ◄──WebSocket───────┘
                                                 │
                                       [SQLite Database]
                                   (Sessions & Farmer Profile)
```

---

## 📋 Prerequisites

Before setting up the project, make sure you have:

1. **Python 3.10+** (Python 3.11 - 3.13 recommended)
2. **Docker & Docker Desktop** (for running Qdrant Vector DB)
3. **Google Gemini API Key** ([Get your API Key from Google AI Studio](https://aistudio.google.com/))

---

## 🚀 Quick Start & Setup

### 1. Clone the Repository

```bash
git clone https://github.com/merhasmukh/khedut-voice-ai.git
cd khedut-voice-ai
```

### 2. Set Up Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

*(On Windows: `venv\Scripts\activate`)*

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Or manually create `.env` and add:

```env
GEMINI_API_KEY="your_actual_gemini_api_key_here"
QDRANT_HOST="localhost"
QDRANT_PORT=6333
QDRANT_COLLECTION="khedut_knowledge"
```

### 5. Start Qdrant Vector DB (Docker)

Start the Qdrant container with persistent storage:

```bash
docker compose up -d
```

- **Qdrant REST API**: `http://localhost:6333`
- **Qdrant Web Dashboard**: [`http://localhost:6333/dashboard`](http://localhost:6333/dashboard)

### 6. Run the FastAPI Application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Open your browser and navigate to:
👉 **[http://localhost:8000](http://localhost:8000)**

---

## 📚 Knowledge Base & RAG Management

### Pre-Built Seed Knowledge
The project comes with pre-populated Gujarati agricultural knowledge in `knowledge_base/`:
- `jeevamrut_and_fertilizers.json`: Formulations for Liquid Jeevamrut, Ghan-Jeevamrut, Beejamrut, and Vermicompost.
- `pest_and_disease_control.json`: Dashaparni Ark, Neem Oil (NSKE 5%), Cotton Pink Bollworm control, and Groundnut Tikka remedies.
- `gujarat_organic_crops.json`: Cumin (Jeera) disease management and soil health.

### Adding Your Own Documents
Drop any of the following file types into the `knowledge_base/` folder:
- **JSON (`.json`)**: Structured guides and recipes.
- **PDF (`.pdf`)**: University crop research, government scheme PDFs, organic manuals.
- **Markdown / Text (`.md`, `.txt`)**: Farming notes and articles.

### Trigger Ingestion
Once new files are added, index them into Qdrant by running:

```bash
curl -X POST http://localhost:8000/api/rag/ingest
```

*(Or restart the FastAPI server — it will auto-index on startup if empty).*

---

## 📡 REST API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Main Web UI Interface |
| `GET` | `/health` | Service health check |
| `WS` | `/ws?conversation_id=...` | Bidirectional Gemini Live voice WebSocket |
| `GET` | `/api/conversations` | List all conversation sessions |
| `POST` | `/api/conversations` | Create a new conversation session |
| `GET` | `/api/conversations/{id}` | Get session details & message history |
| `DELETE` | `/api/conversations/{id}` | Delete a conversation session |
| `GET` | `/api/profile` | Get current farmer profile |
| `POST` | `/api/profile` | Update farmer profile |
| `GET` | `/api/rag/status` | Get Qdrant status & vector count |
| `POST` | `/api/rag/ingest` | Index all files in `knowledge_base/` |
| `POST` | `/api/rag/search` | Semantic vector search test |

---

## 🧪 Testing

Run the automated test suite to verify all components:

```bash
# Run database and context tests
python tests/test_db_and_context.py

# Run RAG unit tests (embeddings, chunking, loaders)
python tests/test_rag_pipeline.py

# Run end-to-end ingestion and semantic search test
python tests/test_seed_and_search.py
```

---

## 📂 Project Structure

```
khedut-voice-ai/
├── ai_services/
│   ├── gemini_api.py           # Gemini Live WebSocket proxy & audio bridging
│   └── profile_extractor.py    # Background AI parser for farmer profile extraction
├── database/
│   ├── connection.py           # Async SQLite engine (aiosqlite)
│   ├── models.py               # SQLAlchemy models (FarmerProfile, Conversation, Message)
│   └── crud.py                 # CRUD operations & dynamic prompt context builder
├── knowledge_base/             # Agricultural knowledge repository (JSON, PDF, MD)
│   ├── jeevamrut_and_fertilizers.json
│   ├── pest_and_disease_control.json
│   └── gujarat_organic_crops.json
├── rag/
│   ├── embeddings.py           # Gemini text-embedding-001 (768 dims)
│   ├── qdrant_client.py        # Qdrant client connection & collection manager
│   ├── ingestion.py            # Multi-format document parser & chunker
│   └── retriever.py            # Vector search & Gujarati context formatter
├── tests/                      # Automated test scripts
├── docker-compose.yml          # Qdrant Docker service configuration
├── main.py                     # FastAPI application, REST APIs, and Web UI
├── requirements.txt            # Python package dependencies
└── README.md                   # Project documentation
```

---

## 🤝 License

Distributed under the **MIT License**. Feel free to use and adapt for agricultural AI initiatives.