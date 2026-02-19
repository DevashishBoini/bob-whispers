# Bob Whispers

**Context-aware, RAG-powered LLM query bot with full voice: speak in, hear back (STT + TTS).**

An AI assistant that combines **conversation memory**, **semantic search (RAG)**, and **full voice interaction**: use **speech-to-text (STT)** to talk or upload audio, get answers grounded in past conversations, and use **text-to-speech (TTS)** to hear the AI’s replies spoken aloud—in the API and in the Gradio UI.

---

## Table of contents

- [Features & Functionality](#features--functionality)
  - [Voice: STT & TTS (speak in, hear back)](#voice-stt--tts-speak-in-hear-back)
  - [Chat & conversations](#chat--conversations)
  - [Context & memory (RAG)](#context--memory-rag)
  - [API & UI](#api--ui)
- [Tech Stack](#tech-stack)
  - [Main directories](#main-directories)
- [Setup](#setup)
- [Quick reference](#quick-reference)
- [API endpoints (summary)](#api-endpoints-summary)
- [License & attribution](#license--attribution)

---

## Features & Functionality

### Voice: STT & TTS (speak in, hear back)

- **Speech-to-Text (STT)**  
  - Send voice instead of typing: record in the browser or upload an audio file.  
  - Audio is sent to **Google Gemini’s audio API** for transcription.  
  - The transcribed text is then processed through the same conversation pipeline as typed messages (including RAG and memory).  
  - Use **`POST /chat/voice`** with an audio file; the backend returns the AI reply (and optionally TTS) for that turn.

- **Text-to-Speech (TTS)**  
  - **Backend:** Optional TTS for every AI response. Request it with `enable_tts=true` on **`POST /chat/message`** or **`POST /chat/voice`**; the API can return TTS-ready content or playback instructions.  
  - **Gradio UI:** “Read text” control to speak any text (e.g. the last AI reply) using the browser’s speech synthesis, with **female / male / default** voice options.  
  - TTS is **on by default** for voice messages in the UI so you can have a full speak-in, hear-back flow.

- **End-to-end voice flow** — Speak or upload audio → STT (Gemini) → same chat + RAG pipeline → AI reply → optional TTS (backend or browser). All with the same conversation history and semantic memory as text chat.

### Chat & conversations
- **Multi-thread conversations** — Create, switch, and delete chat threads; each has its own history and context.
- **Text and voice input** — Type messages or use STT (record/upload); voice is transcribed then handled like text.
- **Persistent history** — All exchanges (text and voice) stored in SQLite; conversation list and history survive restarts.

### Context & memory (RAG)
- **Short-term context** — Recent messages in the current thread are always included for coherent dialogue.
- **Semantic long-term memory** — Past messages are embedded and stored in ChromaDB; relevant snippets are retrieved by meaning, not just keywords.
- **Hybrid context** — The model receives both recent turns and semantically relevant past content so it can reference earlier discussions (e.g. “remember when we talked about API testing?”).

### API & UI
- **REST API** — FastAPI backend with documented endpoints for chat, conversations, history, health, and stats; **STT** via `/chat/voice`, **TTS** via `enable_tts` on `/chat/message` and `/chat/voice`.
- **Gradio UI** — Web UI with conversation sidebar, chat area, text input; **STT**: record or upload audio and “Send Voice”; **TTS**: “Read text” with female/male/default voice, and optional TTS for every AI reply.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| **Backend** | Python 3, FastAPI, Uvicorn |
| **AI / LLM** | Google Gemini (Gemini 2.5 Flash for text and audio) |
| **STT (Speech-to-Text)** | Google Gemini Audio API — transcribes uploaded/recorded audio to text |
| **TTS (Text-to-Speech)** | Backend TTS service + browser Speech Synthesis API in Gradio (female/male/default) |
| **Embeddings & RAG** | Google `embedding-001`, LangChain, ChromaDB |
| **Database** | SQLite (conversations + messages), SQLAlchemy ORM |
| **Vector store** | ChromaDB (persistent, per-conversation collections) |
| **Frontend** | Gradio (chat UI, microphone/upload for STT, “Read text” and voice picker for TTS) |
| **Config** | Pydantic v2, `python-dotenv` (.env) |

### Main directories
- **`say-my-name/`** — Application root: backend, UI, and services.
- **`say-my-name/src/`** — Core app: `main.py` (FastAPI), `config.py`, `database.py`, `gradio_ui.py`.
- **`say-my-name/src/services/`** — Conversation pipeline: base conversation service, enhanced (semantic memory), voice-enhanced (STT + TTS), Gemini client, audio client, memory, semantic search, vector DB.
- **`say-my-name/data/`** — SQLite DB and ChromaDB data (created at runtime).

---

## Setup

### 1. Clone and enter the app directory

```bash
git clone <repo-url> bob-whispers
cd bob-whispers/say-my-name
```

### 2. Create a virtual environment (recommended)

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
```

Edit `.env` and set at least:

- **`GEMINI_API_KEY`** — From [Google AI Studio](https://aistudio.google.com/app/apikey).

Optional (defaults are fine for local use):

- `GEMINI_TEXT_MODEL`, `GEMINI_AUDIO_MODEL` — e.g. `gemini-2.5-flash`
- `DATABASE_PATH` — SQLite path (default `data/conversations.db`)
- `CHROMADB_PATH` — Vector DB path (default `data/chromadb`)
- `APP_HOST`, `APP_PORT` — Backend host/port (default `0.0.0.0:8000`)

### 5. Run the backend

From `say-my-name/`:

```bash
python src/main.py
```

Or with uvicorn:

```bash
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

- API: `http://localhost:8000`
- Docs: `http://localhost:8000/docs`
- Health: `http://localhost:8000/health`

### 6. Run the Gradio UI (optional)

In a second terminal, from `say-my-name/`:

```bash
python src/gradio_ui.py
```

- UI: `http://localhost:7860` (or the URL Gradio prints; `share=True` can give a public link).

Ensure the FastAPI backend is running first; the UI talks to it for chat, voice, and conversations.

---

## Quick reference

| Task | Command (from `say-my-name/`) |
|------|-------------------------------|
| Start API | `python src/main.py` |
| Start UI | `python src/gradio_ui.py` |
| API docs | Open `http://localhost:8000/docs` |
| Health check | `curl http://localhost:8000/health` |

---

## API endpoints (summary)

- **`POST /chat/message`** — Send text message; set **`enable_tts: true`** to get a TTS response for the AI reply.
- **`POST /chat/voice`** — **STT + chat + optional TTS:** upload an audio file; backend transcribes (STT), runs the same chat pipeline, returns AI reply and optionally TTS (e.g. playback commands or audio).
- `POST /conversations` — Create conversation.
- `GET /conversations` — List conversations.
- `GET /conversations/{id}/history` — Message history.
- `DELETE /conversations/{id}` — Delete conversation.
- `GET /conversations/stats` — Stats.
- `GET /health` — Liveness/readiness.

---

## License & attribution

Private educational project. Uses Google Gemini; comply with Google’s API terms and usage limits (e.g. rate limits in `.env`).
