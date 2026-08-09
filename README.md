# TARS — local personal AI assistant

TARS is a privacy-first AI assistant I'm building to learn applied AI in depth. Inspired by the TARS robot from *Interstellar*, it runs primarily locally on my Mac, with a controlled fallback to a cloud model for tasks beyond the local model's capabilities.

This project is both a learning support and a tool I use daily. I document technical choices, trade-offs, and limitations encountered along the way.

## Features

- **Text and voice conversation**, fully local (Whisper for transcription, Piper for speech synthesis, Llama 3.1 8B for generation).
- **Defined personality** via system prompt (dry, direct, no flattery, occasional irony).
- **Persistent memory** in SQLite for the current conversation.
- **RAG** over personal documents (CV, portfolio, Obsidian vault linked via symlink) with ChromaDB and multilingual embeddings.
- **Semantic long-term memory**: substantial user messages and cloud model responses (when invoked) are ingested into a dedicated collection, allowing retrieval of past exchanges.
- **Cloud fallback** with two modes: auto-proposal by the local model when it detects a limitation, and manual escalation via UI button.
- **Consent-first**: every external call is an explicit user decision, with control over what leaves the Mac.
- **Minimalist web interface** available at `localhost:8000/ui`.

## Architecture

![TARS Architecture](./architecture.png)

The system follows a classic client-server pattern with a FastAPI backend orchestrating several modules:

- **Browser** → web interface at `localhost:8000/ui`
- **FastAPI** → central orchestrator
- **Ollama** → Llama 3.1 8B, local, conversational core
- **Whisper small** → audio → text transcription, local
- **Piper** → text → audio synthesis, French voice, local
- **ChromaDB** → vector database for RAG and long-term memory, local
- **SQLite** → conversation history, local
- **Cloud API** → more powerful model, invoked only on explicit validation

The full pipeline works offline for daily conversation. The cloud model is never invoked by default.

## Tech stack

- **Backend**: Python 3.11, FastAPI, uvicorn, aiosqlite, httpx
- **Local LLM**: Ollama, Llama 3.1 8B Q4_K_M
- **Cloud LLM (fallback)**: Anthropic API, Sonnet model
- **Voice**: Whisper small (STT), Piper with fr_FR-gilles-low voice (TTS)
- **RAG**: ChromaDB, sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Frontend**: Vanilla HTML/CSS/JS, MediaRecorder API for microphone

## What I learned building TARS

This project made me touch several concepts I only knew theoretically.

**On models.** Distinguishing model, runtime, and quantization. Understanding why an 8B in Q4 fits in 5 GB while the same model in FP16 takes 16 GB. Empirically observing that pure prompt engineering hits limits fast on a small model, and that no amount of instructions compensates for an undersized model.

**On RAG.** That semantic similarity favors prose-rich documents over structured ones like CVs. That chunk size and distance threshold are real decisions to make empirically, not defaults to leave alone. That the real challenge of RAG isn't the pipeline (easy) but corpus quality (hard).

**On orchestration.** That cloud fallback isn't just an API call, but an architecture and privacy question: what data is allowed to leave, under what user control, with what traceability. I implemented a *consent-first* escalation pattern where the model proposes escalation and the user confirms. Every external call is a deliberate decision.

**On memory.** That an LLM doesn't "remember" in the strong sense, and that what we call memory is reconstruction at each turn. That storing everything is noisy and requires curation: my ingestion rules (only substantial user messages and validated cloud responses) prevent the snowball effect of self-reinforcing hallucinations.

**On trade-offs.** That "100% local" is a principle you sometimes have to compromise. Cloud fallback exists to avoid mediocre responses, but stays explicit and optional.

## Known limitations

- Llama 3.1 8B follows instructions imperfectly as context accumulates. Escalation to a larger model remains possible but optional, and under explicit user control.
- RAG performs worse on highly structured documents (CV) than on prose (portfolio). Long-term fix: enrich the corpus with narrative content.
- Piper voice is decent but far from ElevenLabs. Trade-off accepted to keep TTS fully local.
- Incremental ingestion triggers on server startup, not in real time. A note taken in Obsidian while TARS is running is only indexed at the next restart.

## Planned next steps

- Read-only Gmail and Google Calendar integrations to enrich daily context.
- Mobile version via Tailscale and PWA.
- Light LoRA fine-tuning on the TARS personality to reduce prompt dependency.

## Requirements

- macOS Apple Silicon (M1+) with 16 GB RAM minimum, 24 GB recommended
- Python 3.11
- Homebrew, Ollama
- An API key for the fallback model (optional)

## Installation

Coming soon.

## Project structure

- `main.py` — FastAPI server, main routes
- `ingest.py` — Incremental document ingestion into ChromaDB
- `embeddings.py` — Embedding model singleton
- `memory.py` — Semantic long-term memory
- `escalation.py` — Cloud model fallback logic
- `audio.py` — STT (Whisper) and TTS (Piper)
- `Modelfile.example` — TARS model configuration template for Ollama
- `index.html` — Web interface
- `knowledge/` — Personal documents (gitignored)
- `voices/` — Piper voices (gitignored)

---

*Project by Jules Balzarini.*