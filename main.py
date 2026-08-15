from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import httpx
import aiosqlite
import tempfile
import subprocess
import os
import chromadb
from contextlib import asynccontextmanager
from embeddings import get_embedder
from escalation import (
    detect_escalation_proposal,
    detect_confirmation,
    detect_refusal,
    ask_claude,
)
from memory import ingest_message, retrieve_memory

DB_PATH = "tars.db"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "tars"
CHROMA_DIR = os.path.expanduser("~/tars/chroma_db")
COLLECTION_NAME = "jules_knowledge"
TOP_K = 6

embedder = get_embedder()
chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
try:
    collection = chroma_client.get_collection(COLLECTION_NAME)
    print(f"Collection chargée : {collection.count()} chunks", flush=True)
except Exception:
    collection = None
    print("Aucune collection RAG trouvée. Lance ingest.py d'abord.", flush=True)

pending_escalation = {"question": None}


async def init_db():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        await db.commit()


async def load_history():
    async with aiosqlite.connect(DB_PATH) as db:
        cursor = await db.execute(
            "SELECT role, content FROM messages ORDER BY id ASC"
        )
        rows = await cursor.fetchall()
        return [{"role": row[0], "content": row[1]} for row in rows]


async def save_message(role: str, content: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (role, content) VALUES (?, ?)",
            (role, content)
        )
        await db.commit()


def retrieve_context(query, k=TOP_K, distance_threshold=1.2):
    if collection is None or collection.count() == 0:
        return ""
    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k
    )
    chunks = results["documents"][0]
    sources = [m["source"] for m in results["metadatas"][0]]
    distances = results["distances"][0]

    filtered = [
        (chunk, source, dist)
        for chunk, source, dist in zip(chunks, sources, distances)
        if dist < distance_threshold
    ]

    if not filtered:
        print(f"DEBUG RAG: 0 chunks pertinents (seuil {distance_threshold})", flush=True)
        return ""

    print(f"DEBUG RAG: {len(filtered)}/{len(chunks)} chunks pertinents, sources: {set(f[1] for f in filtered)}", flush=True)

    context_blocks = []
    for chunk, source, dist in filtered:
        context_blocks.append(f"[Source: {source}, distance: {dist:.2f}]\n{chunk}")
    return "\n\n".join(context_blocks)


def build_messages_for_ollama(history, query):
    docs_context = retrieve_context(query)
    memory_context = retrieve_memory(query)
    messages = []

    system_parts = []
    if docs_context:
        system_parts.append(f"""Contexte issu de tes documents personnels sur Jules. Règles :
1. Ce contexte est écrit à la 3e personne. Réponds à la 2e personne.
2. Ne cite JAMAIS ce contexte explicitement.
3. Dis "tu", pas "Jules", quand tu parles à lui.

{docs_context}""")

    if memory_context:
        system_parts.append(f"""SOUVENIRS DE CONVERSATIONS PASSÉES AVEC JULES.

Tu te souviens de ces échanges. Ils sont RÉELS et FIABLES. Ne prétends JAMAIS que tu ne te souviens pas d'une conversation si un souvenir pertinent apparaît ici.

Utilise ces souvenirs pour la continuité, sans les citer explicitement.

{memory_context}""")

    if system_parts:
        messages.append({
            "role": "system",
            "content": "\n\n".join(system_parts)
        })

    messages.extend(history)
    print(f"DEBUG: {len(messages)} messages sent to Ollama", flush=True)
    return messages


async def call_ollama(history, query):
    payload = {
        "model": MODEL,
        "messages": build_messages_for_ollama(history, query),
        "stream": False
    }
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        data = response.json()
    return data["message"]["content"]


async def process_message(user_message):
    if pending_escalation["question"]:
        if detect_confirmation(user_message):
            question = pending_escalation["question"]
            pending_escalation["question"] = None
            print(f"DEBUG: escalade confirmée pour: {question[:50]}...", flush=True)
            history = await load_history()
            claude_reply = ask_claude(question, context_history=history)
            reply = f"[Claude] {claude_reply}"
            ingest_message("assistant", reply)
            return reply, "claude"
        elif detect_refusal(user_message):
            pending_escalation["question"] = None
            print("DEBUG: escalade refusée", flush=True)
            await save_message("user", user_message)
            ingest_message("user", user_message)
            history = await load_history()
            reply = await call_ollama(history, user_message)
            await save_message("assistant", reply)
            return reply, "local"

    await save_message("user", user_message)
    ingest_message("user", user_message)
    history = await load_history()
    reply = await call_ollama(history, user_message)
    await save_message("assistant", reply)

    if detect_escalation_proposal(reply):
        pending_escalation["question"] = user_message
        print(f"DEBUG: escalade proposée pour: {user_message[:50]}...", flush=True)

    return reply, "local"


@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    try:
        print("Vérification de l'index RAG...", flush=True)
        from ingest import ingest
        ingest()
    except Exception as e:
        print(f"Erreur ingestion au démarrage : {e}", flush=True)
    yield


app = FastAPI(lifespan=lifespan)


class Message(BaseModel):
    content: str


@app.get("/")
def root():
    return {"status": "TARS is running"}


@app.get("/ui")
def ui():
    return FileResponse("index.html")


@app.get("/history")
async def get_history():
    history = await load_history()
    return {"history": history}


@app.delete("/history")
async def clear_history():
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages")
        await db.commit()
    pending_escalation["question"] = None
    return {"status": "conversation cleared"}


@app.post("/chat")
async def chat(message: Message):
    reply, source = await process_message(message.content)
    if source == "claude":
        await save_message("assistant", reply)
    return {"reply": reply, "source": source}


@app.post("/ask-claude")
async def ask_claude_route(message: Message):
    await save_message("user", message.content)
    ingest_message("user", message.content)
    history = await load_history()
    print(f"DEBUG: envoi direct à Claude pour: {message.content[:50]}...", flush=True)
    claude_reply = ask_claude(message.content, context_history=history)
    reply = f"[Claude] {claude_reply}"
    await save_message("assistant", reply)
    ingest_message("assistant", reply)
    return {"reply": reply, "source": "claude"}


@app.post("/voice")
async def voice_chat(audio: UploadFile = File(...)):
    import whisper

    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
        f.write(await audio.read())
        webm_path = f.name

    wav_path = webm_path.replace(".webm", ".wav")
    subprocess.run([
        "ffmpeg", "-i", webm_path, "-ar", "16000", "-ac", "1", wav_path, "-y"
    ], capture_output=True, check=True)

    model = whisper.load_model("small")
    result = model.transcribe(wav_path, language="fr")
    text = result["text"].strip()

    os.unlink(webm_path)
    os.unlink(wav_path)

    if not text:
        return {"transcription": "", "reply": ""}

    reply, source = await process_message(text)
    if source == "claude":
        await save_message("assistant", reply)

    return {"transcription": text, "reply": reply, "source": source}


@app.post("/tts")
async def text_to_speech(message: Message):
    from audio import generate_audio_bytes
    source = "claude" if message.content.startswith("[Claude]") else "tars"
    text = message.content.replace("[Claude] ", "", 1) if source == "claude" else message.content
    audio_data = generate_audio_bytes(text, source=source)
    return Response(content=audio_data, media_type="audio/wav")