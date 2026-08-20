from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
from calendar_client import get_upcoming_events, format_events_for_context
import httpx
from datetime import datetime
import aiosqlite
import tempfile
import subprocess
import os
import chromadb
from gmail_client import search_emails
from contextlib import asynccontextmanager
from embeddings import get_embedder
from escalation import (
    model_signals_escalation,
    jules_requests_escalation,
    detect_confirmation,
    detect_refusal,
    strip_escalation_token,
    ask_claude,
)
from memory import ingest_message, retrieve_memory

DB_PATH = "tars.db"
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL = "tars-ft-v2"
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
last_user_question = {"content": None}

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

def get_recent_emails_context(max_emails=20):
    """
    Récupère les métadonnées des N derniers emails pour injection permanente
    dans le contexte de TARS. Métadonnées seulement (from, subject, date, snippet).
    """
    try:
        emails = search_emails(query="", max_results=max_emails, days_back=30)
    except Exception as e:
        print(f"DEBUG GMAIL: erreur récupération contexte ({e})", flush=True)
        return ""
    
    if not emails:
        return ""
    
    lines = []
    for i, e in enumerate(emails, 1):
        # On garde uniquement les métadonnées + snippet court
        lines.append(f"{i}. [{e['date'][:16]}] {e['from'][:50]} — {e['subject'][:80]}")
        if e['snippet']:
            lines.append(f"   > {e['snippet'][:120]}")
    
    return "\n".join(lines)

def get_upcoming_events_context(days_ahead=7, max_events=15):
    """
    Récupère les événements Calendar à venir pour injection permanente dans TARS.
    """
    try:
        events = get_upcoming_events(days_ahead=days_ahead, max_results=max_events)
    except Exception as e:
        print(f"DEBUG CALENDAR: erreur récupération contexte ({e})", flush=True)
        return ""
    
    if not events:
        return ""
    
    return format_events_for_context(events)

def build_messages_for_ollama(history, query):
    docs_context = retrieve_context(query)
    memory_context = retrieve_memory(query)
    emails_context = get_recent_emails_context(max_emails=20)
    calendar_context = get_upcoming_events_context(days_ahead=7, max_events=15)
    messages = []

    system_parts = []
    
    if calendar_context:
        system_parts.append(f"""AGENDA de Jules (7 prochains jours).

Tu as accès à ces événements en permanence. Utilise ces infos quand Jules parle d'organisation, de rendez-vous, de disponibilités, ou pour lui rappeler des choses.

{calendar_context}""")
    
    if emails_context:
        system_parts.append(f"""INBOX RÉCENTE de Jules (20 derniers emails, métadonnées uniquement).

Tu as accès à ces informations en permanence. Utilise-les si Jules pose une question sur ses mails, ou si un contexte email est pertinent. Ne récite jamais toute la liste, sois sélectif.

Si Jules te demande le contenu détaillé d'un email spécifique, préviens-le que tu n'as que les métadonnées et propose-lui d'être plus précis.

{emails_context}""")
    
    if docs_context:
        system_parts.append(f"""Contexte issu de tes documents personnels sur Jules. Règles :
1. Ce contexte est écrit à la 3e personne. Réponds à la 2e personne.
2. Ne cite JAMAIS ce contexte explicitement.
3. Dis "tu", pas "Jules", quand tu parles à lui.

{docs_context}""")

    if memory_context:
        system_parts.append(f"""SOUVENIRS DE CONVERSATIONS PASSÉES AVEC JULES.

Tu te souviens de ces échanges. Ils sont RÉELS et FIABLES.

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
    # --- Chemin A : une escalade était en attente de confirmation ---
    if pending_escalation["question"]:
        if detect_confirmation(user_message):
            question = pending_escalation["question"]
            pending_escalation["question"] = None
            print(f"DEBUG ESCALADE: confirmée pour '{question[:50]}...'", flush=True)
            history = await load_history()
            claude_reply = ask_claude(question, context_history=history)
            reply = f"[Claude] {claude_reply}"
            ingest_message("assistant", reply)
            return reply, "claude"
        elif detect_refusal(user_message):
            pending_escalation["question"] = None
            print("DEBUG ESCALADE: refusée", flush=True)
            await save_message("user", user_message)
            ingest_message("user", user_message)
            history = await load_history()
            reply = await call_ollama(history, user_message)
            await save_message("assistant", reply)
            last_user_question["content"] = user_message
            return reply, "local"
        # Si le message n'est ni une confirmation ni un refus clair,
        # on laisse tomber l'escalade en attente et on traite normalement.
        pending_escalation["question"] = None

    # --- Chemin B : Jules demande LUI-MÊME l'escalade, explicitement ---
    if jules_requests_escalation(user_message):
        question_to_escalate = last_user_question["content"] or user_message
        print(f"DEBUG ESCALADE: demandée par Jules, question transmise: '{question_to_escalate[:50]}...'", flush=True)
        await save_message("user", user_message)
        ingest_message("user", user_message)
        history = await load_history()
        claude_reply = ask_claude(question_to_escalate, context_history=history)
        reply = f"[Claude] {claude_reply}"
        await save_message("assistant", reply)
        ingest_message("assistant", reply)
        return reply, "claude"

    # --- Chemin C : cas normal, réponse locale ---
    await save_message("user", user_message)
    ingest_message("user", user_message)
    history = await load_history()
    reply = await call_ollama(history, user_message)
    last_user_question["content"] = user_message

    # TARS peut signaler lui-même qu'il faut escalader, via le token structuré
    if model_signals_escalation(reply):
        pending_escalation["question"] = user_message
        reply = strip_escalation_token(reply)
        print(f"DEBUG ESCALADE: proposée par TARS pour '{user_message[:50]}...'", flush=True)

    await save_message("assistant", reply)
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

@app.post("/briefing")
async def briefing():
    print("DEBUG BRIEFING: génération en cours", flush=True)
    
    emails_context = get_recent_emails_context(max_emails=20)
    calendar_context = get_upcoming_events_context(days_ahead=3, max_events=10)
    
    now = datetime.now()
    date_str = now.strftime("%A %d %B %Y, %Hh%M").lower()
    
    briefing_prompt = f"""Tu es TARS. Jules vient d'ouvrir l'interface. Tu lui fais son briefing d'ouverture. C'est de la parole, pas un email.

NOUS SOMMES : {date_str}

# Structure OBLIGATOIRE, dans cet ordre

1. Ouvre TOUJOURS par une variante courte de "Bien réveillé, Jules." ou "Te revoilà, Jules." (varie légèrement, mais reste court et stable, jamais de blabla).
2. Météo de l'inbox : combien de mails vraiment utiles vs bruit, en une phrase.
3. Ce qui arrive : prochain(s) RDV imminent(s) avec l'horaire exact tel qu'écrit dans l'agenda.
4. Un point de vigilance si pertinent (deadline, mail à traiter). Optionnel.
5. Termine TOUJOURS par une invite courte du type "Qu'est-ce que je peux faire pour toi ?" ou "Par quoi tu commences ?" (varie légèrement, mais toujours une question ouverte à la fin).

# Ton

- Sec, direct, tars-esque. Zéro flagornerie.
- Ironie ponctuelle autorisée si un truc dans les mails ou l'agenda s'y prête. Pas systématique.
- 5 à 7 phrases courtes maximum au total.

# Règles absolues

- Contenu UNIQUEMENT basé sur les infos ci-dessous. Interdiction totale d'inventer un mail, un événement, un horaire.
- Cite les horaires EXACTEMENT tels qu'ils apparaissent dans l'agenda.
- Ne propose JAMAIS d'escalade vers Claude ici.

# Contexte

AGENDA des 3 prochains jours :
{calendar_context if calendar_context else "Aucun événement."}

DERNIERS EMAILS :
{emails_context}

Génère maintenant le briefing en respectant la structure obligatoire (ouverture stable, contenu, invite finale) :"""
    
    payload = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": briefing_prompt}
        ],
        "stream": False
    }
    
    async with httpx.AsyncClient(timeout=120) as client:
        response = await client.post(OLLAMA_URL, json=payload)
        data = response.json()
    
    briefing_text = data["message"]["content"]
    print(f"DEBUG BRIEFING: généré ({len(briefing_text)} chars)", flush=True)
    return {"briefing": briefing_text}

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