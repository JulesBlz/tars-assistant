import sys, os
sys.path.insert(0, os.path.expanduser("~/tars"))

"""
Les trois outils exposés à l'agent TARS, avec leurs schémas JSON.

Contraintes de mesure intégrées dès la conception :
- add_calendar_event a un effet de bord → mode dry-run (validé sans écrire) pour l'éval.
- web_search est non déterministe → cache disque par requête pour rejouer le jeu de test.

search_knowledge réutilise l'embedder et la collection ChromaDB existants,
sans importer main.py (qui démarrerait le serveur FastAPI).
"""
import os
import json
import hashlib
from pathlib import Path
from datetime import datetime

import chromadb
from dotenv import load_dotenv

from embeddings import get_embedder
from gmail_client import get_google_credentials
from googleapiclient.discovery import build
from tavily import TavilyClient

load_dotenv(os.path.expanduser("~/tars/.env"))

# --- Configuration partagée ---
CHROMA_DIR = os.path.expanduser("~/tars/chroma_db")
COLLECTION_NAME = "jules_knowledge"
CACHE_DIR = Path.home() / "tars" / "agent" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Singletons chargés paresseusement
_embedder = None
_collection = None
_tavily = None


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


def _get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = get_embedder()
    return _embedder


def _get_tavily():
    global _tavily
    if _tavily is None:
        _tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
    return _tavily


# ============================================================
# OUTIL 1 : search_knowledge (RAG sur documents personnels)
# ============================================================

def search_knowledge(query, k=5):
    """
    Cherche dans les documents personnels de Jules (CV, portfolio, notes Obsidian).
    Retourne les passages les plus pertinents.
    """
    try:
        collection = _get_collection()
        embedder = _get_embedder()
    except Exception as e:
        return {"error": f"Base de connaissances indisponible : {e}"}

    if collection.count() == 0:
        return {"results": [], "note": "La base de connaissances est vide."}

    query_embedding = embedder.encode([query]).tolist()
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=k,
    )

    docs = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]

    # On filtre les résultats trop éloignés (seuil cohérent avec main.py)
    passages = []
    for doc, dist in zip(docs, distances):
        if dist <= 1.2:
            passages.append(doc)

    if not passages:
        return {"results": [], "note": "Aucun passage pertinent trouvé dans les documents personnels."}

    return {"results": passages}


# ============================================================
# OUTIL 2 : add_calendar_event (écriture Google Calendar)
# ============================================================

def add_calendar_event(title, date, time=None, duration_minutes=60, dry_run=False):
    """
    Ajoute un événement au Google Calendar de Jules.

    title : titre de l'événement
    date : date au format YYYY-MM-DD
    time : heure au format HH:MM (optionnel ; sinon événement journée entière)
    duration_minutes : durée en minutes (défaut 60)
    dry_run : si True, valide les arguments SANS écrire dans le calendrier (pour l'éval)
    """
    # Validation des arguments (faite dans les deux modes)
    try:
        datetime.strptime(date, "%Y-%m-%d")
    except (ValueError, TypeError):
        return {"error": f"Date invalide : '{date}'. Format attendu : YYYY-MM-DD."}

    if time is not None:
        try:
            datetime.strptime(time, "%H:%M")
        except ValueError:
            return {"error": f"Heure invalide : '{time}'. Format attendu : HH:MM."}

    # En dry-run, on s'arrête ici : les arguments sont validés, rien n'est écrit.
    if dry_run:
        return {
            "status": "dry_run_ok",
            "would_create": {"title": title, "date": date, "time": time, "duration_minutes": duration_minutes}
        }

    # Écriture réelle
    try:
        creds = get_google_credentials()
        service = build("calendar", "v3", credentials=creds)

        if time:
            start_dt = f"{date}T{time}:00"
            # Calcul de l'heure de fin
            start = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
            from datetime import timedelta
            end = start + timedelta(minutes=duration_minutes)
            end_dt = end.strftime("%Y-%m-%dT%H:%M:00")
            event_body = {
                "summary": title,
                "start": {"dateTime": start_dt, "timeZone": "Europe/Paris"},
                "end": {"dateTime": end_dt, "timeZone": "Europe/Paris"},
            }
        else:
            # Événement journée entière
            event_body = {
                "summary": title,
                "start": {"date": date},
                "end": {"date": date},
            }

        created = service.events().insert(calendarId="primary", body=event_body).execute()
        return {"status": "created", "event_id": created.get("id"), "link": created.get("htmlLink")}
    except Exception as e:
        return {"error": f"Échec de création de l'événement : {e}"}


# ============================================================
# OUTIL 3 : web_search (Tavily, avec cache disque)
# ============================================================

def _cache_key(query):
    """Clé de cache stable pour une requête donnée."""
    return hashlib.sha256(query.strip().lower().encode()).hexdigest()[:16]


def web_search(query, max_results=3, use_cache=True):
    """
    Recherche sur le web via Tavily.
    Les résultats sont mis en cache par requête pour que deux exécutions
    du jeu de test soient comparables (web_search est non déterministe).
    """
    key = _cache_key(query)
    cache_file = CACHE_DIR / f"search_{key}.json"

    # Lecture du cache
    if use_cache and cache_file.exists():
        with open(cache_file) as f:
            cached = json.load(f)
        cached["from_cache"] = True
        return cached

    # Appel réel
    try:
        client = _get_tavily()
        res = client.search(query, max_results=max_results)
        results = [
            {"title": r["title"], "url": r["url"], "content": r["content"][:500]}
            for r in res.get("results", [])
        ]
        output = {"query": query, "results": results, "from_cache": False}

        # Écriture du cache
        if use_cache:
            with open(cache_file, "w") as f:
                json.dump(output, f, ensure_ascii=False, indent=2)

        return output
    except Exception as e:
        return {"error": f"Échec de la recherche web : {e}"}


# ============================================================
# SCHÉMAS JSON exposés au modèle (format function calling)
# ============================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "Cherche dans les documents personnels de Jules (CV, portfolio, notes personnelles). Utilise cet outil quand la question porte sur Jules, son parcours, ses projets, ses préférences, ou toute information personnelle le concernant.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "La requête de recherche, formulée pour retrouver l'information pertinente."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "add_calendar_event",
            "description": "Ajoute un événement au calendrier de Jules. Utilise cet outil quand Jules demande de créer un rendez-vous, un rappel, ou de noter quelque chose dans son agenda.",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string", "description": "Le titre de l'événement."},
                    "date": {"type": "string", "description": "La date au format YYYY-MM-DD."},
                    "time": {"type": "string", "description": "L'heure au format HH:MM. Omettre pour un événement sur toute la journée."},
                    "duration_minutes": {"type": "integer", "description": "La durée en minutes. Défaut : 60."}
                },
                "required": ["title", "date"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Recherche des informations sur le web. Utilise cet outil pour des faits récents, des actualités, des informations externes que tu ne connais pas, ou tout ce qui n'est ni personnel à Jules ni dans son calendrier.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "La requête de recherche web."}
                },
                "required": ["query"]
            }
        }
    }
]


# Table de dispatch : nom d'outil → fonction Python
TOOL_FUNCTIONS = {
    "search_knowledge": search_knowledge,
    "add_calendar_event": add_calendar_event,
    "web_search": web_search,
}


# ============================================================
# Test manuel des trois outils
# ============================================================
if __name__ == "__main__":
    print("--- Test search_knowledge ---")
    print(search_knowledge("PFE de Jules"))
    print()
    print("--- Test add_calendar_event (dry-run) ---")
    print(add_calendar_event("Test dentiste", "2026-09-01", "14:00", dry_run=True))
    print()
    print("--- Test web_search ---")
    print(web_search("NeurIPS 2026 dates"))
