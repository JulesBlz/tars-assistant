"""
Mémoire long-terme : ingère les messages user substantiels et les réponses Claude
dans une collection ChromaDB dédiée. Permet le retrieval sur conversations passées.
"""
import os
import time
import chromadb
from embeddings import get_embedder

CHROMA_DIR = os.path.expanduser("~/tars/chroma_db")
CONVERSATIONS_COLLECTION = "conversations"
MIN_USER_WORDS = 30


def get_conversations_collection():
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        return client.get_collection(CONVERSATIONS_COLLECTION)
    except Exception:
        return client.create_collection(name=CONVERSATIONS_COLLECTION)


def should_ingest(role, content):
    """Décide si un message doit être stocké en mémoire longue."""
    if role == "user":
        return len(content.split()) >= MIN_USER_WORDS
    if role == "assistant":
        return content.startswith("[Claude]")
    return False


def ingest_message(role, content):
    """Ajoute un message à la mémoire long-terme s'il est éligible."""
    if not should_ingest(role, content):
        return

    embedder = get_embedder()
    collection = get_conversations_collection()

    clean_content = content.replace("[Claude] ", "", 1) if role == "assistant" else content
    timestamp = int(time.time())
    doc_id = f"{role}_{timestamp}"

    embedding = embedder.encode([clean_content]).tolist()[0]

    collection.add(
        documents=[clean_content],
        embeddings=[embedding],
        metadatas=[{
            "role": role,
            "timestamp": timestamp,
            "source": "claude" if role == "assistant" else "user"
        }],
        ids=[doc_id]
    )
    print(f"DEBUG MEMORY: message {role} ingéré ({len(clean_content.split())} mots)", flush=True)


def retrieve_memory(query, k=3, distance_threshold=1.3):
    """Cherche dans la mémoire long-terme les messages pertinents."""
    collection = get_conversations_collection()
    if collection.count() == 0:
        return ""

    embedder = get_embedder()
    query_embedding = embedder.encode([query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(k, collection.count())
    )

    docs = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    filtered = [
        (doc, meta, dist)
        for doc, meta, dist in zip(docs, metadatas, distances)
        if dist < distance_threshold
    ]

    if not filtered:
        print(f"DEBUG MEMORY: 0 souvenirs pertinents (seuil {distance_threshold})", flush=True)
        return ""

    print(f"DEBUG MEMORY: {len(filtered)} souvenirs pertinents", flush=True)
    for doc, meta, dist in filtered:
        print(f"  -> [{dist:.2f}] {doc[:150]}...", flush=True)

    blocks = []
    for doc, meta, dist in filtered:
        role_label = "Jules" if meta["role"] == "user" else "Claude (via TARS)"
        date_str = time.strftime("%Y-%m-%d", time.localtime(meta["timestamp"]))
        blocks.append(f"[{date_str}, {role_label}, distance {dist:.2f}]\n{doc}")

    return "\n\n".join(blocks)