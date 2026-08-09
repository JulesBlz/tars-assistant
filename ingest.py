import os
import json
import chromadb

KNOWLEDGE_DIR = os.path.expanduser("~/tars/knowledge")
CHROMA_DIR = os.path.expanduser("~/tars/chroma_db")
INDEX_STATE_PATH = os.path.expanduser("~/tars/chroma_db/index_state.json")
COLLECTION_NAME = "jules_knowledge"
CHUNK_SIZE = 400
CHUNK_OVERLAP = 50


def chunk_text(text, size=CHUNK_SIZE, overlap=CHUNK_OVERLAP):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = " ".join(words[i:i + size])
        chunks.append(chunk)
        i += size - overlap
    return chunks


def read_file(path):
    filename = os.path.basename(path)
    if filename.startswith("portrait_"):
        return None
    if path.endswith((".md", ".txt")):
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    if path.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            reader = PdfReader(path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text.strip() if text.strip() else None
        except Exception as e:
            print(f"  Erreur lecture PDF {path}: {e}")
            return None
    return None


def load_index_state():
    if os.path.exists(INDEX_STATE_PATH):
        with open(INDEX_STATE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_index_state(state):
    os.makedirs(os.path.dirname(INDEX_STATE_PATH), exist_ok=True)
    with open(INDEX_STATE_PATH, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def scan_files():
    """Retourne un dict {relative_path: mtime} de tous les fichiers indexables."""
    found = {}
    for root, dirs, files in os.walk(KNOWLEDGE_DIR, followlinks=True):
        for filename in files:
            path = os.path.join(root, filename)
            if filename.startswith("portrait_"):
                continue
            if not path.endswith((".md", ".txt", ".pdf")):
                continue
            try:
                mtime = os.path.getmtime(path)
                relative = os.path.relpath(path, KNOWLEDGE_DIR)
                found[relative] = mtime
            except OSError:
                continue
    return found


def diff_files(previous_state, current_state):
    """Compare les états pour détecter nouveaux, modifiés, supprimés."""
    new_or_modified = []
    for path, mtime in current_state.items():
        if path not in previous_state or previous_state[path] != mtime:
            new_or_modified.append(path)
    deleted = [path for path in previous_state if path not in current_state]
    return new_or_modified, deleted


def ingest():
    from embeddings import get_embedder
    embedder = get_embedder()

    client = chromadb.PersistentClient(path=CHROMA_DIR)
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        collection = client.create_collection(name=COLLECTION_NAME)

    previous_state = load_index_state()
    current_state = scan_files()

    new_or_modified, deleted = diff_files(previous_state, current_state)

    if not new_or_modified and not deleted:
        print(f"Aucun changement. Collection à jour ({collection.count()} chunks).", flush=True)
        return

    print(f"Changements détectés : {len(new_or_modified)} nouveau/modifié, {len(deleted)} supprimé", flush=True)

    for path in deleted:
        try:
            collection.delete(where={"source": path})
            print(f"  Supprimé : {path}", flush=True)
        except Exception as e:
            print(f"  Erreur suppression {path}: {e}", flush=True)

    for relative_path in new_or_modified:
        try:
            collection.delete(where={"source": relative_path})
        except Exception:
            pass

        full_path = os.path.join(KNOWLEDGE_DIR, relative_path)
        content = read_file(full_path)
        if not content:
            continue

        chunks = chunk_text(content)
        print(f"  Indexation : {relative_path} ({len(chunks)} chunks)", flush=True)

        embeddings = embedder.encode(chunks).tolist()
        ids = [f"{relative_path}_{i}" for i in range(len(chunks))]
        metadatas = [
            {"source": relative_path, "chunk_index": i}
            for i in range(len(chunks))
        ]

        collection.add(
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )

    save_index_state(current_state)
    print(f"\nOK. Collection à jour : {collection.count()} chunks total.", flush=True)


if __name__ == "__main__":
    ingest()