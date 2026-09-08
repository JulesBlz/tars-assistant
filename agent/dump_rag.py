import sys, os
sys.path.insert(0, os.path.expanduser("~/tars"))

import chromadb

CHROMA_DIR = os.path.expanduser("~/tars/chroma_db")
COLLECTION_NAME = "jules_knowledge"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

print(f"Total chunks : {collection.count()}\n")

# Récupérer tous les documents et leurs métadonnées
data = collection.get(include=["documents", "metadatas"])

for i, (doc, meta) in enumerate(zip(data["documents"], data["metadatas"])):
    source = meta.get("source", "?") if meta else "?"
    print(f"--- chunk {i+1} | source: {source} ---")
    print(doc[:400])
    print()