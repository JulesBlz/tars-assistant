"""
Module centralisé pour le modèle d'embeddings.
Chargé une seule fois, partagé entre main.py et ingest.py.
"""
from sentence_transformers import SentenceTransformer

_embedder = None


def get_embedder():
    """Retourne l'instance unique du modèle d'embeddings, chargée à la première demande."""
    global _embedder
    if _embedder is None:
        print("Chargement du modèle d'embeddings (une seule fois)...", flush=True)
        _embedder = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
    return _embedder