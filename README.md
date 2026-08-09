# TARS — assistant IA personnel local

TARS est un assistant IA privacy-first que je développe pour apprendre l'IA appliquée en profondeur. Inspiré du robot TARS d'*Interstellar*, il fonctionne principalement en local sur mon Mac, avec fallback contrôlé vers un modèle cloud pour les tâches qui dépassent les capacités du modèle local.

Ce projet est un support d'apprentissage autant qu'un outil que j'utilise au quotidien. Je documente ici les choix techniques, les arbitrages, et les limites que j'ai rencontrées.

## Fonctionnalités

- **Conversation texte et voix** entièrement locale (Whisper pour la transcription, Piper pour la synthèse vocale, Llama 3.1 8B pour la génération).
- **Personnalité définie** via system prompt (sec, franc, sans flagornerie, ironie ponctuelle).
- **Mémoire persistante** dans SQLite pour la conversation courante.
- **RAG** sur mes documents personnels (CV, portfolio, vault Obsidian branché via symlink) avec ChromaDB et embeddings multilingues.
- **Mémoire long-terme sémantique** : mes messages substantiels et les réponses du modèle cloud (lorsqu'invoqué) sont ingérés dans une collection dédiée, permettant de retrouver des échanges anciens.
- **Fallback cloud** avec deux modes : auto-proposition par le modèle local quand il détecte une limite, et escalade manuelle par bouton dans l'interface.
- **Consent-first** : chaque appel externe est une décision explicite de l'utilisateur, avec un contrôle sur ce qui sort du Mac.
- **Interface web minimaliste** accessible sur `localhost:8000/ui`.

## Architecture

Le système suit un pattern client-serveur classique avec un backend FastAPI orchestrant plusieurs modules :

- **Navigateur** → interface web sur `localhost:8000/ui`
- **FastAPI** → orchestrateur central
- **Ollama** → Llama 3.1 8B, local, cerveau conversationnel
- **Whisper small** → transcription audio → texte, local
- **Piper** → synthèse texte → audio, voix française, local
- **ChromaDB** → base vectorielle pour RAG et mémoire long-terme, local
- **SQLite** → historique de conversation, local
- **API cloud** → modèle plus puissant, sollicité uniquement sur validation explicite

Le pipeline complet fonctionne sans internet pour la conversation quotidienne. Le modèle cloud n'est jamais sollicité par défaut.

## Stack technique

- **Backend** : Python 3.11, FastAPI, uvicorn, aiosqlite, httpx
- **LLM local** : Ollama, Llama 3.1 8B Q4_K_M
- **LLM cloud (fallback)** : API Anthropic, modèle Sonnet
- **Voix** : Whisper small (STT), Piper avec voix fr_FR-gilles-low (TTS)
- **RAG** : ChromaDB, sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **Frontend** : HTML/CSS/JS vanilla, MediaRecorder API pour le micro

## Ce que j'ai appris en construisant TARS

Ce projet m'a fait toucher concrètement à plusieurs concepts que je ne connaissais que théoriquement.

**Sur les modèles.** Distinguer modèle, runtime et quantization. Comprendre pourquoi un 8B en Q4 tient dans 5 Go alors que le même modèle en FP16 fait 16 Go. Constater empiriquement que le prompt engineering pur atteint vite ses limites sur un petit modèle, et qu'aucune quantité d'instructions ne compense un modèle sous-dimensionné pour la tâche.

**Sur le RAG.** Que la similarité sémantique favorise les documents riches en prose au détriment des documents structurés type CV. Que le chunk size et le seuil de distance sont des vraies décisions à faire empiriquement, pas des paramètres à laisser par défaut. Que le vrai enjeu du RAG n'est pas le pipeline (facile) mais la qualité du corpus (dur).

**Sur l'orchestration.** Que le fallback cloud n'est pas juste un appel API, mais un enjeu d'architecture et de confidentialité : quelles données on autorise à sortir, sous quel contrôle utilisateur, avec quelle traçabilité. J'ai implémenté un pattern d'escalade *consent-first* où le modèle propose l'escalade et l'utilisateur confirme. Chaque appel externe est une décision assumée.

**Sur la mémoire.** Qu'un LLM ne "se souvient" pas au sens fort, et que ce qu'on appelle mémoire est de la reconstruction à chaque tour. Que stocker tout est bruité, et qu'il faut curater : mes règles d'ingestion (seulement les messages user substantiels et les réponses cloud validées) évitent l'effet boule de neige des hallucinations qui s'auto-renforcent.

**Sur les arbitrages.** Que "100% local" est un principe qu'on doit choisir de compromettre parfois. Le fallback cloud existe pour ne pas fournir des réponses médiocres, mais il reste explicite et optionnel.

## Limites connues

- Llama 3.1 8B suit imparfaitement les instructions quand le contexte s'accumule. L'escalade vers un modèle plus grand reste possible mais optionnelle, et sous contrôle explicite de l'utilisateur.
- Le RAG performe moins bien sur les documents très structurés (CV) que sur la prose (portfolio). Solution durable : compléter le corpus avec du contenu narratif.
- La voix Piper reste correcte mais loin d'ElevenLabs. Choix assumé pour maintenir le 100% local sur le TTS.
- L'ingestion incrémentale se déclenche au démarrage du serveur, pas en temps réel. Une note prise dans Obsidian pendant que TARS tourne n'est indexée qu'au prochain redémarrage.

## Prochaines étapes envisagées

- Intégrations Gmail et Google Calendar en lecture seule pour enrichir le contexte quotidien.
- Version mobile via Tailscale et PWA.
- Fine-tuning léger (LoRA) sur la personnalité TARS pour réduire la dépendance au prompt.

## Prérequis

- macOS Apple Silicon (M1+) avec 16 Go de RAM minimum, 24 Go recommandés
- Python 3.11
- Homebrew, Ollama
- Une clé API pour le modèle de fallback (optionnel)

## Installation

À venir.

## Structure du projet

- `main.py` — Serveur FastAPI, routes principales
- `ingest.py` — Ingestion incrémentale des documents dans ChromaDB
- `embeddings.py` — Singleton du modèle d'embeddings
- `memory.py` — Mémoire long-terme sémantique
- `escalation.py` — Logique de fallback vers le modèle cloud
- `audio.py` — STT (Whisper) et TTS (Piper)
- `Modelfile` — Configuration du modèle TARS pour Ollama
- `index.html` — Interface web
- `knowledge/` — Documents perso (gitignoré)
- `voices/` — Voix Piper (gitignoré)

---

*Projet développé par Jules Balzarini.*