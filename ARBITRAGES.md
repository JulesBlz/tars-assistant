# TARS — arbitrages techniques

Ce document regroupe les principales décisions techniques prises pendant le développement de TARS, avec pour chacune : le contexte, l'option retenue, les alternatives écartées, et le trade-off assumé.

L'objectif est de rendre lisibles les raisonnements derrière les choix, plutôt que de décrire uniquement le résultat.

---

## 1. Modèle local : Llama 3.1 8B

**Contexte.** Le cœur de TARS est un LLM tournant sur un Mac M5 24 Go. Le choix conditionne la personnalité tenable, la latence, et la taille de contexte disponible.

**Décision.** Llama 3.1 8B en quantization Q4_K_M, servi par Ollama.

**Alternatives écartées.**

- *Mistral Nemo 12B* : testé en conditions réelles. Ne respecte pas le system prompt de personnalité et retombe systématiquement en mode assistant générique.
- *Qwen 2.5 14B* : latence perceptible sur Apple Silicon, écosystème moins mature.
- *Llama 3.3 70B* : hors de portée mémoire.

**Trade-off.** Llama 8B est limité sur le raisonnement complexe et hallucine sur les faits précis. Ces limites sont adressées par le fallback cloud, jamais par du prompt engineering additionnel.

---

## 2. Runtime : Ollama

**Contexte.** Le runtime détermine la vitesse d'inférence, l'ergonomie de développement, et l'API disponible.

**Décision.** Ollama.

**Alternatives écartées.**

- *llama.cpp direct* : plus performant en pratique sur Apple Silicon, mais gain marginal à l'échelle du projet, et setup plus lourd.
- *MLX* : le plus rapide sur Apple Silicon mais écosystème plus jeune.

**Trade-off.** Une couche d'abstraction en plus qui coûte quelques tokens/seconde, contre une API HTTP standard, une gestion des modèles par nom, et un Modelfile déclaratif pour la configuration.

**Point d'architecture.** Le runtime est découplé de l'orchestration : FastAPI parle à Ollama via HTTP standard. Changer de runtime n'impacterait que quelques lignes. Ce loose coupling est appliqué partout dans le projet.

---

## 3. Personnalité : Modelfile avec few-shot examples

**Contexte.** TARS doit avoir une personnalité forte qui va à l'encontre du tuning RLHF de Llama, conçu pour être poli et serviable.

**Décision.** System prompt dans le Modelfile, avec exemples de dialogues (few-shot) plutôt que description abstraite.

**Alternatives écartées.**

- *Fine-tuning LoRA* : solution structurellement plus stable, mais l'investissement (dataset, entraînement) n'est pas justifié tant que le prompt tient.
- *Prompt injecté dynamiquement à chaque requête* : plus flexible mais double la consommation de tokens et complexifie l'orchestration.

**Trade-off.** Le prompt engineering est fragile : sur des tours nombreux, le modèle régresse vers ses habitudes d'entraînement. Un fine-tuning serait plus stable mais reste hors scope.

**Observation.** Les few-shot examples ont un impact largement supérieur aux descriptions abstraites. Le modèle imite plus qu'il n'obéit.

---

## 4. RAG : ChromaDB + embeddings multilingues

**Contexte.** TARS mobilise des documents personnels dans ses réponses.

**Décision.** ChromaDB local, embeddings `paraphrase-multilingual-MiniLM-L12-v2`, chunks de 400 mots avec 50 mots de recouvrement, retrieval top-6 avec seuil de distance à 1.2.

**Alternatives écartées.**

- *FAISS* : plus rapide sur gros corpus, mais pas de persistance native. Overkill pour ce volume.
- *Bases vectorielles managées (Pinecone, Weaviate)* : incompatibles avec le principe "100% local".
- *Embeddings anglais* : inadaptés à un corpus majoritairement français.

**Trade-off.** Un modèle d'embeddings plus lourd (bge-m3, e5-mistral) améliorerait le retrieval, au prix d'un démarrage plus long et d'une ingestion plus lente.

**Enseignement.** La similarité sémantique favorise les documents en prose au détriment des documents structurés. Un CV en bullets se fait systématiquement écraser par un portfolio narratif au retrieval. La solution passe par le corpus, pas par le seuil.

---

## 5. Fallback cloud : consent-first

**Contexte.** Certaines tâches dépassent les capacités du modèle local. Le standard industriel est un routing automatique côté serveur, invisible pour l'utilisateur.

**Décision.** Deux modes explicites : (a) auto-proposition par le modèle local quand il détecte une limite ; (b) escalade manuelle par bouton. Aucun appel externe sans confirmation utilisateur.

**Alternatives écartées.**

- *Routing automatique par heuristique* : opaque pour l'utilisateur, incompatible avec le positionnement local-first.
- *Tool calling natif* : Llama 8B est peu fiable sur le function calling. Le pattern retenu (modèle propose, utilisateur valide, serveur intercepte) est plus robuste.

**Trade-off.** Un action utilisateur supplémentaire là où un système automatique serait plus fluide. Assumé : la confidentialité prime sur la fluidité.

**Positionnement.** TARS n'est pas un assistant qui appelle silencieusement le cloud, c'est un assistant local qui demande la permission d'aller ailleurs quand c'est nécessaire.

---

## 6. Cloud : API Anthropic

**Contexte.** Le fallback doit être fiable et abordable pour un usage individuel.

**Décision.** API Anthropic, modèle Sonnet.

**Alternatives écartées.**

- *OpenAI GPT-4* : standard de l'industrie, écarté pour préférence de qualité d'écriture.
- *Groq* : gratuit et rapide, écarté sur des considérations de plateforme.
- *Gemini* : tier gratuit indisponible sur mon compte.

**Trade-off.** Coût payant vs alternatives gratuites. Négligeable à l'échelle d'un usage personnel (5€ couvrent plusieurs mois).

---

## 7. Contexte transmis au cloud : historique conversationnel uniquement

**Contexte.** Quand Claude est invoqué, on peut transmettre plus ou moins de contexte : rien, l'historique de la conversation courante, ou aussi le portrait et les résultats du RAG.

**Décision.** Uniquement l'historique de conversation courante. Aucun document personnel, aucun profil, aucun résultat de RAG n'est transmis à l'API externe.

**Alternatives écartées.**

- *Envoyer portrait + RAG à Claude* : améliorerait la qualité des réponses, mais violerait la promesse "les données personnelles ne sortent pas du Mac".
- *Envoyer un résumé compressé du profil* : compromis intermédiaire, mais introduit une décision floue sur ce qui est safe à transmettre.

**Trade-off.** Claude a moins de contexte et répond parfois de manière plus générique sur les sujets personnels. C'est le prix de la cohérence architecturale.

---

## 8. Mémoire long-terme : ingestion sélective

**Contexte.** Une mémoire persistante entre sessions est nécessaire pour que TARS puisse retrouver des échanges anciens. Tout stocker serait naïf.

**Décision.** Collection ChromaDB dédiée, avec deux règles strictes : (a) messages utilisateur de plus de 30 mots uniquement ; (b) réponses cloud uniquement, jamais les réponses du modèle local.

**Alternatives écartées.**

- *Tout ingérer* : bruit important, et surtout risque d'auto-renforcer les hallucinations du modèle local.
- *Résumer les conversations à la fin de chaque session avec le modèle cloud* : violerait le principe local-first.

**Trade-off.** Le modèle ne conserve pas la mémoire de ses propres réponses locales. En pratique, la question originale suffit à reprendre le contexte.

**Raisonnement de fond.** Une IA qui apprend de ses propres hallucinations dérive. Ne pas ingérer les réponses du modèle local est un choix de robustesse, pas de complétude.

---

*Document maintenu à jour au fur et à mesure des décisions structurantes.*