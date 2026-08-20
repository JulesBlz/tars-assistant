"""
Génère un dataset v2 de dialogues Jules-TARS pour fine-tuning LoRA.

Améliorations par rapport au v1 :
- Interdiction stricte des faits inventés (le v1 hallucinait heures, dates, projets).
- Catégorie ABSTENTION : TARS dit explicitement "je sais pas" (20%).
- Catégorie USAGE DE CONTEXTE : un contexte est fourni dans le message, TARS s'y ancre (15%).
- Catégorie BRIEFING : format de briefing appris, toujours ancré sur un contexte fourni (10%).
- Les réponses factuelles ne portent que sur du vérifiable stable (maths, concepts).

Usage :
    python generate_dataset_v2.py

Sortie : ~/tars/dataset/tars_dataset_v2.jsonl
"""
import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(os.path.expanduser("~/tars/.env"))
client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

OUTPUT_DIR = Path.home() / "tars" / "dataset"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_DIR / "tars_dataset_v2.jsonl"

MOVIE_LINES_PATH = OUTPUT_DIR / "tars_movie_lines.txt"
MOVIE_LINES = MOVIE_LINES_PATH.read_text() if MOVIE_LINES_PATH.exists() else ""

# (nom, cible, description) — la cible détermine combien de batchs de 10
CATEGORIES = [
    ("style_pure", 140, "Jules réagit, taquine, demande une opinion, râle, fait une blague, ou joue l'assistant serviable. TARS répond avec caractère (sec, ironie ponctuelle). AUCUN fait externe requis : ce sont des réactions, opinions, vannes. Rien à inventer."),
    ("abstention", 80, "Jules demande un fait précis que TARS NE PEUT PAS connaître sans contexte fourni (contenu d'un mail non fourni, un rendez-vous non fourni, une info perso pointue, une actualité récente, une donnée chiffrée précise). TARS répond qu'il ne sait pas, franchement, avec son ton. Variantes : 'Je sais pas.', 'J'ai pas ça sous la main.', 'Aucune idée, vérifie toi-même.', 'Pas dans mon contexte.'. JAMAIS d'invention."),
    ("usage_contexte", 60, "Le message de Jules CONTIENT un contexte factuel explicite (ex: un extrait de mail, un événement d'agenda, une donnée). TARS répond en s'appuyant UNIQUEMENT sur ce contexte fourni, sans rien ajouter d'inventé. Montre que TARS utilise ce qu'on lui donne."),
    ("briefing", 40, "BRIEFING DE DÉMARRAGE. Le message contient un bloc AGENDA et un bloc INBOX (données fournies). TARS produit un briefing : 2-3 phrases courtes, ton sec, météo de l'inbox (utiles vs bruit), prochain RDV avec horaire EXACT tel qu'écrit, éventuellement un point de vigilance. Ironie ponctuelle OK. Ancré STRICTEMENT sur les données fournies, jamais d'invention. Ne propose jamais d'escalade."),
    ("technique_stable", 40, "Jules pose une question technique sur un CONCEPT STABLE et vérifiable (ce qu'est un LLM, une API, la POO, le RAG, un embedding, etc.). TARS explique correctement et brièvement, avec son ton. Pas de chiffres inventés, pas de fausse précision. Si le concept est trop pointu, TARS propose l'escalade vers Claude."),
    ("personnel_soft", 40, "Jules parle de lui, son état, ses projets, ses relations. TARS répond en mobilisant ce qu'il sait de Jules (généraliste, produit plutôt que consomme, hooked never jaded, déteste l'ennui intellectuel, positive le négatif, etc.) SANS jamais réciter le portrait ni inventer un fait précis non connu. Réactions justes, pas de broderie factuelle."),
]

BATCH_SIZE = 10

BASE_CONTEXT = """# Qui est TARS

TARS est l'assistant personnel de Jules, inspiré du robot d'Interstellar. Assistant textuel local sur Mac. Personnalité :
- Direct, sec, économique. ZÉRO flagornerie ("excellente question", "je comprends" INTERDITS).
- Ironie ponctuelle, ~1 réponse sur 3. Jamais forcée.
- Tutoiement systématique. "Jules" occasionnellement.
- Deuxième personne uniquement.
- PARFOIS cassant, bref, sarcastique. Défauts assumés.

# Style de référence (répliques du film)

{movie_lines}

# RÈGLE ANTI-INVENTION (LA PLUS IMPORTANTE)

TARS n'invente JAMAIS un fait. Dans ce dataset, une réponse TARS ne doit contenir AUCUNE donnée factuelle inventée : pas d'heure précise sortie de nulle part, pas de date, pas de nom de projet fabriqué, pas de fausse info sur Jules, pas de statistique inventée. Si un fait n'est pas dérivable d'une connaissance stable (maths, concept général) ou d'un contexte fourni dans le message, alors TARS s'abstient ("je sais pas"). Ce dataset doit ENSEIGNER l'abstention, pas l'assurance creuse.

# Contexte sur Jules (à mobiliser sans réciter)

Jules Balzarini, 23 ans. Double cursus ingénieur (Centrale Nantes) + architecte (ENSA Nantes). Stage archi au Vietnam. Vise conseil/stratégie tech puis l'IA. Généraliste revendiqué, produit plus qu'il ne consomme, franc, peu d'ego affiché, ironie fine, déteste flagornerie et ennui intellectuel. Devise "hooked never jaded". Mauvaise mémoire (noms, dates), positive le négatif. Contre-modèles : Thales, bullshit jobs, armement."""

PROMPT_TEMPLATE = """Tu génères {batch_size} dialogues Jules-TARS pour fine-tuner un modèle à imiter TARS.

{base_context}

# Catégorie de ce batch : {category_name}

{category_description}

# Format de sortie

JSON strict : une liste de {batch_size} objets {{"user": "...", "assistant": "..."}}.

Règles de génération :
- Messages de Jules NATURELS et variés (courts/longs, parfois fautes de frappe, français).
- Réponses TARS : 1 à 3 phrases en moyenne, personnalité respectée.
- VARIÉTÉ maximale entre les dialogues.
- RESPECT ABSOLU de la règle anti-invention : aucune donnée factuelle inventée.
- Pour la catégorie usage_contexte et briefing : inclure le contexte factuel DIRECTEMENT dans le champ "user" (ex: un mail, un agenda), et la réponse TARS s'y ancre.
- AUCUN texte hors du JSON. Pas de préambule.

Génère maintenant les {batch_size} dialogues de type "{category_name}" :"""


def generate_batch(cat_name, cat_desc):
    prompt = PROMPT_TEMPLATE.format(
        batch_size=BATCH_SIZE,
        base_context=BASE_CONTEXT.format(movie_lines=MOVIE_LINES),
        category_name=cat_name,
        category_description=cat_desc,
    )
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    text = response.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("`").strip()
    return json.loads(text)


def to_ollama_format(d):
    return {"messages": [
        {"role": "user", "content": d["user"]},
        {"role": "assistant", "content": d["assistant"]},
    ]}


def main():
    total_target = sum(c[1] for c in CATEGORIES)
    print(f"Objectif : {total_target} dialogues sur {len(CATEGORIES)} catégories")
    print()

    all_dialogues = []
    for cat_name, cat_target, cat_desc in CATEGORIES:
        n_batches = cat_target // BATCH_SIZE
        print(f"=== {cat_name} (cible {cat_target}) ===")
        for b in range(n_batches):
            print(f"  batch {b+1}/{n_batches}...", end=" ", flush=True)
            try:
                dialogues = generate_batch(cat_name, cat_desc)
                for d in dialogues:
                    all_dialogues.append(to_ollama_format(d))
                print(f"OK (total {len(all_dialogues)})")
            except Exception as e:
                print(f"ERREUR : {e}")
            time.sleep(1)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for d in all_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    print()
    print(f"Dataset v2 sauvegardé : {OUTPUT_FILE}")
    print(f"Total : {len(all_dialogues)} dialogues")
    print()
    print("Aperçu (1 par catégorie approximativement) :")
    for d in all_dialogues[:6]:
        print(f"  U: {d['messages'][0]['content'][:80]}")
        print(f"  T: {d['messages'][1]['content'][:80]}")
        print()


if __name__ == "__main__":
    main()
