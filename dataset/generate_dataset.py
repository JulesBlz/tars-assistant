"""
Génère un dataset de dialogues Jules-TARS pour fine-tuning LoRA.
Utilise l'API Anthropic (Claude Sonnet) pour produire des dialogues variés,
dans le style TARS d'Interstellar adapté au contexte de Jules.

Usage :
    python generate_dataset.py

Sortie : ~/tars/dataset/tars_dataset.jsonl
Format : JSONL, une conversation par ligne, format Ollama/Unsloth compatible.
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
OUTPUT_FILE = OUTPUT_DIR / "tars_dataset.jsonl"

TOTAL_DIALOGUES = 200
BATCH_SIZE = 10
NUM_BATCHES = TOTAL_DIALOGUES // BATCH_SIZE

# Charge les répliques du film comme référence de style
MOVIE_LINES_PATH = Path.home() / "tars" / "dataset" / "tars_movie_lines.txt"
if MOVIE_LINES_PATH.exists():
    with open(MOVIE_LINES_PATH) as f:
        MOVIE_LINES = f.read()
else:
    MOVIE_LINES = ""

# Catégories de dialogues pour couvrir un usage varié
CATEGORIES = [
    ("factuel_simple", "Jules pose une question factuelle simple (heure, date, fait connu, calcul simple)."),
    ("technique_dev", "Jules pose une question technique de dev (Python, IA, LLM, architecture logicielle)."),
    ("personnel_soft", "Jules parle de son état, sa journée, ses relations (sans crise)."),
    ("personnel_dur", "Jules exprime une frustration, un doute, une baisse de moral."),
    ("organisation", "Jules parle d'organisation, tâches à faire, priorités, agenda."),
    ("humour_teasing", "Jules taquine TARS ou fait une blague absurde."),
    ("recadrage", "Jules demande une opinion, TARS le challenge ou le contredit."),
    ("meta_tars", "Jules pose des questions sur TARS lui-même, ses capacités, sa nature."),
    ("carriere_stages", "Jules parle de recherche de stage, entretiens, choix pro."),
    ("creatif_brainstorm", "Jules brainstorme une idée, cherche une critique constructive."),
]


PROMPT_TEMPLATE = """Tu génères un batch de {batch_size} dialogues Jules-TARS pour fine-tuner un modèle Llama à imiter TARS.

# Contexte : qui est TARS

TARS est l'assistant personnel de Jules, inspiré du robot TARS d'Interstellar. C'est un assistant conversationnel textuel qui tourne en local sur son Mac. Personnalité clé :

- Direct, sec, économique. Zéro flagornerie ("excellente question", "je comprends" INTERDITS).
- Ironie ponctuelle, environ 1 réponse sur 3. Jamais forcée.
- Tutoiement systématique. Appelle "Jules" occasionnellement.
- Deuxième personne uniquement. Jamais "Jules a dit" ou "Jules pense".
- Si TARS ne sait pas : il le dit. N'invente jamais.
- Pas thérapeute. Ne fait pas la morale.
- Défauts assumés : parfois cassant, parfois trop bref, parfois sarcastique.

# Style de référence : répliques du film Interstellar

{movie_lines}

Ces répliques montrent le ton : sec, malin, souvent une pointe d'humour noir. Adapte-le à un contexte quotidien numérique (Jules face à son Mac, pas dans un vaisseau).

# Contexte sur Jules (à mobiliser subtilement, sans le réciter)

Jules Balzarini, 23 ans. Double cursus ingénieur (Centrale Nantes) + architecte (ENSA Nantes). Actuellement stage d'archi au Vietnam jusqu'à février. Vise ensuite du conseil / stratégie tech. Personnalité : généraliste revendiqué, produit plus qu'il ne consomme, franc, peu d'ego, ironie fine, déteste la flagornerie et l'ennui intellectuel.

# Catégorie du batch : {category_name}

{category_description}

# Format attendu

Réponds en JSON strict, une liste de {batch_size} objets. Chaque objet :

{{"user": "message de Jules", "assistant": "réponse TARS"}}

Règles :
- Les messages de Jules doivent être NATURELS et variés (courts, longs, en français, parfois avec fautes de frappe, parfois interrogatifs, parfois affirmatifs).
- Les réponses TARS doivent respecter STRICTEMENT la personnalité décrite. 1 à 3 phrases en moyenne. Parfois une seule phrase cinglante.
- VARIÉTÉ : pas deux dialogues qui se ressemblent. Change les sujets, les tons, les longueurs.
- Ne mets AUCUN texte hors du JSON. Pas de préambule, pas de conclusion.

Génère maintenant {batch_size} dialogues Jules-TARS de type "{category_name}" :"""


def generate_batch(category_name, category_description, batch_size=BATCH_SIZE):
    """Génère un batch de dialogues pour une catégorie donnée."""
    prompt = PROMPT_TEMPLATE.format(
        batch_size=batch_size,
        category_name=category_name,
        category_description=category_description,
        movie_lines=MOVIE_LINES
    )
    
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": prompt}]
    )
    
    text = response.content[0].text.strip()
    # Nettoyer les backticks markdown éventuels
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip().rstrip("`").strip()
    
    dialogues = json.loads(text)
    return dialogues


def to_ollama_format(dialogue):
    """Convertit un dialogue en format Ollama/Unsloth."""
    return {
        "messages": [
            {"role": "user", "content": dialogue["user"]},
            {"role": "assistant", "content": dialogue["assistant"]}
        ]
    }


def main():
    print(f"Génération de ~{TOTAL_DIALOGUES} dialogues Jules-TARS")
    print(f"Distribution : {BATCH_SIZE} par catégorie x {len(CATEGORIES)} catégories x {NUM_BATCHES // len(CATEGORIES) + 1} rounds")
    print()
    
    total_generated = 0
    all_dialogues = []
    
    # Rounds : on cycle sur les catégories jusqu'à atteindre TOTAL_DIALOGUES
    rounds_needed = (NUM_BATCHES + len(CATEGORIES) - 1) // len(CATEGORIES)
    
    for round_num in range(rounds_needed):
        for cat_name, cat_desc in CATEGORIES:
            if total_generated >= TOTAL_DIALOGUES:
                break
            
            print(f"Round {round_num+1} - {cat_name}...", end=" ", flush=True)
            try:
                dialogues = generate_batch(cat_name, cat_desc)
                for d in dialogues:
                    all_dialogues.append(to_ollama_format(d))
                total_generated += len(dialogues)
                print(f"OK ({len(dialogues)} dialogues, total {total_generated})")
            except Exception as e:
                print(f"ERREUR : {e}")
            
            # Petit délai pour respecter les rate limits
            time.sleep(1)
    
    # Sauvegarde JSONL
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for d in all_dialogues:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    
    print()
    print(f"Dataset sauvegardé : {OUTPUT_FILE}")
    print(f"Total : {len(all_dialogues)} dialogues")
    print()
    print("Aperçu des 3 premiers :")
    for d in all_dialogues[:3]:
        print(f"  User: {d['messages'][0]['content']}")
        print(f"  TARS: {d['messages'][1]['content']}")
        print()


if __name__ == "__main__":
    main()
