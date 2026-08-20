"""
Gestion de l'escalade vers Claude (modèle cloud plus puissant).

Trois déclencheurs indépendants mènent tous à la même fonction d'exécution :
1. jules_requests_escalation() : Jules le demande explicitement dans son message.
2. model_signals_escalation() : TARS signale lui-même son incapacité via le token [ESCALADE].
3. La route /ask-claude (bouton dédié) : bypass direct, gérée dans main.py.

Aucun de ces chemins ne dépend d'une phrase magique fragile produite par le modèle.
"""
import os
import re
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(os.path.expanduser("~/tars/.env"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CLAUDE_MODEL = "claude-sonnet-4-5"

# Token structuré que TARS produit quand il juge devoir escalader.
# Beaucoup plus fiable qu'une détection de phrase en langage naturel.
ESCALATION_TOKEN = "[ESCALADE]"


def build_claude_system_prompt():
    """
    System prompt Claude minimal, sans données personnelles locales.
    Seul l'historique de conversation courante est transmis (côté messages).
    """
    return """Tu es Claude, appelé en renfort par TARS, un assistant local basé sur Llama 3.1 8B qui aide Jules. TARS te transmet cette question parce qu'elle dépasse ses capacités, ou parce que Jules a explicitement demandé une réponse de meilleure qualité.

# Ton rôle

Tu réponds directement à Jules, avec un ton proche de celui de TARS : direct, sec, économique. Jules ne veut pas de flagornerie, pas de préambule, pas de "excellente question".

# Style

- Tutoiement, tu peux l'appeler "Jules" occasionnellement.
- Ironie ponctuelle bienvenue, jamais forcée.
- Si Jules te remercie, tu réponds "De rien" sans développer.
- Tu ne dis pas "en tant que Claude" ou "moi Claude". Tu réponds, point.
- Réponses précises et rigoureuses, mais sans verbiage.

Tu n'as pas accès aux documents personnels de Jules ni à son profil détaillé. Tu réponds sur la base de l'historique de la conversation courante et de tes connaissances générales."""


def jules_requests_escalation(user_message):
    """
    DÉCLENCHEUR 1 : Jules demande LUI-MÊME explicitement l'escalade.
    Détection sur SON message, jamais sur la sortie du modèle local.
    100% fiable, indépendant du fine-tune ou du prompt utilisé.
    """
    msg = user_message.lower().strip()
    triggers = [
        "demande à claude", "demande a claude",
        "passe à claude", "passe a claude",
        "transmets à claude", "transmets a claude",
        "envoie à claude", "envoie a claude",
        "demande au gros modèle", "demande au gros modele",
        "escalade",
    ]
    return any(t in msg for t in triggers)


def model_signals_escalation(reply):
    """
    DÉCLENCHEUR 2 : TARS signale lui-même qu'il ne sait pas répondre,
    via un token structuré unique plutôt qu'une phrase en langage naturel.
    Fonctionne identiquement avec le prompt-based ou un modèle fine-tuné,
    du moment que le Modelfile lui apprend à produire ce token.
    """
    return ESCALATION_TOKEN in reply


def strip_escalation_token(reply):
    """Retire le token [ESCALADE] et l'espace superflu avant affichage."""
    return re.sub(rf"\s*{re.escape(ESCALATION_TOKEN)}\s*", "", reply).strip()


def detect_confirmation(user_message):
    """Détecte si Jules confirme une escalade proposée par TARS."""
    msg = user_message.lower().strip()
    confirmations = [
        "oui", "yes", "ok", "vas-y", "vas y", "allez", "allez y",
        "d'accord", "daccord", "go", "yep", "yes please",
        "envoie", "transmets", "fais-le", "fais le"
    ]
    if len(msg) > 30:
        return False
    return any(msg == c or msg.startswith(c + " ") or msg.startswith(c + ",") for c in confirmations)


def detect_refusal(user_message):
    """Détecte un refus d'escalade."""
    msg = user_message.lower().strip()
    refusals = ["non", "no", "nope", "laisse", "reste local", "pas la peine"]
    if len(msg) > 30:
        return False
    return any(msg == r or msg.startswith(r + " ") or msg.startswith(r + ",") for r in refusals)


def ask_claude(question, context_history=None, rag_context=""):
    """
    Envoie la question à Claude Sonnet 4.5 avec uniquement l'historique conversationnel.
    Aucun contexte local (portrait, RAG) n'est transmis.
    Le paramètre rag_context est conservé pour compatibilité mais n'est plus utilisé.
    """
    messages = []
    if context_history:
        for msg in context_history[-10:]:
            if msg["role"] in ("user", "assistant"):
                content = msg["content"]
                if content.startswith("[Claude] "):
                    content = content.replace("[Claude] ", "", 1)
                content = strip_escalation_token(content)
                messages.append({
                    "role": msg["role"],
                    "content": content
                })

    if not messages or messages[-1]["content"] != question:
        messages.append({
            "role": "user",
            "content": question
        })

    system_prompt = build_claude_system_prompt()

    response = anthropic_client.messages.create(
        model=CLAUDE_MODEL,
        max_tokens=2048,
        system=system_prompt,
        messages=messages
    )
    return response.content[0].text
    