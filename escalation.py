import os
from dotenv import load_dotenv
from anthropic import Anthropic

load_dotenv(os.path.expanduser("~/tars/.env"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

CLAUDE_MODEL = "claude-sonnet-4-5"


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


def detect_escalation_proposal(reply):
    """
    Détecte si la réponse de TARS contient une proposition d'escalade vers Claude.
    """
    reply_lower = reply.lower()
    if "claude" not in reply_lower:
        return False
    if "?" not in reply:
        return False
    escalation_keywords = [
        "transmette",
        "transmettre",
        "passer à claude",
        "passe à claude",
        "escalade",
        "escalader",
        "gros modèle",
        "demander à claude",
        "demande à claude",
    ]
    return any(kw in reply_lower for kw in escalation_keywords)


def detect_confirmation(user_message):
    """
    Détecte si l'utilisateur confirme une escalade.
    """
    msg = user_message.lower().strip()
    confirmations = [
        "oui", "yes", "ok", "vas-y", "vas y", "allez", "allez y",
        "d'accord", "daccord", "go", "yep", "yes please",
        "envoie", "transmets", "escalade", "fais-le", "fais le"
    ]
    if len(msg) > 30:
        return False
    return any(msg == c or msg.startswith(c + " ") or msg.startswith(c + ",") for c in confirmations)


def detect_refusal(user_message):
    """
    Détecte un refus d'escalade.
    """
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