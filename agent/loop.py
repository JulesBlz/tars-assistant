"""
La boucle agentique — le coeur du système.

Principe : au lieu d'un aller simple (question -> réponse), on boucle.
À chaque tour :
  1. On appelle le modèle avec l'historique courant + les outils disponibles.
  2. Si le modèle répond du TEXTE -> c'est la réponse finale, on sort.
  3. Si le modèle demande un OUTIL -> on l'exécute, on ajoute le résultat
     à l'historique, et on reboucle. Le modèle voit alors le résultat et
     décide de la suite (répondre, ou appeler un autre outil).

Garde-fous :
  - Limite dure d'itérations (MAX_ITERATIONS) : un 8B part facilement en
    boucle infinie quand un outil échoue ou qu'il ne sait pas s'arrêter.
  - Gestion d'erreur d'outil : si un outil lève une exception, on renvoie
    le message d'erreur AU MODÈLE comme résultat, pour qu'il puisse se
    rattraper ou abandonner proprement, plutôt que de crasher la boucle.

Tout est loggé à chaque tour via le RunLogger.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/tars"))

import time
import json

from tools import TOOL_SCHEMAS, TOOL_FUNCTIONS
from model import call_model, get_model_config
from logger import RunLogger

MAX_ITERATIONS = 5

SYSTEM_PROMPT = """Tu es TARS, l'assistant de Jules. Tu as accès à trois outils :
- search_knowledge : pour toute information personnelle sur Jules (parcours, projets, préférences).
- add_calendar_event : pour créer un événement dans son agenda.
- web_search : pour des faits externes, récents, ou que tu ne connais pas.

Règles :
- Utilise un outil UNIQUEMENT quand c'est nécessaire. Pour une question simple (calcul, définition générale que tu connais), réponds directement sans outil.
- Si une question est ambiguë ou manque d'information (ex: "c'est quand ?" sans contexte), demande une clarification au lieu d'appeler un outil au hasard.
- Si aucune source ne peut répondre, dis-le honnêtement plutôt que d'inventer.
- Quand tu as l'information nécessaire, donne une réponse finale claire et concise, sans appeler d'outil supplémentaire.
"""


def run_agent(question, logger, case_id, model_config, dry_run_calendar=True):
    """
    Exécute l'agent sur une question. Retourne un dict récapitulatif.

    dry_run_calendar : si True, add_calendar_event valide sans écrire
                       (mode éval, reproductible). En usage réel, False.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    tools_called = []
    total_latency_ms = 0
    total_tokens = 0
    stopped_reason = None
    final_answer = None
    hit_limit = False

    for step_index in range(MAX_ITERATIONS):
        # --- Appel du modèle ---
        t0 = time.time()
        response = call_model(messages, TOOL_SCHEMAS, model_config)
        latency_ms = int((time.time() - t0) * 1000)
        total_latency_ms += latency_ms
        step_tokens = response.tokens_prompt + response.tokens_completion
        total_tokens += step_tokens

        # --- Cas 1 : le modèle répond du texte -> réponse finale ---
        if response.output_type == "text":
            final_answer = response.text
            stopped_reason = "final_answer"
            logger.log_step(
                case_id=case_id,
                step_index=step_index,
                model_output_type="text",
                latency_ms=latency_ms,
                tokens_prompt=response.tokens_prompt,
                tokens_completion=response.tokens_completion,
                messages_in_count=len(messages),
            )
            break

        # --- Cas 2 : le modèle demande un ou plusieurs outils ---
        # On ajoute la demande d'outil à l'historique (format OpenAI/Ollama)
        assistant_msg = {
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in response.tool_calls
            ],
        }
        messages.append(assistant_msg)

        # On exécute chaque outil demandé
        for tc in response.tool_calls:
            tools_called.append(tc.name)
            tool_error = None
            tool_result = None

            func = TOOL_FUNCTIONS.get(tc.name)
            if func is None:
                tool_error = f"Outil inconnu : {tc.name}"
                tool_result = {"error": tool_error}
            else:
                try:
                    args = dict(tc.arguments)
                    # Injection du dry-run pour le calendrier (contrainte d'éval)
                    if tc.name == "add_calendar_event":
                        args["dry_run"] = dry_run_calendar
                    tool_result = func(**args)
                    # Un outil peut renvoyer un dict {"error": ...} sans lever d'exception
                    if isinstance(tool_result, dict) and "error" in tool_result:
                        tool_error = tool_result["error"]
                except Exception as e:
                    tool_error = str(e)
                    tool_result = {"error": tool_error}

            # Log du tour (avec l'outil et son résultat)
            logger.log_step(
                case_id=case_id,
                step_index=step_index,
                model_output_type="tool_call",
                tool_name=tc.name,
                tool_args=tc.arguments,
                tool_result=tool_result,
                tool_error=tool_error,
                latency_ms=latency_ms,
                tokens_prompt=response.tokens_prompt,
                tokens_completion=response.tokens_completion,
                messages_in_count=len(messages),
            )

            # On renvoie le résultat de l'outil au modèle (format role: tool)
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": json.dumps(tool_result, ensure_ascii=False),
            })

        # latence/tokens déjà comptés une fois pour l'appel modèle ci-dessus ;
        # on ne les recompte pas par outil.

    else:
        # La boucle for s'est terminée sans break -> limite atteinte
        hit_limit = True
        stopped_reason = "iteration_limit"
        final_answer = "(Limite d'itérations atteinte sans réponse finale.)"

    return {
        "final_answer": final_answer,
        "tools_called": tools_called,
        "num_steps": step_index + 1,
        "stopped_reason": stopped_reason,
        "total_latency_ms": total_latency_ms,
        "total_tokens": total_tokens,
        "hit_iteration_limit": hit_limit,
    }


# ============================================================
# Test manuel sur quelques questions
# ============================================================
if __name__ == "__main__":
    config = get_model_config()
    print(f"Config : {config}\n")

    test_questions = [
        "Combien font 47 fois 89 ?",                    # aucun outil attendu
        "Quelle est la date de NeurIPS 2026 ?",         # web_search attendu
        "Ajoute un rdv dentiste le 2026-09-15 à 14h",   # add_calendar_event (dry-run)
    ]

    with RunLogger(config["model"], config["backend"]) as logger:
        for i, q in enumerate(test_questions):
            case_id = f"manual_{i+1:03d}"
            print(f"\n{'='*60}")
            print(f"Q: {q}")
            result = run_agent(q, logger, case_id, config, dry_run_calendar=True)
            print(f"Outils appelés : {result['tools_called']}")
            print(f"Tours : {result['num_steps']} | Arrêt : {result['stopped_reason']}")
            print(f"Latence : {result['total_latency_ms']} ms | Tokens : {result['total_tokens']}")
            print(f"Réponse : {result['final_answer'][:200] if result['final_answer'] else '(vide)'}")
        print(f"\n\nLog écrit dans : {logger.log_path}")
