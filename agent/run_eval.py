"""
Lanceur d'évaluation : exécute tout le jeu de test sur UN modèle et écrit le log.

Usage :
    AGENT_MODEL=llama3.1:8b python3 run_eval.py
    AGENT_MODEL=tars-ft-v2 python3 run_eval.py
    AGENT_BACKEND=anthropic AGENT_MODEL=claude-sonnet-4-5 python3 run_eval.py

Le modèle et le backend sont lus depuis les variables d'environnement (via model.py).
Chaque run produit un fichier de log JSONL horodaté et identifié par le modèle.

L'évaluation tourne en dry-run calendrier (rien n'est écrit dans le vrai agenda)
et avec le cache web activé (résultats reproductibles). Ces deux contraintes
rendent chaque run rejouable et comparable.
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/tars"))

import json
import time
from pathlib import Path

from loop import run_agent
from model import get_model_config
from logger import RunLogger

TESTSET_PATH = Path(__file__).parent / "testset.json"


def load_testset():
    with open(TESTSET_PATH, encoding="utf-8") as f:
        data = json.load(f)
    return data["cases"]


def main():
    config = get_model_config()
    cases = load_testset()

    print(f"{'='*60}")
    print(f"ÉVALUATION")
    print(f"  Backend : {config['backend']}")
    print(f"  Modèle  : {config['model']}")
    print(f"  Cas     : {len(cases)}")
    print(f"{'='*60}\n")

    t_start = time.time()

    with RunLogger(config["model"], config["backend"]) as logger:
        for i, case in enumerate(cases):
            case_id = case["id"]
            question = case["question"]
            expected = case.get("expected_tools", [])

            print(f"[{i+1}/{len(cases)}] {case_id} : {question[:55]}")

            result = run_agent(
                question=question,
                logger=logger,
                case_id=case_id,
                model_config=config,
                dry_run_calendar=True,   # éval : on n'écrit jamais dans le vrai agenda
            )

            # Log du récapitulatif de cas (factuel, sans jugement)
            logger.log_case(
                case_id=case_id,
                question=question,
                tools_called=result["tools_called"],
                tools_expected=expected,
                num_steps=result["num_steps"],
                stopped_reason=result["stopped_reason"],
                final_answer=result["final_answer"],
                total_latency_ms=result["total_latency_ms"],
                total_tokens=result["total_tokens"],
                hit_iteration_limit=result["hit_iteration_limit"],
            )

            # Feedback console minimal
            called = result["tools_called"] or ["(aucun)"]
            print(f"      → outils: {called} | {result['num_steps']} tours | {result['total_latency_ms']} ms")

        log_path = logger.log_path

    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"Terminé en {elapsed:.1f}s")
    print(f"Log : {log_path}")
    print(f"{'='*60}")
    print(f"\nPour scorer ce run : python3 score.py {log_path.name}")


if __name__ == "__main__":
    main()
