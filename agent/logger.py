"""
Logging JSONL de l'exécution de l'agent.

Deux niveaux d'enregistrement, écrits dans le même fichier .jsonl :
- "step" : un enregistrement par tour de boucle (appel modèle + outil éventuel).
- "case" : un enregistrement récapitulatif par cas de test complet.

Principe : le log est un enregistrement FACTUEL de ce qui s'est passé.
Il ne contient AUCUN jugement (pas de "correct/incorrect"). Le scoring est
fait séparément par score.py, qui lit ce log et le ground truth. Cela permet
de recalculer les métriques sans réexécuter l'agent (les appels coûtent du
temps et parfois de l'argent).
"""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

LOG_DIR = Path.home() / "tars" / "agent" / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)


class RunLogger:
    """
    Un logger par run (une passe complète du jeu de test avec un modèle donné).
    Le run_id identifie la passe : horodatage + modèle, pour comparer les runs.
    """

    def __init__(self, model_name, backend):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        safe_model = model_name.replace(":", "-").replace("/", "-")
        self.run_id = f"{timestamp}_{backend}_{safe_model}"
        self.model_name = model_name
        self.backend = backend
        self.log_path = LOG_DIR / f"{self.run_id}.jsonl"
        self._file = open(self.log_path, "a", encoding="utf-8")

    def _write(self, record):
        record["run_id"] = self.run_id
        record["model"] = self.model_name
        record["backend"] = self.backend
        self._file.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._file.flush()

    def log_step(self, case_id, step_index, model_output_type,
                 tool_name=None, tool_args=None, tool_result=None,
                 tool_error=None, latency_ms=0,
                 tokens_prompt=0, tokens_completion=0,
                 messages_in_count=0):
        """Enregistre un tour de boucle."""
        # On tronque le résultat d'outil pour garder le log lisible
        result_preview = None
        if tool_result is not None:
            result_str = json.dumps(tool_result, ensure_ascii=False) if not isinstance(tool_result, str) else tool_result
            result_preview = result_str[:500]

        self._write({
            "record_type": "step",
            "case_id": case_id,
            "step_index": step_index,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "messages_in_count": messages_in_count,
            "model_output_type": model_output_type,
            "tool_name": tool_name,
            "tool_args": tool_args,
            "tool_result_preview": result_preview,
            "tool_error": tool_error,
            "latency_ms": latency_ms,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
        })

    def log_case(self, case_id, question, tools_called, tools_expected,
                 num_steps, stopped_reason, final_answer,
                 total_latency_ms, total_tokens, hit_iteration_limit):
        """Enregistre le récapitulatif d'un cas complet."""
        self._write({
            "record_type": "case",
            "case_id": case_id,
            "question": question,
            "tools_called": tools_called,
            "tools_expected": tools_expected,
            "num_steps": num_steps,
            "stopped_reason": stopped_reason,
            "final_answer": final_answer,
            "total_latency_ms": total_latency_ms,
            "total_tokens": total_tokens,
            "hit_iteration_limit": hit_iteration_limit,
        })

    def close(self):
        self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
