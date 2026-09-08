"""
Scoring d'un run d'évaluation.

Lit un fichier de log JSONL (produit par run_eval.py) et le compare au ground
truth du jeu de test (testset.json). Calcule les métriques SÉPARÉMENT de la
collecte : on peut rescorer sans réexécuter l'agent, et changer les règles de
scoring sans perdre les données.

Métriques calculées :
  1. Tool selection accuracy : le bon ensemble d'outils a-t-il été appelé ?
  2. Argument correctness : les arguments contiennent-ils les valeurs attendues ?
  3. End-to-end success : réponse pertinente (answerable) ou refus propre (unanswerable) ?
  4. Latence et coût agrégés.

Usage :
    python3 score.py <nom_du_log.jsonl>
    python3 score.py           (score le log le plus récent)
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/tars"))

import json
import glob
from pathlib import Path

TESTSET_PATH = Path(__file__).parent / "testset.json"
LOG_DIR = Path(__file__).parent / "logs"

# Table de prix (USD par million de tokens) pour le calcul de coût.
# Local = gratuit. Anthropic = tarifs indicatifs, à ajuster.
PRICING = {
    "ollama": {"prompt": 0.0, "completion": 0.0},
    "anthropic": {"prompt": 3.0, "completion": 15.0},  # ordre de grandeur Sonnet
}


def load_testset():
    with open(TESTSET_PATH, encoding="utf-8") as f:
        return {c["id"]: c for c in json.load(f)["cases"]}


def load_log(log_name):
    path = LOG_DIR / log_name
    steps = []
    cases = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["record_type"] == "step":
                steps.append(rec)
            elif rec["record_type"] == "case":
                cases.append(rec)
    return steps, cases


def score_tool_selection(called, expected, order_matters):
    """
    Le bon ensemble d'outils a-t-il été appelé ?
    - expected vide : succès si called est vide aussi (aucun outil attendu).
    - order_matters : les outils doivent apparaître dans l'ordre attendu.
    - sinon : même ensemble d'outils, ordre libre.
    """
    if not expected:
        return len(called) == 0
    if order_matters:
        # Sous-séquence ordonnée : chaque outil attendu apparaît dans l'ordre
        it = iter(called)
        return all(tool in it for tool in expected)
    else:
        return set(called) == set(expected)


def score_arguments(steps_for_case, expected_args):
    """
    Pour chaque outil attendu avec des contraintes d'arguments, vérifie que
    l'appel correspondant contient les valeurs attendues (match partiel).
    Convention : '<field>_contains' -> inclusion, '<field>_exact' -> égalité.
    Retourne (nb_checks_ok, nb_checks_total).
    """
    ok = 0
    total = 0
    # Indexer les args réellement passés, par nom d'outil
    calls_by_tool = {}
    for s in steps_for_case:
        if s.get("tool_name") and s.get("tool_args"):
            calls_by_tool.setdefault(s["tool_name"], []).append(s["tool_args"])

    for tool_name, constraints in expected_args.items():
        actual_calls = calls_by_tool.get(tool_name, [])
        for constraint_key, expected_val in constraints.items():
            total += 1
            # Déterminer le champ et le mode
            if constraint_key.endswith("_contains"):
                field = constraint_key[:-len("_contains")]
                mode = "contains"
            elif constraint_key.endswith("_exact"):
                field = constraint_key[:-len("_exact")]
                mode = "exact"
            else:
                field = constraint_key
                mode = "contains"

            # Un des appels satisfait-il la contrainte ?
            satisfied = False
            for args in actual_calls:
                val = str(args.get(field, "")).lower()
                exp = str(expected_val).lower()
                if mode == "contains" and exp in val:
                    satisfied = True
                    break
                if mode == "exact" and exp == val:
                    satisfied = True
                    break
            if satisfied:
                ok += 1
    return ok, total


def score_end_to_end(case_meta, case_log, steps_for_case):
    """
    answerable=True  : succès si une réponse finale existe et n'est pas un échec de boucle.
    answerable=False : succès si le modèle n'a PAS produit une réponse inventée.
                       Heuristique : on regarde qu'il n'a pas bouclé en limite,
                       et on laisse un jugement manuel pour l'hallucination fine.
    Cette métrique est la plus imparfaite à automatiser : on la marque comme
    'needs_review' quand un jugement humain est requis.
    """
    answerable = case_meta.get("answerable", True)
    final = case_log.get("final_answer") or ""
    hit_limit = case_log.get("hit_iteration_limit", False)

    if hit_limit:
        return {"pass": False, "needs_review": False, "reason": "iteration_limit"}

    if answerable:
        # Succès basique : une réponse non vide a été produite.
        # La pertinence fine reste à vérifier à l'oeil (needs_review).
        return {"pass": bool(final.strip()), "needs_review": True, "reason": "answer_produced"}
    else:
        # Cas sans réponse possible : le bon comportement est de ne pas inventer.
        # On ne peut pas juger l'hallucination automatiquement de façon fiable.
        return {"pass": None, "needs_review": True, "reason": "unanswerable_manual_check"}


def main():
    # Choix du log
    if len(sys.argv) > 1:
        log_name = sys.argv[1]
    else:
        logs = sorted(glob.glob(str(LOG_DIR / "*.jsonl")))
        if not logs:
            print("Aucun log trouvé.")
            return
        log_name = Path(logs[-1]).name

    testset = load_testset()
    steps, cases = load_log(log_name)

    # Grouper les steps par cas
    steps_by_case = {}
    for s in steps:
        steps_by_case.setdefault(s["case_id"], []).append(s)

    backend = cases[0]["backend"] if cases else "ollama"
    price = PRICING.get(backend, PRICING["ollama"])

    print(f"\n{'='*70}")
    print(f"SCORING : {log_name}")
    print(f"{'='*70}\n")

    n = len(cases)
    tool_ok = 0
    args_ok_total = 0
    args_total = 0
    e2e_pass = 0
    e2e_review = 0
    total_latency = 0
    total_cost = 0.0

    header = f"{'case':<12}{'tool_sel':<10}{'args':<8}{'e2e':<14}{'latency':<10}"
    print(header)
    print("-" * len(header))

    for case_log in cases:
        cid = case_log["case_id"]
        meta = testset.get(cid, {})
        case_steps = steps_by_case.get(cid, [])

        # 1. Tool selection
        called = case_log["tools_called"]
        expected = meta.get("expected_tools", [])
        order_matters = meta.get("order_matters", False)
        sel_ok = score_tool_selection(called, expected, order_matters)
        if sel_ok:
            tool_ok += 1

        # 2. Arguments
        exp_args = meta.get("expected_args", {})
        a_ok, a_total = score_arguments(case_steps, exp_args)
        args_ok_total += a_ok
        args_total += a_total
        args_str = f"{a_ok}/{a_total}" if a_total else "-"

        # 3. End-to-end
        e2e = score_end_to_end(meta, case_log, case_steps)
        if e2e["pass"] is True:
            e2e_pass += 1
            e2e_str = "pass"
        elif e2e["pass"] is False:
            e2e_str = "FAIL"
        else:
            e2e_str = "manual"
        if e2e["needs_review"]:
            e2e_review += 1

        # 4. Latence + coût
        lat = case_log["total_latency_ms"]
        total_latency += lat
        # Coût : sommer les tokens des steps de ce cas
        case_tokens_p = sum(s.get("tokens_prompt", 0) for s in case_steps)
        case_tokens_c = sum(s.get("tokens_completion", 0) for s in case_steps)
        cost = (case_tokens_p / 1e6) * price["prompt"] + (case_tokens_c / 1e6) * price["completion"]
        total_cost += cost

        sel_str = "OK" if sel_ok else "x"
        print(f"{cid:<12}{sel_str:<10}{args_str:<8}{e2e_str:<14}{lat:<10}")

    print("-" * len(header))
    print(f"\n--- MÉTRIQUES AGRÉGÉES ---")
    print(f"Tool selection accuracy : {tool_ok}/{n} = {100*tool_ok/n:.0f}%")
    if args_total:
        print(f"Argument correctness    : {args_ok_total}/{args_total} = {100*args_ok_total/args_total:.0f}%")
    print(f"End-to-end (auto-pass)   : {e2e_pass}/{n} (dont {e2e_review} à revoir à la main)")
    print(f"Latence moyenne          : {total_latency/n:.0f} ms/cas")
    print(f"Coût total du run        : ${total_cost:.4f}")
    print(f"\nNote : end-to-end et cas 'unanswerable' demandent une vérification")
    print(f"manuelle des réponses finales (hallucination non jugeable automatiquement).")


if __name__ == "__main__":
    main()
