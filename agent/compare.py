"""
Analyse comparative de plusieurs runs d'évaluation.

Lit tous les logs d'un dossier (ou une liste donnée), les score chacun contre
le ground truth, et produit :
  1. Un tableau comparatif des métriques agrégées (un modèle par colonne).
  2. Une décomposition de la sélection d'outil PAR CATÉGORIE.
  3. Le comptage des échecs francs et des boucles par modèle.

Définition unique d'un "échec franc" : cas answerable=true nécessitant au moins
un outil (expected_tools non vide) où la sélection d'outil a échoué. Autrement
dit, le modèle devait agir et n'a pas accompli la tâche. Les cas no_tool /
ambiguous / unanswerable en sont exclus (ne rien faire peut y être correct).

Les logs sans record 'case' (mini-tests) sont ignorés automatiquement.

Usage :
    python3 compare.py                  # compare tous les logs valides du dossier
    python3 compare.py log1 log2 ...    # compare des logs precis
"""
import sys, os
sys.path.insert(0, os.path.expanduser("~/tars"))

import glob
from pathlib import Path
from collections import defaultdict

from score import (
    load_testset, load_log,
    score_tool_selection, score_arguments,
    PRICING,
)

LOG_DIR = Path(__file__).parent / "logs"


def score_run(log_name, testset):
    """Score un run. Renvoie None si le log ne contient aucun cas."""
    steps, cases = load_log(log_name)
    if not cases:
        return None

    steps_by_case = defaultdict(list)
    for s in steps:
        steps_by_case[s["case_id"]].append(s)

    backend = cases[0]["backend"]
    model = cases[0]["model"]
    price = PRICING.get(backend, PRICING["ollama"])

    n = len(cases)
    tool_ok = 0
    args_ok = 0
    args_total = 0
    fails = 0
    loops = 0
    total_latency = 0
    total_cost = 0.0
    by_cat = defaultdict(lambda: [0, 0])

    for case_log in cases:
        cid = case_log["case_id"]
        meta = testset.get(cid, {})
        case_steps = steps_by_case.get(cid, [])
        cat = meta.get("category", "?")

        called = case_log["tools_called"]
        expected = meta.get("expected_tools", [])
        order_matters = meta.get("order_matters", False)
        sel_ok = score_tool_selection(called, expected, order_matters)

        by_cat[cat][1] += 1
        if sel_ok:
            tool_ok += 1
            by_cat[cat][0] += 1

        a_ok, a_tot = score_arguments(case_steps, meta.get("expected_args", {}))
        args_ok += a_ok
        args_total += a_tot

        if case_log.get("hit_iteration_limit"):
            loops += 1

        requires_tool = len(expected) > 0
        if meta.get("answerable", True) and requires_tool and not sel_ok:
            fails += 1

        total_latency += case_log["total_latency_ms"]
        tp = sum(s.get("tokens_prompt", 0) for s in case_steps)
        tc = sum(s.get("tokens_completion", 0) for s in case_steps)
        total_cost += (tp / 1e6) * price["prompt"] + (tc / 1e6) * price["completion"]

    return {
        "model": model,
        "backend": backend,
        "n": n,
        "tool_selection_pct": 100 * tool_ok / n,
        "args_pct": 100 * args_ok / args_total if args_total else 0,
        "fails": fails,
        "loops": loops,
        "latency_ms": total_latency / n,
        "cost": total_cost,
        "by_cat": dict(by_cat),
    }


def main():
    testset = load_testset()

    if len(sys.argv) > 1:
        log_names = [Path(a).name for a in sys.argv[1:]]
    else:
        log_names = sorted(Path(p).name for p in glob.glob(str(LOG_DIR / "*.jsonl")))

    if not log_names:
        print("Aucun log a comparer.")
        return

    results = []
    skipped = []
    for name in log_names:
        r = score_run(name, testset)
        if r is None:
            skipped.append(name)
        else:
            results.append(r)

    if skipped:
        print(f"Ignores (aucun cas) : {len(skipped)} log(s)")
    if not results:
        print("Aucun log valide a comparer.")
        return

    order = {"llama3.1:8b": 0, "tars:latest": 1, "tars-ft:latest": 2,
             "tars-ft-v2:latest": 3, "claude-sonnet-4-5": 4}
    results.sort(key=lambda r: order.get(r["model"], 99))

    def short(r):
        return r["model"].replace("claude-sonnet-4-5", "claude").replace(":latest", "")[:14]

    labels = [short(r) for r in results]
    col = 14

    def row(title, values):
        print(f"{title:<24}" + "".join(f"{v:<{col}}" for v in values))

    width = 24 + col * len(results)
    print("\n" + "=" * width)
    print("COMPARAISON DES MODELES - metriques agregees")
    print("=" * width + "\n")
    row("Modele", labels)
    print("-" * width)
    row("Tool selection", [f"{r['tool_selection_pct']:.0f}%" for r in results])
    row("Argument correct.", [f"{r['args_pct']:.0f}%" for r in results])
    row("Echecs francs", [str(r["fails"]) for r in results])
    row("Boucles (limite)", [str(r["loops"]) for r in results])
    row("Latence moy. (ms)", [f"{r['latency_ms']:.0f}" for r in results])
    row("Cout du run ($)", [f"{r['cost']:.4f}" for r in results])

    print("\n" + "=" * width)
    print("SELECTION D'OUTIL PAR CATEGORIE (reussis / total)")
    print("=" * width + "\n")
    all_cats = sorted({c for r in results for c in r["by_cat"]})
    row("Categorie", labels)
    print("-" * width)
    for cat in all_cats:
        vals = []
        for r in results:
            ok, tot = r["by_cat"].get(cat, [0, 0])
            vals.append(f"{ok}/{tot}" if tot else "-")
        row(cat[:22], vals)

    print("\n" + "=" * width)
    print("Echec franc = cas answerable necessitant un outil ou la selection a")
    print("echoue (le modele devait agir, n'a pas accompli la tache).")
    print("Categories no_tool/ambiguous/unanswerable : succes si aucun outil appele.")
    print("=" * width)


if __name__ == "__main__":
    main()
