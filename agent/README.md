# TARS — Instrumented Agentic Layer

A hand-written tool-use agent built on top of TARS, designed to be **measured**, not just to work. The goal was less "make an agent" than "evaluate one rigorously": quantify tool-selection accuracy, end-to-end success, latency and cost across five model configurations, on a 50-case test set written before implementation and designed to be hard.

## What it does

The agent exposes three tools to the model and lets it decide when to call them, via Ollama's native function calling (and Anthropic's tool use for the cloud comparison):

- **search_knowledge** — RAG over personal documents (CV, portfolio, Obsidian notes), reusing the existing ChromaDB collection.
- **add_calendar_event** — writes to Google Calendar. Has a **dry-run mode** that validates the call without writing, so evaluation runs are replayable and never pollute the real calendar.
- **web_search** — live web search via Tavily, with a **per-query disk cache** so that two runs of the test set are comparable despite web non-determinism.

## The loop (no framework)

The agentic loop is written by hand in ~150 lines, not delegated to LangChain/LangGraph. On each turn: call the model with the current message history and the tool schemas; if it returns text, that's the final answer and we stop; if it requests a tool, we execute it, append the result to the history, and loop again. A hard cap of 5 iterations prevents the infinite loops small models fall into when a tool fails. Tool errors are returned to the model as the tool result, so it can recover or give up cleanly rather than crashing the loop.

Writing the loop by hand was a deliberate choice: on three tools, a framework saves half a day and costs the understanding — which is the point of the project — and it complicates the logging, which is the main deliverable.

## Measurement design

Two constraints were baked in from the start, because evaluating a system with side effects and non-determinism is otherwise impossible to replay:

- **Dry-run** for the calendar (side effect neutralized during eval).
- **Response cache** for web search (determinism restored).

Logging is JSONL, two levels (one record per loop turn, one summary per case), and is **purely factual** — it records what happened, never a verdict. Scoring is a separate script that compares the log to the ground truth. This means metrics can be recomputed without re-running the agent (which costs time and, for the cloud model, money).

A **model-abstraction layer** (`call_model`) normalizes Ollama and Anthropic into a single response object, so the same loop, test set and scoring run against any backend by changing one environment variable.

## Test set

50 cases with ground truth, spread across 8 categories and deliberately built to be hard: single-tool tasks, dependent two-tool chains, independent two-tool tasks, ambiguous questions (the right move is to ask for clarification, not to call a tool), no-tool-needed questions (calling a tool is a false positive), and unanswerable questions (the failure is hallucinating an answer). Knowledge cases are anchored on content actually indexed in the RAG, not invented.

## Results

Five configurations on the same 50 cases: the base model, the same model with a personality **system prompt**, two **LoRA fine-tunes** (v1 = 200 dialogues, v2 = 400 with anti-hallucination examples), and **Claude** (cloud ceiling).

| Metric | llama3.1:8b | tars (prompt) | tars-ft (v1) | tars-ft-v2 | Claude |
|---|---|---|---|---|---|
| Tool selection | 60% | 58% | 54% | 64% | **90%** |
| Argument correctness | 82% | 82% | 64% | 62% | **96%** |
| Hard failures | 7 | 8 | 10 | 11 | **4** |
| Mean latency (ms) | 6038 | 6423 | 7206 | **3520** | 5670 |
| Run cost | $0 | $0 | $0 | $0 | $0.62 |

Tool selection by category (correct / total):

| Category | base | prompt | ft-v1 | ft-v2 | Claude |
|---|---|---|---|---|---|
| single_tool_knowledge | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| single_tool_calendar | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 |
| single_tool_websearch | 9/9 | 8/9 | 9/9 | 8/9 | 7/9 |
| two_tools_dependent | 1/5 | 1/5 | 1/5 | 1/5 | **4/5** |
| two_tools_independent | 3/6 | 4/6 | 0/6 | 0/6 | **5/6** |
| ambiguous | 2/5 | 2/5 | 2/5 | 3/5 | **5/5** |
| no_tool_needed | 1/6 | 1/6 | 1/6 | 3/6 | **6/6** |
| unanswerable | 0/5 | 0/5 | 0/5 | 3/5 | **4/5** |

## Three findings

**1. Local matches cloud on simple tasks; the gap is in reasoning.** On single-tool tasks the local models are already at ceiling (knowledge 8/8 everywhere). The cloud advantage is concentrated in multi-step chaining (dependent chains: 1/5 local vs 4/5 Claude) and abstention judgment (knowing when *not* to act). For a personal assistant, this argues for local-first with cloud escalation reserved for complex tasks.

**2. Personality via prompt preserves agentic ability; via fine-tuning it doesn't.** The prompt-based model keeps 82% argument correctness (identical to the base) and still chains independent tools (4/6). Both fine-tunes collapse to 62–64% arguments and **0/6** on independent chains. Fine-tuning a strong personality onto a small model measurably damages its tool-use — a counter-intuitive result, since "fine-tuned" is often assumed to beat "prompted".

**3. The two fine-tunes fail differently.** v1 (200 dialogues, no abstention examples) over-uses tools and loops until the iteration cap. v2 (400 dialogues, 45% abstention examples) over-corrects into passivity: it abstains well (best local model on unanswerable, 3/5) but stops doing the multi-step tasks at all. The dataset doesn't just make the model "better" or "worse" — it changes *how* it fails.

## A note on ground-truth limits

The metric penalizes Claude on two cases (capital of Australia, CEO of Anthropic) where it answered from memory instead of calling web_search — but its answers were correct and the tool was unnecessary. This is a limitation of a rigid ground truth, not a Claude failure, and it's why qualitative review still matters alongside the numbers.

## Files

- `tools.py` — the three tools, their JSON schemas, dry-run and cache.
- `model.py` — model abstraction normalizing Ollama and Anthropic.
- `loop.py` — the agentic loop.
- `logger.py` — two-level JSONL logging (factual, no verdict).
- `run_eval.py` — runs the whole test set on one model.
- `score.py` — scores one run against the ground truth.
- `compare.py` — aggregates several runs into the comparison tables above.
- `testset.json` — the 50 cases with ground truth.

## Running it

```bash
# One model over the full test set (dry-run calendar, cached web)
AGENT_MODEL=llama3.1:8b python3 run_eval.py
AGENT_MODEL=tars-ft-v2 python3 run_eval.py
AGENT_BACKEND=anthropic AGENT_MODEL=claude-sonnet-4-5 python3 run_eval.py

# Score one run, then compare all of them
python3 score.py
python3 compare.py
```
