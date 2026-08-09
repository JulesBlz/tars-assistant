# TARS — technical decisions

This document lists the main technical decisions made during TARS development, with for each one: context, chosen option, rejected alternatives, and accepted trade-off.

The goal is to make the reasoning behind choices readable, rather than just describing the result.

---

## 1. Local model: Llama 3.1 8B

**Context.** TARS core is an LLM running on a Mac M5 24 GB. The choice determines viable personality, latency, and available context size.

**Decision.** Llama 3.1 8B in Q4_K_M quantization, served by Ollama.

**Rejected alternatives.**

- *Mistral Nemo 12B*: tested in real conditions. Doesn't respect personality system prompts and consistently falls back to generic assistant mode.
- *Qwen 2.5 14B*: noticeable latency on Apple Silicon, less mature ecosystem.
- *Llama 3.3 70B*: out of memory range.

**Trade-off.** Llama 8B is limited on complex reasoning and hallucinates on precise facts. These limits are addressed by cloud fallback, never by additional prompt engineering.

---

## 2. Runtime: Ollama

**Context.** The runtime determines inference speed, developer experience, and available API.

**Decision.** Ollama.

**Rejected alternatives.**

- *Direct llama.cpp*: more performant in practice on Apple Silicon, but marginal gain at this project scale, and heavier setup.
- *MLX*: fastest on Apple Silicon but younger ecosystem.

**Trade-off.** An additional abstraction layer costing a few tokens/second, in exchange for a standard HTTP API, name-based model management, and a declarative Modelfile for configuration.

**Architecture point.** Runtime is decoupled from orchestration: FastAPI talks to Ollama via standard HTTP. Switching runtime would only impact a few lines. This loose coupling is applied throughout the project.

---

## 3. Personality: Modelfile with few-shot examples

**Context.** TARS needs a strong personality that goes against Llama's RLHF tuning, designed to be polite and helpful.

**Decision.** System prompt in the Modelfile, with dialogue examples (few-shot) rather than abstract description.

**Rejected alternatives.**

- *LoRA fine-tuning*: structurally more stable solution, but the investment (dataset, training) isn't justified as long as the prompt holds.
- *Dynamically injected prompt on each request*: more flexible but doubles token consumption and complicates orchestration.

**Trade-off.** Prompt engineering is fragile: over many turns, the model regresses to its training habits. Fine-tuning would be more stable but remains out of scope.

**Observation.** Few-shot examples have significantly higher impact than abstract descriptions. The model imitates more than it obeys.

---

## 4. RAG: ChromaDB + multilingual embeddings

**Context.** TARS draws on personal documents in its responses.

**Decision.** Local ChromaDB, `paraphrase-multilingual-MiniLM-L12-v2` embeddings, 400-word chunks with 50-word overlap, top-6 retrieval with distance threshold at 1.2.

**Rejected alternatives.**

- *FAISS*: faster on large corpora, but no native persistence. Overkill for this volume.
- *Managed vector databases (Pinecone, Weaviate)*: incompatible with the "100% local" principle.
- *English embeddings*: inadequate for a predominantly French corpus.

**Trade-off.** A heavier embedding model (bge-m3, e5-mistral) would improve retrieval at the cost of longer startup and slower ingestion.

**Learning.** Semantic similarity favors prose documents over structured ones. A bulleted CV consistently gets overshadowed by a narrative portfolio at retrieval. The solution goes through the corpus, not the threshold.

---

## 5. Cloud fallback: consent-first

**Context.** Some tasks exceed local model capabilities. Industry standard is automatic server-side routing, invisible to the user.

**Decision.** Two explicit modes: (a) auto-proposal by the local model when it detects a limit; (b) manual escalation via button. No external call without user confirmation.

**Rejected alternatives.**

- *Automatic routing by heuristic*: opaque to the user, incompatible with local-first positioning.
- *Native tool calling*: Llama 8B is unreliable on function calling. The chosen pattern (model proposes, user validates, server intercepts) is more robust.

**Trade-off.** An additional user action where an automatic system would be smoother. Assumed: confidentiality prevails over fluidity.

**Positioning.** TARS isn't an assistant that silently calls the cloud, it's a local assistant that asks permission to go elsewhere when needed.

---

## 6. Cloud: Anthropic API

**Context.** The fallback must be reliable and affordable for individual use.

**Decision.** Anthropic API, Sonnet model.

**Rejected alternatives.**

- *OpenAI GPT-4*: industry standard, rejected for writing quality preference.
- *Groq*: free and fast, rejected on platform considerations.
- *Gemini*: free tier unavailable on my account.

**Trade-off.** Paid vs free alternatives. Negligible at individual usage scale (5€ covers several months).

---

## 7. Context transmitted to cloud: conversation history only

**Context.** When Claude is invoked, we can transmit more or less context: nothing, current conversation history, or also personality portrait and RAG results.

**Decision.** Only current conversation history. No personal documents, no profile, no RAG results are transmitted to the external API.

**Rejected alternatives.**

- *Send portrait + RAG to Claude*: would improve response quality, but violate the "personal data doesn't leave the Mac" promise.
- *Send a compressed profile summary*: intermediate compromise, but introduces a fuzzy decision about what's safe to transmit.

**Trade-off.** Claude has less context and sometimes responds more generically on personal topics. This is the price of architectural consistency.

---

## 8. Long-term memory: selective ingestion

**Context.** Persistent memory across sessions is needed for TARS to retrieve old exchanges. Storing everything would be naive.

**Decision.** Dedicated ChromaDB collection, with two strict rules: (a) user messages over 30 words only; (b) cloud responses only, never local model responses.

**Rejected alternatives.**

- *Ingest everything*: significant noise, and especially risk of self-reinforcing local model hallucinations.
- *Summarize conversations at each session's end with the cloud model*: would violate the local-first principle.

**Trade-off.** The model doesn't retain memory of its own local responses. In practice, the original question is enough to resume context.

**Underlying reasoning.** An AI that learns from its own hallucinations drifts. Not ingesting its own responses is a robustness choice, not a completeness one.

---

*Document maintained as structural decisions are made.*