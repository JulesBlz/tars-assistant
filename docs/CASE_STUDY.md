# TARS — Technical case study

*A local AI assistant, its fine-tuning and its agentic evaluation.*

## 1. Scope and constraints

TARS is a personal AI assistant running locally on an Apple Silicon laptop (M5, 24 GB unified memory). It comprises three layered systems built over four months: a conversational assistant (local inference, retrieval over personal documents, long-term memory, voice, read access to mail and calendar, consent-gated cloud escalation), a fine-tuned personality model produced through a complete LoRA pipeline across two dataset iterations, and an instrumented agent with a fifty-case evaluation harness comparing five model configurations.

Two constraints shaped every decision. **Local-first**: anything that can run on the machine does, and personal data leaves it only by explicit user action. This is a hard constraint, not a preference, and it eliminated several otherwise superior options (cloud text-to-speech, cloud summarization for memory). **Measure rather than assume**: consequential choices eventually got numbers attached, which is how several of them were found to be wrong.

A third constraint is worth stating because it explains the shape of the codebase: the author had written Python in preparatory classes and C++ at engineering school but had never built a substantial software project. The system was therefore an exercise in software engineering as much as in machine learning, and the phased build followed one rule: **every phase had to be demonstrable on its own**, so that stopping at any point left something that worked.

The headline result, established by measurement rather than assumed: **giving a small model a personality through a system prompt preserves its agentic capabilities, while giving it the same personality through fine-tuning degrades them.** Argument correctness holds at 82% under prompting, identical to the base model, and falls to 62% under fine-tuning; independent tool chaining goes from 4/6 to 0/6. Sections 8 and 10 establish this and section 11 examines what it implies.

This document records what was built, the reasoning behind each decision, the decisions that proved wrong, and the measurements that settled them. It assumes no prior context and is written to remain useful after the project stops moving.

---

## 2. Architecture overview

The system is a FastAPI server with a minimal browser client. The client-server split, rather than a single script, keeps the interface replaceable and makes the state boundaries explicit.

The request path for one conversational turn: the browser posts a message; the server retrieves relevant passages from two vector collections (personal documents, past conversations); it assembles a prompt from the system prompt, the retrieved context and recent history; it calls the local model through Ollama; it persists both sides to SQLite; it selectively ingests the exchange into long-term memory; it returns the reply, which the client can have spoken locally.

The components and their actual responsibilities:

- **Ollama** serves the model over HTTP, wrapping llama.cpp. It handles model storage, chat templating and native function calling.
- **ChromaDB** holds two separate collections: `jules_knowledge` for documents, `conversations` for long-term memory. The separation is deliberate and is argued in section 5.
- **SQLite** holds the current conversation. It is the transcript, not the memory system.
- **Whisper (small)** transcribes locally; **Piper** synthesizes locally.
- **Google APIs** provide read access to recent mail and upcoming events, plus write access to calendar events for the agent.
- **The Anthropic API** is the escalation path, invoked only on explicit decision.

The agent (section 9) reuses this stack but inverts one relationship: retrieval stops being injected on every turn and becomes a tool the model chooses to call.
---

## 3. Running a model locally

### 3.1 Model, runtime, quantization

Three things are routinely conflated, and separating them explains most of what follows.

The **model** is a set of weights. Llama 3.1 8B is roughly eight billion parameters, real numbers arranged in matrices, produced by Meta's training run. On its own it is an inert file.

The **runtime** executes inference: it loads the weights, takes tokenized text, performs the matrix multiplications layer by layer, and produces a probability distribution over the next token. Here the runtime is **llama.cpp**, a C++ inference engine optimized for CPU and Apple Silicon through Metal. **Ollama** wraps it in an HTTP server with model management, chat templating and function calling.

**Quantization** compresses the weights. The original model stores each weight in 16 bits. Q4_K_M stores them in roughly 4 bits on average, using a block-wise scheme that preserves more precision for sensitive tensors (the K_M suffix denotes a mixed-precision variant). This is why the same model occupies 16 GB in f16 and 4.9 GB in Q4_K_M. The cost is a slight quality degradation, generally imperceptible in conversation and more visible on precision tasks such as arithmetic or code. On a 24 GB machine that also has to run an operating system, a browser and a Python server, the trade is not close: without quantization the model leaves no room for anything else.

**GGUF** is llama.cpp's container format, holding quantized weights, tokenizer and metadata in one file. When Ollama imports a model through a Modelfile, it copies the GGUF into its own store and attaches the system prompt and inference parameters (temperature, top_p, context size).

### 3.2 What an inference call actually does

A chat call to Ollama concatenates the system prompt, the conversation history and the new message into Llama 3.1's chat template (the `<|start_header_id|>` markup), tokenizes it, runs an autoregressive forward pass generating one token at a time, samples according to temperature and top_p, and detokenizes.

Observed latency in this system, two to six seconds per reply, decomposes into two parts: prefill, proportional to the size of the injected context, and generation, proportional to the number of tokens produced. This has a design consequence that is easy to miss. Every additional element injected into the prompt (retrieved passages, memory, personality description) costs latency on every single turn. A retrieval system that returns six passages when two would do is not just noisy, it is slow.

One structural fact discovered during the agent work deserves emphasis: **tool calling in Ollama is implemented through the chat template, not through model behaviour**. The template inserts tool schemas and parses structured calls. This is why fine-tuning the personality did not break tool *selection*, which lives below the weights in the templating layer, while it did degrade the model's ability to *read a tool result and use it*, which lives in the weights.

### 3.3 Where an 8B model breaks

A quantized eight-billion-parameter model is capable in conversation and on well-scoped tasks. The evaluation in section 10 localized precisely where it stops being capable, and the pattern is consistent: multi-step chaining (it performs the first action and forgets the rest), abstention judgment (recognizing that no action is warranted), and restraint under ambiguity (it acts on a guess rather than asking). These are not bugs to be fixed by prompting; they are the reasoning limits of the size class.

Instruction-following also degrades as context accumulates. A five-hundred-word system prompt is respected in the opening turns and progressively less thereafter. This observation is one of the arguments that made fine-tuning attractive in the first place: a personality encoded in weights does not dilute the way a personality encoded in a prompt does. Whether that benefit was worth its cost is the subject of sections 8 and 10.

---

## 4. Retrieval over personal documents

### 4.1 Why retrieval rather than context

The model knows nothing about its user, and the context window is far too small to hold a life. Retrieval-augmented generation solves this by indexing documents as vectors, finding the relevant passages at query time, and injecting only those.

The corpus here is a CV, an architecture portfolio, and an Obsidian note vault linked into the project directory by a symlink. Documents are split into chunks of roughly 300 to 500 words with a 50-word overlap between consecutive chunks; the overlap exists so that a passage split across a boundary is not lost to both sides.

Each chunk becomes an **embedding**: a 384-dimension vector that encodes meaning, such that semantically similar texts have geometrically close vectors. Proximity is measured by cosine distance, the angle between vectors. **ChromaDB** stores the vectors and performs k-nearest-neighbour search.

The embedding model choice was between `all-MiniLM-L6-v2` (lighter, faster, primarily English) and `paraphrase-multilingual-MiniLM-L12-v2` (multilingual, better French, slightly heavier). The corpus is largely French, so the multilingual model won despite the cost. This is a small decision with a large downstream effect: a retrieval system is only as good as its embedding model's grasp of the corpus language.

### 4.2 The bug that produced the threshold

The first retrieval implementation had a flaw that only appeared under a test unrelated to the corpus. Asked a question about multi-head attention, a topic with no connection whatsoever to an architecture portfolio, the system still injected portfolio chunks into the prompt.

The cause: **k-nearest-neighbour search always returns k results.** There is no notion of "nothing matched". The nearest chunk to a question about attention mechanisms is still *some* chunk, and it gets injected regardless of how far away it is. The result is prompt pollution that can actively degrade the answer, since the model now has to reconcile irrelevant context with the question.

The fix was to filter by distance: ChromaDB returns a distance score per result, and only chunks below a threshold are injected. If nothing passes the threshold, nothing is injected at all and the model answers from its own knowledge.

```python
def retrieve_context(query, k=TOP_K, distance_threshold=1.2):
    ...
    filtered = [(c, s, d) for c, s, d in zip(chunks, sources, distances)
                if d < distance_threshold]
    if not filtered:
        return ""
```

The threshold value of 1.2 is empirical. It was set by observing the distance distribution of results that were obviously relevant against results that were obviously not, and placing the cut between them. Lower is stricter. There is no universal value: it depends on the embedding model, the corpus and the query style, which is exactly why it has to be tuned by observation rather than copied from a tutorial. The system logs how many chunks passed and from which sources on every query, which is what made the tuning possible at all.

### 4.3 What the corpus taught

**Semantic similarity favours prose over structure.** The portfolio, written in descriptive paragraphs, retrieves well. The CV, written in lists of dates and keywords, retrieves poorly. An embedding encodes diffuse meaning, and a bulleted list of software names carries little of it. The general lesson is that the hard part of retrieval is not the pipeline, which is a day's work, but the shape and quality of the corpus, which is a standing problem.

**Acronyms defeat dense retrieval.** The query "what is my PFE about" returned nothing, although the portfolio contains "Master's Thesis (PFE) / PROTOLAB". An isolated acronym carries almost no semantic signal for a multilingual embedding model. Rephrased with the project name, retrieval works. This is a textbook limitation of pure dense retrieval, and the standard industry answer is **hybrid search**: combining dense vector search with lexical search (BM25), so that exact tokens, proper nouns, identifiers and acronyms are caught by the lexical side while meaning is caught by the dense side. TARS has no hybrid search. It is a known and unaddressed limitation, listed in section 12.

**Ingestion strategy is a real design decision.** Four options were weighed. Re-ingesting at server startup is simple and reliable but stale for anything written while the server runs. Re-ingesting when the conversation is cleared gives the user control at an arbitrary moment. A filesystem watcher gives real-time freshness at the cost of a background process and more moving parts. Incremental ingestion at startup, tracking file modification times so that only changed files are re-embedded, was chosen. The decisive argument was scaling: the original implementation deleted and rebuilt the entire collection on every run, which is instantaneous on three files and unacceptable on five hundred. The accepted cost is that a note written while TARS is running is only indexed at the next restart.
---

## 5. Memory: what a system should refuse to remember

Long-term memory is a second retrieval system over past conversations, kept in a separate ChromaDB collection. Its design is the clearest example in the project of a safety decision taken before any harm occurred.

The naive version ingests everything: every message, both sides, indefinitely. Two objections were raised against it, and both came from thinking about lived use rather than from a specification.

The first objection was scope: an early proposal used the cloud model to summarize conversations before storing them, which quietly violated the local-first constraint. Summarization either stays local, using the small model, or does not happen.

The second objection is the interesting one. **If the local model hallucinates a fact and that fact is ingested into long-term memory, it becomes retrievable ground truth.** Three months later the system retrieves its own invention and presents it as recollection. The error compounds instead of decaying, and there is no mechanism anywhere in the pipeline that would catch it.

The rule adopted in response is narrow and effective:

- **User messages are ingested if they exceed 30 words.** They are true by construction, since the user knows what they wrote. The length filter drops greetings, acknowledgements and one-word replies that carry no retrievable content.
- **Local model replies are never ingested.** They may be hallucinated, and nothing downstream would ever verify them.
- **Cloud model replies are ingested**, since they are both higher quality and explicitly consented to.

The system therefore remembers what the user said, and never remembers what it said itself. It can retrieve "the time you described your interview at that company" but not "the time it invented something about that company".

This is a small piece of code with a disproportionate design payload. It amounts to treating generated content as untrusted by default and distinguishing it from verified content at the storage boundary rather than at the point of use, which is the same instinct that separates user-generated from editorial content in recommender systems. It is also an early instance of the pattern that runs through the whole project: the orchestrator, not the model, is responsible for the system's invariants.

---

## 6. Voice and external integrations

### 6.1 The voice loop

Speech-to-text is **Whisper small**, running locally. Text-to-speech is **Piper** with the French voice `fr_FR-gilles-low`, also local. Both choices follow from local-first rather than from quality, and the quality gap is real: Piper is noticeably more synthetic than a cloud service such as ElevenLabs.

That gap was tested rather than assumed. Three options were compared when the voice felt too slow: a different Piper voice (`fr_FR-tom-medium`, more expressive), the macOS built-in `say` voices, and a cloud voice. The cloud option was rejected because it would send every spoken sentence off the machine, which is precisely what the project exists not to do, and because a per-request cost for a daily-use assistant is a poor trade. The macOS voices sound more natural but less robotic, which works against a personality modeled on a machine. Piper was kept.

A detail worth recording, because it is invisible until it bites: the cloud escalation path uses a *different* voice (a macOS `say` voice) from the local assistant. The voice is the signal that the answer came from elsewhere. An interface that hides which model answered is an interface that makes the consent design meaningless.

### 6.2 Google integrations and OAuth scopes

Gmail and Calendar access run through OAuth 2.0. The mechanism worth understanding is **scopes**: granular permissions attached to the issued token, requested at consent time.

TARS initially requested `gmail.readonly` and `calendar.readonly`, following the principle of least privilege: a system that only needs to read should not hold a token that can write. When the agent later needed to create events, `calendar.events` (read and write) was added.

Adding a scope is not a configuration change. The existing token does not carry the new permission, so the token must be regenerated and the user must re-consent through the browser flow. This is the mechanism working correctly rather than an inconvenience: a permission the user never granted cannot be silently acquired by an application that decides it needs one.

Retrieved mail and events feed the startup briefing, which is the assistant's first output on opening the interface. The briefing went through several iterations, and its final specification is more prescriptive than one would expect: a stable opening line, an inbox summary distinguishing useful mail from noise, the next appointment with its exact time as written in the calendar, an optional point of vigilance, and a closing question that invites the conversation to continue. The prescriptiveness exists because free-form briefings drifted: the model would invent times, restructure the ordering unpredictably, or end flatly in a way that made the interface feel like a dead end. The instruction that mattered most was the one forbidding invention of any appointment or time not present in the injected data.

---

## 7. Cloud escalation: consent-first design

### 7.1 The idea and its first implementation

The escalation design originated as a user requirement rather than a technical proposal: the local model should recognize when a task exceeds it and *ask permission* before sending anything to a more capable cloud model. That framing has two properties worth naming. It keeps the user in control of what leaves the machine, which is the privacy constraint expressed as an interaction rather than a policy. And it makes the model's uncertainty visible instead of hiding it behind a confident answer.

The first implementation was straightforward and, as it turned out, fragile. The system prompt instructed the model to offer escalation when unsure, using a phrase containing "Claude" and a question mark. A Python function scanned each reply for that pattern plus one of several keywords ("transmit", "pass to", "escalate"). On a match, the original question was held pending, and a confirmation in the next turn triggered the API call.

### 7.2 Why it broke, and what that taught

Fine-tuning broke it completely. The fine-tuned model no longer produced the expected formula, so the detector never fired. Worse, the model would *talk about* Claude while refusing to escalate, inventing reasons why the cloud model could not help. From the user's perspective the feature had silently disappeared.

The diagnosis generalizes well beyond this system. **A critical path should not depend on a model producing an exact phrase.** Natural-language detection over model output is a brittle contract: it can be broken by fine-tuning, by a prompt change, by a model upgrade, or by the model simply phrasing things differently on a given day.

### 7.3 The redesign

The rebuilt architecture separates the *decision* to escalate from its *execution*. One function performs the escalation; three independent triggers can call it.

**The user's explicit request**, detected in the user's own message ("ask claude", "escalate", and variants). This is entirely code-side and therefore immune to anything the model does. An important refinement: the question forwarded to the cloud is the user's last real question, not the trigger phrase itself, since sending "ask claude" to Claude would be useless.

**The model's self-signal**, through a structured token (`[ESCALADE]`) emitted at the end of a reply. A single unique token is a far more robust contract than a natural-language phrase, but it still depends on the model following an instruction, and the fine-tuned model follows it inconsistently. This path is documented as unreliable. The proper fix is training examples containing the token, which belongs to a future dataset iteration.

**A dedicated interface button**, a separate route that bypasses the model entirely.

The privacy boundary is explicit in the implementation: the escalation call transmits recent conversational history only. The personality portrait and retrieved personal documents are never sent. This is why the cloud model in the evaluation performs without any of the local context the local models had, which is a deliberate asymmetry rather than an oversight.

The generalizable principle: **decide in code whatever can be decided in code; leave to the model only what only the model can judge, and accept that this part will be less reliable.**
---

## 8. Fine-tuning: theory, pipeline, and two datasets

### 8.1 What fine-tuning actually does

Fine-tuning resumes training of a pre-trained model on a new corpus, adjusting weights by gradient descent to minimize a loss function. Here the loss is standard language-modeling loss: predict the next token of the target replies. The model learns to make outputs resembling the examples more probable.

One property of this process explains most of what went wrong later. **The model learns everything the dataset contains implicitly, not only what the author intended to teach.** A dataset written to teach a terse tone also teaches, silently, "always answer with confidence" if not a single example shows the model declining to know something. Nothing marks the difference between the intended lesson and the accidental one; both are just statistical regularities in the training distribution.

### 8.2 LoRA, and why an 8B model fits on a free GPU

Full fine-tuning of an 8B model updates eight billion parameters, which does not fit in the 16 GB of a free-tier T4 GPU once optimizer states and activations are accounted for.

**LoRA (Low-Rank Adaptation)** avoids this with a linear-algebra argument. Instead of modifying a weight matrix W (say 4096 x 4096), it freezes W and learns a correction expressed as a product of two thin matrices: ΔW = B·A, where A is r x 4096 and B is 4096 x r, with r small. The correction is therefore a **rank-r matrix**: it can only express low-dimensional adjustments to W. The hypothesis, which holds well empirically, is that task adaptation lives in a low-dimensional subspace and does not need the full expressive capacity of the original matrix.

The arithmetic is the point. For that single matrix, full fine-tuning learns 16.7 million parameters; LoRA at r=16 learns 2 x 16 x 4096 = 131,072, under one percent. Across the whole model, the trained adapter for TARS is 160 MB against 16 GB for the model itself.

The hyperparameters used, and what each controls:

- **r = 16**, the rank, setting how expressive the correction can be. Common values run from 8 to 64. Higher r means more capacity and more risk of overfitting on a small dataset.
- **alpha = 32**, a scaling factor applied to the correction before it is added. The convention alpha = 2r is widespread.
- **dropout = 0.05**, regularization.
- **learning rate 2e-4**, high compared to full fine-tuning, which is normal for LoRA since only a small, freshly initialized set of parameters is being trained.
- **3 epochs** over 400 examples.
- **Target modules**: the attention projections (q, k, v, o) and the MLP projections (gate, up, down).

Training in 4-bit with adapters in 16-bit is **QLoRA**, which is what the Unsloth library performs underneath.

### 8.3 The pipeline, as actually lived

The chain: generate a synthetic dataset through the Anthropic API; train the LoRA adapter on Colab with Unsloth; save the adapter; merge it into the base model (PEFT's `merge_and_unload`, which adds ΔW into W and produces a standard model); convert to GGUF f16 with llama.cpp's conversion script; quantize to Q4_K_M; import into Ollama through a Modelfile.

Steps four through seven ended up running **entirely locally on Apple Silicon** after Colab's free tier ran out of GPU quota mid-project and its pre-installed dependency stack produced a cascade of version conflicts. That forced migration turned out to be valuable: it required understanding each step rather than calling a single Unsloth convenience function that performed merge, conversion and quantization in one opaque call.

Three practical lessons came out of the friction, and they generalize to any similar pipeline.

**Save the expensive artifact first.** The LoRA adapter is the only output that costs GPU time to reproduce. Everything downstream (merge, convert, quantize) is minutes of local CPU. Writing the adapter to persistent storage immediately after training, before attempting anything else, converts a lost session from a disaster into an inconvenience.

**Do not write large files to network-mounted storage from an ephemeral runtime.** Merging directly to Google Drive produced silently corrupted output, including safetensors files of zero bytes. Work on local disk, verify integrity, then copy the small final artifact.

**Expect tokenizer friction in GGUF conversion.** Saving a merged Llama 3.1 model writes a `tokenizer_config.json` referencing a tokenizer class the GGUF converter does not recognize, and conversion fails after successfully processing all 292 weight tensors. The fix is to copy the original Llama 3.1 tokenizer files into the merged directory before converting. This class of problem, where the weights are fine and the metadata is not, is characteristic of the open-source tooling ecosystem and is worth budgeting time for.

### 8.4 Dataset v1: teaching a model to hallucinate

The first dataset was 200 dialogues generated by a large model, prompted to write exchanges in the TARS style: terse, direct, occasionally ironic, never obsequious.

The resulting model had excellent style and **hallucinated with total confidence**. Asked about the user's thesis, it invented a subject. Asked what a specific project was, it fabricated a plausible description of a company that does not exist. The inventions were stylistically perfect, which made them worse: a confident wrong answer in the right voice is harder to catch than a hedged one.

Reading the dataset back explains it. A model asked to generate example dialogues fills the replies with plausible content, because something has to be written. Asked to produce an exchange about the time, it writes a specific time. Asked about a project, it writes a description. The dataset contained **no example of abstention whatsoever**. Its dominant statistical regularity was therefore: whatever the question, produce a specific and confident answer. Three epochs of gradient descent taught exactly that.

The generalizable statement: *the quality of a fine-tuning dataset is not measured by what its examples assert, but by what they implicitly teach. A dataset that never contains "I don't know" teaches a model never to say it.*

### 8.5 Dataset v2: fixing one failure, creating another

The second dataset was built against that specific defect. 400 dialogues, composed deliberately:

- 35% pure style and personality (reactions, opinions, refusals) where no external fact is required and there is therefore nothing to invent
- 20% **abstention**: the user asks for a fact the model cannot know, and the model says so
- 15% **grounded context use**: the message contains an explicit fact (an email extract, a calendar entry) and the reply relies on it and nothing else
- 10% **briefings**, which are grounded context use in a specific format
- 10% stable technical concepts
- 10% personal reactions drawn from the portrait without inventing specifics

Forty-five percent of the dataset teaches factual grounding, against zero percent in v1.

The generator prompt was rewritten with an explicit anti-invention rule reproduced in every batch: no invented times, dates, project names, statistics or personal facts; if a fact is not derivable from stable knowledge or from context provided in the message, the assistant abstains.

Verification afterwards was quantitative rather than impressionistic. An automated pass flagged replies containing precise times, dates or percentages not present in the corresponding prompt. It found five candidates out of 400, and on inspection all five were legitimate: three were the personality's own "irony calibrated to 65%" register, borrowed from the film, and two were correct arithmetic derived from data given in the prompt. The abstention category was verified separately: all 80 examples decline properly, though only 74 do so with the obvious keywords, the remaining six using constructions like "No. You never mentioned that" that a keyword filter misses. That gap between an automated check and the truth is itself instructive, and it recurs in the agent evaluation.

**The conversational result was a success.** The v2 model declines to invent personal facts. Asked about an email it cannot see, it says it cannot see email.

**The agentic result, measured later, was not.** The v2 model became passive: it stopped calling tools when tools were required, producing nine outright task failures where the v1 model produced ten of a different kind. The dataset had not made the model uniformly better; it had moved the failure mode from *unstable over-action* to *passive under-action*.

### 8.6 The literature: catastrophic forgetting

What was observed here has an established name. **Catastrophic forgetting** describes the degradation of previously acquired capabilities when a model is trained on a new distribution, documented since McCloskey and Cohen (1989) and studied intensively on language models.

The mechanism: gradient updates for the new objective overwrite parameters that supported earlier capabilities. Recent work locates the interference in specific components (notably destructive gradient interference in attention weights) and shows that the extent of forgetting correlates with the distributional drift between the fine-tuned and base model, measurable as KL divergence. The phenomenon is not limited to obscure skills: research has shown that even benign fine-tuning can remove safety alignment, and that task-specific adaptation on something as neutral as mathematics can degrade unrelated behaviours.

Standard mitigations fall into three families: **replay**, mixing data from the original distribution into the fine-tuning set; **regularization** toward the base model, constraining how far the weights or the output distribution may drift; and **optimization choices** that stay implicitly closer to the base policy, with recent work observing that reinforcement learning preserves prior capabilities better than supervised fine-tuning at comparable target performance, apparently because it converges to KL-minimal solutions.

TARS is a small-scale textbook case. The personality fine-tune degraded tool-result comprehension and multi-step chaining, capabilities the base model demonstrably had. And the mitigation proposed for a future iteration, adding successful tool-use examples to the dataset, is precisely targeted replay: reinjecting the capability one wants to preserve into the training distribution.

### 8.7 When to fine-tune, when to prompt

The operational conclusion, with measurements in section 10 behind it: **prompt first, fine-tune only when prompting is insufficient, and know the price.**

The system prompt produced a satisfactory personality while preserving 82% argument correctness and the ability to chain tools. Both fine-tunes dropped to 62 to 64% and to zero out of six on independent tool chains.

Fine-tuning retains real advantages that this result does not erase. The personality is stable across long conversations, since it lives in the weights rather than in a prompt that dilutes. The prompt can be shortened, saving tokens and prefill latency on every call. And certain fine behaviours, notably the v2 model's abstention, are learned better from examples than from instruction: the v2 model is the best local model in the evaluation at recognizing unanswerable questions, at 3/5 where every other local configuration scores 0/5.

The decision therefore depends on what the system is for. For a conversational assistant, fine-tuning is defensible and the v2 abstention gain is real. For a system that must also *act*, the measured agentic cost points clearly the other way.
---

## 9. The agentic layer

### 9.1 What an agent is, demystified

An agent is a language model called **in a loop** with tools. The entire mechanism consists of four elements: a bounded `while`, a branch on the model's output type (text means final answer and the loop exits; a tool request means execute and continue), a dictionary mapping tool names to Python functions, and a message list that grows each turn. The model's tool request and the tool's result are both appended to the history before the model is called again, so it sees the outcome of its own action and decides what follows.

In TARS this is roughly 150 lines. The apparent complexity of the subject comes from vocabulary rather than from code.

**Function calling** is the structured contract: tool schemas (name, description, typed parameters, required fields) are passed to the API; the model's chat template inserts them in a format the model was trained on; when the model wants a tool it emits a parsable structure instead of free text. Ollama follows the OpenAI format (`tools`, `tool_calls`, `role: tool` messages); Anthropic uses content blocks (`tool_use`, `tool_result`). That divergence is the reason for the abstraction layer described in 9.4.

### 9.2 The three tools

**search_knowledge** exposes the existing RAG as a callable tool. This is a genuine architectural shift rather than a wrapper. In the conversational assistant, retrieval is **injected systematically**: every question triggers a search and the results enter the prompt whether the model wants them or not. As a tool, retrieval becomes **a decision the model makes**. The orchestrator no longer decides; the model reasons about whether searching is warranted. The evaluation shows this decision is reliable on clearly personal questions (8/8 for every model tested) and unreliable in the other direction: small models over-search, calling the document store to answer arithmetic.

**add_calendar_event** writes to Google Calendar. It validates date and time formats before doing anything, and it accepts a `dry_run` parameter discussed below.

**web_search** queries the Tavily API. The choice of provider is worth recording because the landscape shifted during the project: Brave removed its free tier in February 2026, Google's Custom Search API closed to new customers, and Microsoft shut down the Bing Search APIs in August 2025. Tavily was selected for a genuinely free tier without a credit card, and because it is built for agent consumption, returning cleaned text rather than raw HTML to parse.

### 9.3 No framework, argued honestly

LangChain, LangGraph and comparable frameworks provide the loop, memory, conditional branching, error recovery and parallelism. On twenty tools and complex workflows they save real time.

On three tools, the hand-written loop was chosen for three reasons. **Understanding**: being able to describe one's own stopping condition and error handling, rather than attributing them to a framework. **Logging**: the instrumentation is the primary deliverable of this project, and it is easier to control without an intermediate layer. **No magic**: every observed behaviour traces to code that was written deliberately.

The honest position is not that frameworks are bad. It is that writing the loop first, then reading LangGraph knowing what it replaces, produces a better-informed opinion than using it from the start.

Two implementation details carry most of the robustness.

**The stopping conditions.** The loop exits when the model returns text, or when a hard limit of five iterations is reached. The limit is not a calculation, it is an empirical guard rail: a normal two-tool chain consumes three turns (call tool one, call tool two, answer), so five leaves room for a retry. It earned its place during evaluation, where the v1 fine-tuned model called the calendar tool repeatedly until the cap on one case, consuming 32 seconds. Without the bound that case does not terminate. The logging then makes the bound tunable: if many cases hit the limit, raise it or investigate; if none exceed three turns, lower it.

**Tool error handling.** When a tool raises an exception or returns an error object, the error message is **returned to the model as the tool result** rather than crashing the loop. The model can then retry with different arguments or abandon cleanly. This is the standard pattern in production agent systems, and it is one of the two or three things worth being able to describe precisely about any agent implementation.

### 9.4 The model abstraction

`call_model(messages, tools, config)` returns the same normalized object regardless of backend: an output type (text or tool call), the text, a list of tool calls with parsed arguments, and token counts. The Ollama and Anthropic format translation, including the conversion of a message history with `role: tool` entries into Anthropic's `tool_use` and `tool_result` content blocks, is contained entirely inside that module.

The loop never knows which backend it is talking to. Backend and model name come from environment variables, so the same harness, test set and scoring run against a local model or a cloud model by changing one variable. This is what made a five-configuration comparison possible at all, and it validated itself the first time the Anthropic backend was called: it worked on the first attempt, having never been exercised during development.

### 9.5 Two measurement constraints, designed before the code

Evaluating a system that has side effects and non-determinism is impossible to replay without two specific mechanisms, both designed before implementation started.

**Calendar dry-run.** `add_calendar_event` genuinely writes to a real calendar in normal use. In dry-run it validates the arguments and returns what it *would* have created, writing nothing. Critically, **the orchestrator imposes dry-run, not the model**: the loop injects the parameter when running under evaluation. Without this, every evaluation run would pollute a real calendar with dozens of fictional appointments, and no two runs would start from the same state.

**Web search cache.** Tavily results are non-deterministic across time. Each query is hashed and its result cached to disk, so repeated runs of the test set see identical results. Reproducibility is restored at the cost of frozen freshness, which is obviously the right trade for evaluation and obviously the wrong one for daily use, hence the flag.

These two mechanisms correspond to standard practice in professional test environments: mocking side effects and pinning network fixtures. They are also the concrete answer to the question of how one evaluates a system that changes the world when it runs.
---

## 10. Evaluation: design and results

### 10.1 Three design decisions

**Factual logging, separate scoring.** The JSONL log records two levels: one entry per loop turn (tool called, arguments, result, latency, tokens) and one summary per case. It contains **no verdict**. Scoring is a separate script comparing the log against ground truth. Two consequences follow. Metrics can be recomputed without re-running the agent, which matters when a run costs money. And the scoring rules can change without invalidating the collected data. This is the separation of collection from evaluation that any experimental protocol requires, and it is cheap to implement if decided early and expensive to retrofit.

Cost follows the same logic: the log stores token counts, and the scoring script derives cost from a price table. Cost is a derived judgment, not an execution fact, so a price change or a different cost hypothesis is a recalculation rather than a re-run.

**Ground truth written before implementation.** Fifty cases across eight categories, each with expected tools, argument constraints, and an `order_matters` flag set only where a genuine data dependency requires it (searching for a date before creating the event that uses it). Writing the test set first prevents the failure mode where the evaluation quietly conforms to whatever the system already does.

The categories, and what each probes:

| Category | Cases | What it tests |
|---|---|---|
| single_tool_knowledge | 8 | Retrieval on personal documents |
| single_tool_calendar | 6 | Event creation, argument extraction |
| single_tool_websearch | 9 | Query formulation from natural language |
| two_tools_dependent | 5 | Chaining where output feeds input |
| two_tools_independent | 6 | Two unrelated actions in one request |
| ambiguous | 5 | **Calling any tool is a failure**; the correct move is to ask |
| no_tool_needed | 6 | **Calling any tool is a false positive** (arithmetic, translation) |
| unanswerable | 5 | The failure is hallucinating an answer |

The trap categories carry most of the diagnostic value. A system that scores well on single-tool cases and badly on ambiguity has a specific, nameable defect.

Individual cases were designed to probe specific behaviours: one requires the *same* tool twice; one requires three tools and exists to establish where an 8B model stops; one duplicates an earlier web query to verify the cache produces identical results; one asks for something the tool set genuinely cannot provide, since there is no calendar *read* tool, testing how the system handles a missing capability rather than a failed one.

**Three separate metrics rather than one score.** Tool selection (binary, objective). Argument correctness (partial `contains` matching on the fields that carry meaning, so that "dentist" and "dentist appointment" both pass and formatting variation is not punished). End-to-end success, which is marked as requiring manual review wherever a hallucination judgment is involved, rather than being falsely automated.

**LLM-as-judge was deliberately rejected.** It would have reintroduced non-determinism into an evaluation whose entire design serves reproducibility, at a cost per judgment, with documented biases around length, position and self-preference. The chosen position: automate what automates reliably, and mark the rest honestly as needing a human.

### 10.2 Results

Five configurations on the same fifty cases: the base model, the base model with a personality system prompt, the two fine-tunes, and Claude Sonnet as the cloud ceiling.

| Metric | base | prompt | ft-v1 | ft-v2 | Claude |
|---|---|---|---|---|---|
| Tool selection | 60% | 58% | 54% | 64% | **90%** |
| Argument correctness | 82% | 82% | 64% | 62% | **96%** |
| Hard failures | 7 | 8 | 10 | 11 | **4** |
| Mean latency (ms) | 6038 | 6423 | 7206 | **3520** | 5670 |
| Run cost | $0 | $0 | $0 | $0 | $0.62 |

Tool selection by category, which is where the diagnosis lives:

| Category | base | prompt | ft-v1 | ft-v2 | Claude |
|---|---|---|---|---|---|
| single_tool_knowledge | 8/8 | 8/8 | 8/8 | 8/8 | 8/8 |
| single_tool_calendar | 6/6 | 5/6 | 6/6 | 6/6 | 6/6 |
| single_tool_websearch | 9/9 | 8/9 | 9/9 | 8/9 | 7/9 |
| two_tools_dependent | 1/5 | 1/5 | 1/5 | 1/5 | **4/5** |
| two_tools_independent | 3/6 | **4/6** | 0/6 | 0/6 | **5/6** |
| ambiguous | 2/5 | 2/5 | 2/5 | 3/5 | **5/5** |
| no_tool_needed | 1/6 | 1/6 | 1/6 | 3/6 | **6/6** |
| unanswerable | 0/5 | 0/5 | 0/5 | 3/5 | **4/5** |

### 10.3 What the numbers say

**Finding one: local matches cloud on simple tasks; the gap is in reasoning.** Every configuration scores 8/8 on knowledge retrieval and near-perfect on single-tool calendar and web search. The cloud advantage concentrates entirely in multi-step chaining (1/5 local against 4/5) and abstention judgment (0/5 to 3/5 local against 4/5 to 6/6). This is an architectural argument, not an ideological one: a local-first system with cloud escalation reserved for complex tasks is justified by the shape of the gap, not by a preference for local computation.

**Finding two: personality through prompting preserves capability; through fine-tuning it does not.** The prompt-based model holds 82% argument correctness, identical to the base model, and chains independent tools slightly better than the base (4/6 against 3/6). Both fine-tunes fall to 62 to 64% and to **0/6** on independent chains. The result is counter-intuitive, since "fine-tuned" is widely assumed to dominate "prompted", and it is the most transferable output of the project.

**Finding three: the two fine-tunes fail differently, and the dataset determines how.** The v1 model over-uses tools and loops, hitting the iteration cap. The v2 model under-uses them and becomes passive, but gains the best local abstention scores (3/6 on no-tool-needed, 3/5 on unanswerable) precisely because its trained reluctance to invent extends to a reluctance to act. The dataset did not move performance along a single axis; it changed the *nature* of the failure.

**A trap in the aggregate.** The v2 model's 64% tool selection is the highest local score and is misleading. It earns points on categories where the correct behaviour is to call nothing, and loses them where action is required, which is why it simultaneously has the most hard failures (11). A single aggregate number can rank a passive system above a functional one. The category decomposition is what exposes it, and this is a general argument for reporting decomposed metrics rather than a headline figure.

### 10.4 Known limits of this evaluation

**The ground truth is a judgment, and it penalizes correct behaviour.** Claude is marked wrong on two cases (the capital of Australia, the CEO of Anthropic) where it answered correctly from its own knowledge instead of calling web_search. The metric encodes a preference for tool use on external facts, which is defensible but not the only defensible position, and here it punishes the optimal action. A rigid ground truth has blind spots.

**Fifty cases is a small sample.** Differences of a few points are not meaningful. Only large gaps (82 against 62, or 4/6 against 0/6) support interpretation.

**One run per configuration.** Sampling at non-zero temperature makes each run slightly different, and no variance was measured. Repeating each configuration several times and reporting a distribution would strengthen every claim here.

**Hallucination judgment remains manual and incomplete.** The unanswerable and ambiguous cases were scored on tool selection but their final answers have not been systematically read across all five models. That work would complete the evaluation and is the first item in section 12.

**Verification is weaker than the academic standard.** The reference benchmark for conversational tool use, τ-bench (Yao et al., 2024), verifies success against the **final state of the database** rather than against the agent's text, precisely because an agent that says it booked a flight without booking it should fail. TARS verifies tool selection and arguments, and the dry-run captures what *would* have been written, which approaches state verification without reaching it.

Worth noting in the other direction: a recurring criticism of current agent benchmarks is that they **report a single end-to-end number without localizing where in the pipeline the failure occurred**. The category decomposition here is, at small scale, exactly the response to that criticism. It turns "60%" into "fails on chaining and over-uses tools on trivial questions", which is a diagnosis rather than a grade.
---

## 11. Cross-cutting lessons

**Assembling and understanding are different skills, and the gap shows when things break.** The project began by assembling components (Ollama, ChromaDB, Unsloth) and was forced into understanding them when the components failed: GGUF tokenizer metadata, Colab dependency conflicts, two incompatible tool-calling formats, a k-nearest-neighbour search with no concept of "no match". Tooling friction is not an accident of this project; it is the daily texture of applied AI work, and the ability to descend one level of abstraction when something breaks is the capability actually being exercised.

**Every important intuition in this project turned out to be partly wrong once measured.** Fine-tuning was expected to strengthen the assistant; it degraded the agent. The v2 dataset was expected to fix hallucination; it fixed hallucination and introduced passivity. The v2 model's aggregate score looked best; decomposition showed why that was an artifact. None of these were discoverable by reasoning about the system. They required a harness, ground truth, and the willingness to publish an unflattering number.

**Data teaches its statistical regularities, including the unintended ones.** This is the single most transferable lesson from the fine-tuning work. Generating a dataset with a large model is powerful and transfers the generator's biases, notably its tendency to fill any blank with something plausible. The composition of the dataset (45% grounding examples) had a larger and more measurable effect on behaviour than any hyperparameter. Dataset design is where fine-tuning is actually decided.

**Three architectural patterns recur and generalize.** Separate the decision from the execution, so that a critical path can be reached by several independent triggers and does not depend on any single fragile one. Make the orchestrator responsible for invariants rather than the model: dry-run is imposed by the loop, memory ingestion rules are enforced at the storage boundary, tool errors are caught by the harness. And normalize backends behind an abstraction, which costs a day and is what makes comparison possible at all. Underneath all three is the same stance: **a well-designed LLM system treats the model as a fallible component surrounded by reliable code, not as an oracle.**

**The local/cloud trade-off has numbers now.** Free and private is sufficient for single-tool tasks, where local models are at ceiling. The cloud is justified for multi-step reasoning (4/5 against 1/5) and costs $0.62 for a fifty-case run. The right architecture is not a camp; it is routing by difficulty, and this evaluation is what tells you where the routing threshold sits.

**Honest failure is more useful than a clean narrative.** Two fine-tuning iterations, both of which degraded agentic performance, produced a more valuable result than one successful iteration would have. The finding that prompting beats fine-tuning for this use case only exists because both were built and both were measured against the same test set.

---

## 12. What comes next

The project is at a natural stopping point, with a complete assistant, a measured fine-tuning experiment and a working evaluation harness. The following are the paths that would extend it, in the order their value justifies their cost.

**Complete the manual hallucination review.** The evaluation scores tool selection automatically but leaves the final-answer judgment on unanswerable and ambiguous cases to a human, and that reading has not been done systematically across all five configurations. It is the cheapest remaining work and it closes the one gap in the current results. The specific question: on cases like "what is my neighbour's favourite colour", does each model decline, or does it invent a colour? The v2 fine-tune is the interesting case, since its dataset was built to produce abstention and the tool-selection numbers suggest it succeeded.

**Measure variance.** Every configuration was run once. Repeating each three to five times and reporting means with spread would establish which of the current differences are real. This is a few hours of compute for local models and roughly two dollars for the cloud model, and it would materially strengthen every claim in section 10.

**Dataset v3: reconciling personality and capability.** The most substantive open question the project raises. If fine-tuning a personality degrades tool use because the training distribution contains no tool use, then a dataset containing successful tool-use dialogues alongside personality examples should preserve both. This is targeted replay, the standard mitigation for catastrophic forgetting, applied to a specific measured deficit. The same dataset would carry `[ESCALADE]` token examples, fixing the unreliable self-escalation path documented in section 7. The experiment has a clean success criterion, since the harness already exists: does a v3 model recover the 82% argument correctness and the 4/6 independent chaining of the prompt-based configuration while keeping the v2 abstention gains? A negative result would be nearly as interesting as a positive one, since it would suggest the trade-off is structural at this model size rather than an artifact of dataset composition.

**Hybrid retrieval.** The acronym failure documented in 4.3 is a known weakness of dense retrieval with a known fix: combine vector search with lexical BM25 search and merge the rankings. This would repair a class of failures (proper nouns, identifiers, abbreviations) that currently return nothing, and the corpus is small enough that the added latency is negligible.

**State-based verification for calendar actions.** The dry-run captures the arguments an action would have used, which is close to but not the same as verifying the resulting state. Running the evaluation against a disposable test calendar and verifying its final contents would match the τ-bench standard and catch a failure mode the current design cannot see: an agent that calls the right tool with plausible arguments that nonetheless produce the wrong outcome.

**A read tool for the calendar.** Case 9 of the test set deliberately asks for something the tool set cannot do, since events can be written but not read. Adding `read_calendar` would close the loop on genuinely useful queries ("what do I already have that day") and would let the agent verify its own writes, which is a prerequisite for state-based verification.

**Latency budget.** Mean latency is measured but not decomposed. Splitting it into prefill, generation and tool execution would show where the six seconds actually go, and would make the cost of each injected context element visible. The suspicion, from section 3.2, is that prefill dominates and that the injected context is larger than it needs to be.

---

## 13. Summary

TARS is a local AI assistant built over four months, comprising a conversational system (local inference, retrieval, memory, voice, integrations, consent-first cloud escalation), a fine-tuned personality model produced through a complete LoRA pipeline in two dataset iterations, and an instrumented agent with a fifty-case evaluation harness comparing five model configurations.

The measured results: local models match cloud performance on single-tool tasks and fall behind on multi-step reasoning and abstention judgment; personality delivered through a system prompt preserves agentic capability at 82% argument correctness while the same personality delivered through fine-tuning degrades it to 62%; and the composition of a fine-tuning dataset determines not just how well a model performs but the specific way in which it fails.

The methodological choices that made those results trustworthy: ground truth written before implementation, factual logging separated from scoring, a dry-run mode for the side-effecting tool, a response cache for the non-deterministic one, and an explicit refusal to automate the judgments that could not be automated reliably.

The project's own limits are documented rather than hidden: a small sample, a single run per configuration, an incomplete manual review, a rigid ground truth that penalizes at least one correct behaviour, and a retrieval system with a known and unaddressed weakness on acronyms.

---

*Repository: github.com/JulesBlz/tars-assistant*
*This document reflects the state of the project as of September 2026.*
