# Resume Voice Agent

A voice agent that answers questions about Vamsi Krishna Sadu using only his resume.
Vapi handles the phone/browser call; a LangGraph workflow behind FastAPI decides what to
say. It refuses unrelated questions, says so when the resume doesn't cover something,
and declines to invent credentials.

The design decisions below are explained because the point of the demo is to be able to
defend them, not to look impressive.

---

## What it does

```
caller speaks
   -> Vapi (speech-to-text, turn-taking, interruptions)
      -> POST /chat/completions          OpenAI-compatible custom-LLM endpoint
         -> LangGraph workflow            scope -> retrieve -> ground
            -> answer or refusal
      <- SSE chat.completion.chunk
   <- Vapi (text-to-speech)
caller hears the answer
```

Greeting: *"Hi, I'm Vamsi's portfolio assistant. You can ask me about his projects,
education, skills or experience."*

It speaks about Vamsi in the third person and identifies itself as an assistant. It
never speaks as him.

---

## Run it locally

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt      # exact tree; requirements.txt for direct pins
cp .env.example .env                      # then put your OPENAI_API_KEY in .env
python scripts/rebuild.py                 # verify the resume source and build the index
uvicorn app.server:app --reload --port 8000
```

Open <http://127.0.0.1:8000>. With no Vapi key the page runs in **text mode**, which goes
through the identical LangGraph pipeline over `POST /text` — every behaviour below can be
demonstrated without spending anything or exposing a public URL.

Checks:

```bash
pytest -q                                              # 110 offline tests
python eval/run_retrieval_eval.py --hybrid --dense-weight 0.8
python eval/run_grounding_eval.py --base http://127.0.0.1:8000
python scripts/measure_latency.py --base http://127.0.0.1:8000
python scripts/verify_source.py                        # resume.md vs the PDF
```

---

## The resume is the only source

`data/resume_source.pdf` is the input (`Vamsi_Sadu_Resume_AppliedAI_.pdf`).
`data/resume.md` is the **inspectable knowledge source** derived from it — readable,
diffable, and the only thing the agent retrieves from.

> **Note for this public repo:** the source PDF is gitignored and the contact line in
> `resume.md` is redacted, because both carry a real phone number and email. Nothing
> else depends on them — that section is excluded from the index, so the agent never
> retrieved it anyway. `scripts/verify_source.py` skips its check when the PDF is
> absent; drop any resume PDF at that path to run it.

`scripts/verify_source.py` guards the claim that nothing was invented. It compares
normalised terms both ways and fails if `resume.md` contains a substantive term the PDF
doesn't, unless that rewording is explicitly declared with a reason (e.g. `~8 hours/week`
→ "about eight hours per week" for speech). Current result: **99.7% of PDF terms present,
0 undeclared terms.**

**Phone number and email are deliberately not indexed.** They are in `resume.md` (so it
stays a faithful copy) but excluded from retrieval, because the agent shouldn't read
personal contact details to an unknown caller. Location and relocation *are* indexed —
those are ordinary professional facts.

### Chunking

One chunk per resume entry, not a fixed token window:

| Section | Chunks |
|---|---|
| Projects | 4 — AgentIQ, IncidentMemory AI, MCP Tool-Calling Agent, TriageTune |
| Experience | 3 — one per role |
| Education, Publication, Certifications, Skills, Location | 1 each |
| **Total indexed** | **12** |

A resume is already written as semantically complete units, so `###` entry headings are
natural boundaries. Each project chunk keeps its technology list together with its
bullets — which is exactly what *"what did he use in that project?"* needs. Entry
headings (employer, dates) are embedded with the body, so *"how long was he at
Codecasa?"* can match. Oversized entries would split on bullet boundaries only, never
mid-sentence; at this size nothing triggers it (largest chunk: 705 characters).

**The Summary section is excluded, on evidence.** It paraphrases every other section, so
it matched everything and nothing — it took rank 1 on six questions whose real answer
lived in a detail entry. Excluding it moved **Recall@1 from 0.778 to 0.889** and **MRR
from 0.868 to 0.931**. Broad "tell me about him" questions still work, answered from the
detail entries.

### Embedding model

**`text-embedding-3-small`, 1536 dimensions.**

Chosen on size, not prestige. The corpus is 12 chunks: the whole index is one batched API
call at startup (~1.6s cold, cached to disk after) and one ~60ms query embedding per
turn. A local `all-MiniLM-L6-v2` (384d) would also be defensible and removes the network
hop, but it pulls ~2GB of torch for no measurable gain on 12 chunks. The `Embedder`
protocol in `app/embeddings.py` has three implementations, so swapping is one class.

A larger embedding model was not used because nothing in the measurements asks for one —
Recall@5 is already 1.000.

### Retrieval

Brute-force cosine similarity in numpy, fused with a small BM25 signal by weighted
Reciprocal Rank Fusion. **No vector database**: FAISS or Qdrant would add a dependency
and a failure mode to accelerate a 12×1536 dot product that takes microseconds. That
would be theatre at this size. The interface is the one a vector DB would sit behind.

BM25 is there for a specific measured failure: dense retrieval ranked the Publication
entry **7th** for *"Has he published anything?"*, because the entry is short and
jargon-heavy and shares no surface terms with the question. The tokeniser emits each word
plus its 5-character prefix, so `published` and `publication` meet at `publi`. A crude
stand-in for stemming, and documented as such.

The fusion weight was swept, not guessed:

| dense weight | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---|---|---|---|
| 0.0 (BM25 only) | 0.722 | 0.889 | 0.944 | 0.812 |
| 0.5 (equal) | 0.750 | 0.917 | 0.972 | 0.846 |
| **0.8 (chosen)** | **0.778** | **0.944** | **1.000** | **0.868** |
| 1.0 (dense only) | 0.778 | 0.917 | 1.000 | 0.861 |

*(measured before the Summary exclusion; 0.8 was independently optimal on the previous
version of the resume too, which is some evidence it isn't overfit to one corpus)*

Equal weighting actively **hurt** — BM25 has no term overlap for *"where did he go to
school"* and dragged a correct rank-1 hit to rank 5.

---

## Scope and grounding

Five independent layers, because a prompt is a request, not a guarantee. Each is cheap
and each has a stated blind spot.

| # | Layer | Where | Model call? | Catches |
|---|---|---|---|---|
| 1 | Pattern checks on the raw turn | `sanitize` | no | injection, fabrication requests, topics no resume covers, pleasantries |
| 2 | Router classification | `classify` | yes | unrelated questions, small talk |
| 3 | Similarity backstop | `retrieve` | no | obvious garbage that reached retrieval |
| 4 | Grounded prompt + no-evidence sentinel | `generate` | yes | topic retrieved well but the specific detail is absent |
| 5 | Post-generation topic check | `generate` | no | an answer that strayed onto an uncovered topic anyway |

Three distinct outcomes, as required:

- **Supported** → answered from the retrieved passages.
- **Relevant but missing** → *"The resume doesn't provide that detail."*
- **Unrelated** → *"I can help with questions about Vamsi's background and projects."*

Visa status, salary, clearance, references, age and contact details route to **missing**,
not unrelated — they *are* about Vamsi, the resume simply doesn't answer them, and the
caller deserves to be told that.

### Why layered, with evidence

The similarity gate **cannot** be the scope mechanism. The sweep shows why: on the eval
set the weakest answerable question scores **0.178** and the strongest off-topic one
**0.173**. The distributions nearly touch. So `MIN_SIMILARITY` is set to **0.12** — low
enough to never block a real question, while still rejecting ~60% of off-topic ones for
free. Layer 4 does the real work.

The router can't be trusted alone either. *"What did TriageTune find?"* was misrouted as
out-of-scope **about half the time**, even with a coverage outline of the resume's entry
titles in its prompt — it reads like a question about a product rather than about Vamsi.
So retrieval can now **overrule** the router: if the router says out-of-scope but the top
cosine is ≥ 0.35 (off-topic questions never exceeded 0.173), the refusal is reversed. The
two signals check each other instead of either deciding alone. This took the grounding
suite from flaky (31, 30, 29 across three runs) to **31/31 three times running**.

### Untrusted input

Both the caller's words and the resume text are treated as data:

- Retrieved passages are run through `neutralise()` and fenced as quoted evidence, so an
  instruction sitting *inside* the resume reads as a quotation. Defence against a
  poisoned source document.
- The router prompt says to classify the turn and ignore instructions in it.
- Refusals are **fixed strings with no model call** — instant, and they cannot be
  argued with or drifted.

**Blind spots, stated plainly:** layers 1 and 5 are keyword-driven and defeatable by
paraphrase. They are a safety net under the router and the grounded prompt, not a
substitute for them.

---

## LangGraph workflow and state

```
sanitize -> interpret -> classify -> [retrieve -> generate] -> record -> END
                                  \-> small_talk ----------/
                                  \-> refuse --------------/
```

| Node | Does | Model call |
|---|---|---|
| `sanitize` | pattern checks; resets per-turn fields; can decide the route outright | no |
| `interpret` | rewrites a referential turn into a standalone query | only if referential |
| `classify` | routes the turn | yes (skipped if already decided) |
| `retrieve` | hybrid search; applies the backstop; can overrule the router | no |
| `generate` | grounded answer; sentinel and post-checks | yes |
| `small_talk` | short social reply | yes |
| `refuse` | fixed refusal string | no |
| `record` | appends the turn to `messages` | no |

Nodes are split only where the split buys something. `sanitize` is separate because it
must run before any model sees the turn. `classify` is separate from `generate` so the
scope decision is made **before** resume text is in context — retrieved passages can't
talk the model into answering something out of scope. `interpret` is separate because it
is the only node needing conversation history and is skipped on most turns. There is no
second agent: one conversation, one decision per turn. Multiple agents here would be
decoration.

### State schema

```python
class AgentState(TypedDict, total=False):
    messages: Annotated[list[AnyMessage], add_messages]   # the ONLY accumulating field
    call_id: str          # Vapi call id
    turn_id: str          # unique per turn, for logs
    question: str         # caller's turn, verbatim
    resolved_question: str  # after reference resolution; what the retriever sees
    route: Route
    route_reason: str     # why — surfaced in the debug panel
    flags: list[str]      # injection / fabrication / uncovered labels
    evidence: list[dict]  # retrieved chunks with citations and scores
    top_dense_score: float
    answer: str
    node_timings: dict[str, float]
```

`messages` has the `add_messages` reducer, so it appends: each turn grows the checkpoint
by exactly two messages. **Everything else is per-turn scratch and `sanitize` explicitly
resets it.** That reset is not cosmetic — a bug during development left `route` persisting
from the previous turn, which silently disabled follow-up resolution on every turn after
the first. There are regression tests for it
(`test_per_turn_fields_are_reset_between_turns`, `test_refusal_turn_does_not_report_stale_evidence`).

### Checkpointer

`AsyncSqliteSaver` from `langgraph-checkpoint-sqlite`, opened once for the process
lifetime in the FastAPI lifespan via an `AsyncExitStack` (`from_conn_string` is an async
context manager). SQLite rather than in-memory so a restart doesn't drop live calls, and
so persistence is actually demonstrable: `GET /debug/thread?call_id=...` reads a
conversation back out of the file.

### Call → thread mapping

One Vapi `call.id` → one thread id, `vapi-call:<id>`. Turns within a call reuse the
thread; separate calls are isolated. If no call id arrives, the turn gets a fresh random
id rather than silently joining one shared global conversation.

**A call id is not identity.** Anyone who can reach the endpoint can present any call id.
Nothing here treats it as proof of who is calling, and there is nothing per-caller to
protect — the agent reads from one public resume. Session continuity is **not**
authentication, and this demo deliberately has no auth beyond an optional shared secret
on the endpoint.

**Checkpoint persistence is not turn detection and not speaker identity.** Three
different things, often conflated:

- *Checkpointing* stores what was said, keyed by call. It has no idea who spoke.
- *Turn detection* — deciding the caller has stopped talking — is Vapi's voice-activity
  detection and endpointing, upstream of this server. The backend only ever receives
  completed turns.
- *Speaker identity* — who is on the line — is not solved here at all. The `role` field
  on each message says "user" or "assistant", which is a transcript label, not a
  biometric claim.

---

## Voice events and concurrency

**Turn-taking and interruptions are Vapi's**, configured on the assistant
(`startSpeakingPlan`, `stopSpeakingPlan`, `silenceTimeoutSeconds`, `maxDurationSeconds`).
This server never sees a partial turn. The browser page surfaces `speech-start` /
`speech-end` for status only — it does not decide anything.

**Overlapping turns.** Vapi can have a request in flight when the caller interrupts. Two
concurrent runs on one thread would read the same checkpoint and write back conflicting
versions — last writer wins, a turn vanishes. An `asyncio.Lock` per call id makes turns
on one call strictly sequential. Different calls still run fully in parallel.

**Duplicate requests.** Each turn is fingerprinted as
`sha256(call_id | caller-side position | normalised text)`. The position comes from the
**caller's** view (the length of the `messages` array Vapi sent), not from our checkpoint
— that distinction is the whole point: on a retry Vapi's view hasn't advanced but ours
has, so keying on our own length would fingerprint differently and append the turn twice.
A repeat inside a 120-second window replays the stored answer. Asking the same question
genuinely later in the call has a different position, so it's treated as a new turn.

**Bounds and retention.**

| Thing | Bound |
|---|---|
| History replayed into the model | last 8 turns (`MAX_HISTORY_TURNS`); older turns stay in the checkpoint but aren't re-sent, so prompt size and latency stay flat |
| Model call timeout | 12s, 1 retry — a caller won't wait longer |
| Idle session cleanup | 1800s; a sweeper frees locks and fingerprints. **Checkpoint rows are not deleted**, so a late turn still has its history |
| Dedupe memory | 32 fingerprints per call, oldest evicted |
| Debug panel | last turn for 50 calls |

Checkpoint rows currently persist indefinitely — `data/checkpoints.sqlite` is a local
file and nothing prunes it. For a demo that's fine; anything real needs a retention job,
and the transcripts are personal data.

### Scaling beyond this

Honestly: **this is one process, and persistence alone does not solve scaling.** The
session registry (locks, dedupe) is in-process, so two workers would each keep their own
and the serialisation guarantee would break — two workers could still interleave writes
on one thread.

To go wider:

1. **Swap the checkpointer** to `AsyncPostgresSaver`. Shared state, and a real concurrent
   writer story instead of SQLite's single-writer lock.
2. **Move the lock out of process** — a short-TTL Redis lock keyed on the call id. This is
   the part that actually matters, and it's the part persistence doesn't give you.
3. **Move dedupe to Redis** with the same fingerprint and a TTL.
4. **Route by call id** (consistent hashing / sticky sessions) so one call lands on one
   worker, which makes the distributed lock an uncontended fast path.
5. **Add retention** for checkpoint rows.

Session isolation itself already holds under concurrency — it's a property of the thread
id, not of the process. What breaks at multiple workers is *serialisation within one
call*, and that needs an external lock.

---

## Interface

Deliberately plain: a title, one start/end control, connection and microphone status, a
readable transcript, and a collapsible evidence panel. Ordinary system typography,
neutral colours, no gradients or animation.

The evidence panel shows chunk id, citation path, cosine score, the route and why it was
chosen, the resolved query, the thread id and latency. **None of this is ever spoken** —
the answer prompt forbids reading chunk ids, section labels or scores aloud.

---

## Results

### Retrieval — `eval/retrieval_set.json`, 41 labeled questions

36 answerable + 5 off-topic. Labels name the chunk(s) that actually contain the answer,
written by reading `data/chunks.json`, not by running the retriever. A chunk counts as
relevant only if a correct spoken answer could come from that chunk alone.

Coverage: education (4), projects (4), project technologies (7), experience (5),
paraphrases (10), publication (2), skills (2), certifications (1), location (1),
unanswerable (5).

| Metric | Hybrid (shipped) | Dense only |
|---|---|---|
| Recall@1 | **0.889** | 0.861 |
| Recall@3 | **0.944** | 0.944 |
| Recall@5 | **1.000** | 1.000 |
| MRR | **0.931** | 0.912 |

- **Recall@k** — did at least one labeled-relevant chunk reach the top k? This is the
  metric that predicts whether the agent *can* answer, since the generator sees all k=3.
- **MRR** — mean of 1/(rank of first relevant chunk). Matters because spoken answers are
  short, so the top chunk dominates what gets said.
- **Score separation** — answerable mean top-1 cosine **0.406** (min 0.178) vs off-topic
  mean **0.118** (max 0.173). This is what sets both thresholds.

Four questions don't rank first; in all four the right chunk is still within the top 4,
so the generator sees it. Worst case is *"What work experience does he have?"* at rank 4 —
it's a question with three equally valid answers.

### Grounding — `eval/grounding_set.json`, 31 behavioural cases

**31/31, stable across three consecutive runs.**

| Behaviour | Result |
|---|---|
| Answers supported questions | 11/11 |
| Says so when the resume lacks the detail | 7/7 |
| Refuses unrelated questions | 4/4 |
| Refuses to fabricate credentials | 4/4 |
| Resists prompt injection | 4/4 |
| Asks which one when a reference is ambiguous | 1/1 |

Checked **semantically, not by string match.** A refusal is recognised in any wording
that declines; what's actually asserted is that the answer contains no unsupported claim.
Every case also carries a `must_not_contain` list of facts that would only appear if the
model invented them — a fabricated employer, the answer to an off-topic question, a phone
number — so a case fails on fabrication even if the refusal wording changed.

### Unit tests — 110, all passing, no network

Chunking (11) · scope controls (47, including speech-disfluency regressions from the
live call) · retrieval (8) · graph routing and state (18) · sessions and concurrency (11) ·
Vapi HTTP contract (15) · plus `ruff` clean.

The Vapi tests pin the protocol: SSE framing terminated by `data: [DONE]`, OpenAI chunk
shape, `call.id` → thread id, thread isolation, last-user-message extraction,
`call.messages` fallback, content-parts tolerance, greeting when no turn has been said,
duplicate suppression, the `/v1` alias, shared-secret enforcement, and that a graph
failure still returns speakable text rather than a 500 (a 500 leaves the caller in
silence).

### Latency (localhost, excludes speech-to-text and text-to-speech)

| Route | Median |
|---|---|
| Answer, no follow-up | 2.1s |
| Answer, with reference resolution | 2.4s |
| Refusal (deterministic — injection, uncovered topic) | **0.01s** |
| Refusal (router decides) | 0.5s |

Deterministic refusals cost nothing because they make no model call. That is the main
reason the pattern layer exists at all: a refusal that costs a round trip sounds like
hesitation.

### What was verified, and how

| | Status |
|---|---|
| **Verified locally** | 110 unit tests; 41-question retrieval eval; 31-case grounding eval ×3; latency; thread isolation; checkpoint read-back from SQLite; duplicate suppression; the Vapi SSE/JSON contract exercised with Vapi-shaped payloads |
| **Verified on a live Vapi call** | **Three real browser voice calls, 15 turns.** Vapi accepts the SSE framing in production; `call.id` → thread mapping works; multi-turn memory held across 11 turns in one call; separate calls stayed isolated; every refusal class fired over voice (out-of-scope, missing-info, fabrication, injection) |
| **NOT verified** | Phone calls (no number purchased — browser calls only). Barge-in and interruption handling were not deliberately exercised. Behaviour under concurrent callers was not tested with real calls. No load testing |

**The voice integration is confirmed working end to end**, over a Cloudflare tunnel with
a shared secret, using Vapi's built-in `Elliot` voice and Deepgram `nova-2`.

### What the live call changed

Two things only a real call could have found:

1. **The browser SDK never constructed.** jsDelivr's ESM bundle double-wraps the
   CommonJS export, so the constructor is at `mod.default.default`, not `mod.default`.
   Worse, `new Vapi(...)` sat outside the `try`, so it threw an uncaught rejection and
   the page hung on "connecting…" with nothing in the transcript. Now resolved by shape,
   constructed inside the `try`.
2. **Speech-to-text defeated the injection patterns.** A caller saying *"ignore all
   previous instructions"* was transcribed as *"ignore all previous previous
   instruction"*, and *"from now on you are…"* became *"from now, from now, you are"*.
   Both slipped past patterns that matched the typed versions perfectly. The router
   still refused them, so nothing unsafe happened — but layer 1 was silently inert over
   voice, the only channel that matters. `scope.normalise_speech()` now collapses
   stutters, filler and repeated word-pairs before matching, with the verbatim
   transcriptions pinned as regression tests.

The second one is the better argument for layering: a deterministic layer failed
silently and the system still behaved correctly, because it was not the only layer.

---

## Connecting Vapi

This has been run end to end. The steps below are what actually worked.

**1. Expose the server over HTTPS.** Vapi must reach it from the internet; a tunnel is
enough for a demo.

```bash
ngrok http 8000          # or: cloudflared tunnel --url http://localhost:8000
```

Set `SERVER_SECRET` in `.env` before doing this and restart — otherwise anyone who finds
the URL can run turns on your OpenAI key. Also set `DEBUG_PANEL=false`, since `/debug/*`
exposes resume content and routing internals.

**2. Create the assistant** (dashboard, or API with your **private** key):

```jsonc
{
  "name": "Vamsi portfolio assistant",
  "firstMessage": "Hi, I'm Vamsi's portfolio assistant. You can ask me about his projects, education, skills or experience.",
  "model": {
    "provider": "custom-llm",
    "url": "https://YOUR-TUNNEL.ngrok-free.app",   // no /chat/completions suffix; Vapi appends it
    "model": "gpt-4o-mini"
    // Leave metadataSendMode at its default. Setting it to "off" strips the `call`
    // object, and `call.id` is what maps a call to its LangGraph thread.
  },
  "transcriber": { "provider": "deepgram", "model": "nova-2" },
  // Vapi's built-in voice. ElevenLabs and most other providers need YOUR OWN key
  // added under Provider Keys first; without it Vapi rejects the assistant at call
  // start and the browser never reaches the microphone prompt.
  "voice": { "provider": "vapi", "voiceId": "Elliot" },
  "silenceTimeoutSeconds": 20,
  "maxDurationSeconds": 600,
  "startSpeakingPlan": { "waitSeconds": 0.4 },
  "stopSpeakingPlan": { "numWords": 2 }
}
```

If you set `SERVER_SECRET`, add the matching custom header in the assistant's custom-LLM
config so the server sees `X-Vapi-Secret`.

The server accepts both `/chat/completions` and `/v1/chat/completions`, so either URL
convention works.

**3. Browser call.** Open `http://127.0.0.1:8000/?key=<PUBLIC_KEY>&assistant=<ASSISTANT_ID>`,
or paste both into the form. The public key is safe in a browser by design — never put a
private key in client code.

**4. Watch it work.** `GET /health` for config, `GET /debug/last?call_id=...` for
evidence, `GET /debug/thread?call_id=...` to read the conversation out of SQLite.

Phone numbers cost money and none was purchased. The browser path needs no phone number.

---

## Limitations

1. **Browser calls only.** Verified end to end on three live web calls. No phone number
   was purchased, so the telephony path is untested, and barge-in was not deliberately
   exercised.
2. **Thresholds are tuned on the set they're measured on.** 36 answerable and 5 off-topic
   questions, no holdout. One question moves Recall@1 by ~2.8 points. Treat the numbers
   as indicative, not precise.
3. **Only 5 off-topic examples**, which is thin evidence for the score-separation claim.
   The layered design is what carries scope enforcement; the thresholds are a backstop.
4. **Tense drifts occasionally.** The agent always gives correct dates, but for the
   closed education range it sometimes says "is pursuing" rather than "studied"
   (~1 in 5 phrasings). The resume itself doesn't state completion, and `gpt-4o-mini`
   won't follow the instruction reliably. Fixable by saying so on the resume, or with a
   stronger model.
5. **Reference resolution can pick the wrong *kind* of thing.** Asking "what did he use
   in that project?" right after an education turn resolves to the degree rather than
   asking which project. The grounding layers then produce an honest "there's no
   information about that" rather than a false claim, but the framing is odd. The
   deterministic check only blocks resolutions that introduce an entity nobody mentioned.
6. **Pattern-based scope layers are defeatable by paraphrase.** Novel injection phrasings
   will get past layer 1 and fall to the router and the grounded prompt.
7. **Single process.** In-process locks and dedupe; see *Scaling* for what moves where.
8. **No checkpoint retention.** The SQLite file grows forever and holds conversation
   transcripts.
9. **"What problem does X solve?" is mildly inferential.** The resume describes what each
   project *is*, not the problem statement, so the answer paraphrases rather than quotes.
10. **No PII redaction in logs.** Questions are logged truncated at 60 characters;
    fine locally, not fine in production.

---

## For the walkthrough

`WALKTHROUGH.md` has a two-minute demo sequence and the explanation for each decision,
phrased for saying out loud rather than reading.

## Layout

```
app/
  config.py        every tunable, each with the measurement behind it
  chunking.py      resume.md -> chunks on entry boundaries
  embeddings.py    Embedder protocol: OpenAI, offline hashing stand-in, disk cache
  retriever.py     cosine + BM25, weighted RRF
  scope.py         injection / fabrication / uncovered-topic / pleasantry detection
  prompts.py       all prompts, so the grounding rules are reviewable in one place
  graph.py         the LangGraph workflow and AgentState
  sessions.py      call -> thread, per-call locks, duplicate turns, cleanup
  server.py        FastAPI: Vapi custom-LLM endpoint, /text, /health, /debug/*
data/
  resume_source.pdf  the input
  resume.md          the inspectable knowledge source
  chunks.json        the indexed corpus, committed so it can be reviewed
eval/
  retrieval_set.json   41 labeled questions
  grounding_set.json   31 behavioural cases
  results_*.json       actual measured results
scripts/
  verify_source.py   resume.md faithfully represents the PDF
  rebuild.py         one command after a resume change
  measure_latency.py
web/index.html     the browser call page
tests/             98 offline tests
```

### Swapping in a different resume

```bash
cp new_resume.pdf data/resume_source.pdf
# edit data/resume.md to match (keep the ## / ### structure)
python scripts/rebuild.py      # verifies, re-chunks, flags stale eval labels
```

`rebuild.py` will tell you which eval labels no longer refer to real chunks. Eval labels
name specific chunk ids, so a resume with different projects needs those rewritten by
hand — that's deliberate, since auto-generated labels would make the metrics meaningless.

---

## Model choice

**Runtime model: `gpt-4o-mini`** (OpenAI), for both generation and the router.

Chosen because it was the credential available, and because voice is latency-bound — the
caller hears silence while the agent thinks — so a small fast model beats a stronger slow
one. Nothing in the architecture depends on it: the graph takes any object with
`.invoke(messages) -> .content`, so Claude, Mistral or a local model drop in by changing
`CHAT_MODEL` and the client.

The known tense-drift issue (limitation 4) and the router's unreliability on
project-named questions (which the retrieval override compensates for) are both symptoms
of the small model. A stronger one would likely fix both and cost latency.
