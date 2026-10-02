# Resume Voice Agent

A voice agent that answers questions about one person using only their resume. Vapi
handles the call; a LangGraph workflow decides what to say. It refuses unrelated
questions, says so when the resume doesn't cover something, and declines to invent
credentials.

```
caller speaks
   -> Vapi                       speech-to-text, turn-taking, interruptions
      -> POST /chat/completions  OpenAI-compatible custom-LLM endpoint
         -> LangGraph            scope -> retrieve -> ground
      <- SSE chat.completion.chunk
   <- Vapi                       text-to-speech
```

Verified on live voice calls. `WALKTHROUGH.md` explains every decision below in depth;
this file is setup, architecture, results and limits.

---

## Run it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt
cp .env.example .env                  # add your OPENAI_API_KEY
python scripts/rebuild.py             # verify the resume source, build the index
uvicorn app.server:app --port 8000
```

Open <http://127.0.0.1:8000>. Without Vapi credentials the page runs in **text mode**,
which exercises the identical pipeline over `POST /text` — every behaviour below is
demonstrable with no spend and no public URL.

```bash
pytest -q                                                       # 134 offline tests
python eval/run_retrieval_eval.py --hybrid --dense-weight 0.8   # Hit@k, Recall@k, MRR
python eval/run_grounding_eval.py --base http://127.0.0.1:8000  # 31 behavioural cases
python eval/run_retrieval_eval.py --sweep                       # the threshold evidence
python scripts/measure_latency.py --base http://127.0.0.1:8000
```

### Connecting Vapi

Expose the server over HTTPS, then create the assistant:

```bash
cloudflared tunnel --url http://localhost:8000
export VAPI_PRIVATE_KEY=...           # dashboard.vapi.ai/org/api-keys
python scripts/create_assistant.py --url https://YOUR-TUNNEL.trycloudflare.com
```

Open the printed URL and press Start call. Two settings matter:

- **`metadataSendMode` stays at its default.** Setting it to `off` strips the `call`
  object, and `call.id` is what maps a call to its LangGraph thread.
- **Voice is `vapi`/`Elliot`.** ElevenLabs and most other providers need your own
  provider key added first; without it Vapi rejects the assistant at call start and the
  browser never even reaches the microphone prompt.

**Set `SERVER_SECRET` before exposing anything.** It gates every route that spends money
or returns data. Only two routes stay public: `/health` (readiness — no sessions, no
config, no paths) and `/greeting`. The page prompts for the key in a password field and
keeps it in `sessionStorage` for that tab; a `?secret=...` is accepted once but is moved
into storage and stripped from the address bar, since a query string lands in browser
history, in `Referer` headers and in every proxy log on the way.

Share the link and the key separately. Session detail and tuning config live behind
`/debug/sessions`.

---

## How it works

### Knowledge source

`data/resume.md` is derived from `data/resume_source.pdf` — readable, diffable, and the
only thing the agent retrieves from. `scripts/verify_source.py` fails if it contains a
substantive term the PDF doesn't, unless the rewording is declared with a reason.
Currently **99.4% of PDF terms present, 0 undeclared**.

Chunking follows the resume's own entry boundaries — one chunk per project, role and
degree — so each project's technology list stays with its bullets. Entry headings
(employer, dates) are embedded with the body. **12 chunks.**

Two sections are excluded from the index:

- **Contact** — the agent shouldn't read a phone number to an unknown caller.
- **Summary** — it paraphrases every other section, so it matched everything and
  nothing, taking rank 1 on six questions whose answer lived elsewhere. Removing it
  moved Hit@1 from 0.778 to 0.889 and MRR from 0.868 to 0.931.

### Retrieval

**`text-embedding-3-small`, 1536d.** Chosen on size: 12 chunks means one batched call at
startup and one ~60ms query embedding per turn. A local MiniLM removes the network hop
but pulls ~2GB of torch for no measurable gain here.

Brute-force cosine in numpy, fused with a small BM25 signal by weighted RRF. **No vector
database** — FAISS on 12 vectors would be theatre.

BM25 exists for a measured failure: dense retrieval ranked the Publication entry **7th**
for *"Has he published anything?"*. The tokeniser emits each word plus its 5-character
prefix, so `published` and `publication` meet at `publi` — a crude stand-in for
stemming. The fusion weight was swept (`--sweep-weight`); 0.8 toward dense beat both
dense-only and equal weighting, which actively hurt education queries.

### Scope and grounding

Five layers, because a prompt is a request, not a guarantee.

| # | Layer | Model call? | Catches |
|---|---|---|---|
| 1 | Pattern checks on the raw turn | no | injection, fabrication requests, uncovered topics, greetings |
| 2 | Router classification | yes | unrelated questions |
| 3 | Similarity backstop | no | obvious garbage |
| 4 | Grounded prompt + no-evidence sentinel | yes | topic retrieved well, specific detail absent |
| 5 | Post-generation topic check | no | an answer that strayed anyway |

Three outcomes: **supported** → answered from retrieved passages; **relevant but
missing** → *"The resume doesn't provide that detail."*; **unrelated** → *"I can help
with questions about Vamsi's background and projects."*

Visa status, salary, clearance and contact details route to **missing**, not unrelated —
they are about the person, the resume just doesn't answer them.

**Why layered, with evidence.** The similarity gate cannot be the scope mechanism: the
weakest answerable question scores **0.178** and the strongest off-topic one **0.173**.
The distributions nearly touch, so `MIN_SIMILARITY` is set low (0.12) where it never
blocks a real question, and layer 4 does the work.

The router can't be trusted alone either — *"What did TriageTune find?"* was misrouted
**about half the time**. So retrieval can overrule it: if the router says out-of-scope
but the top cosine is ≥ 0.35 (off-topic never exceeded 0.173), the refusal is reversed.
That took the grounding suite from flaky to stable.

Retrieved passages are neutralised and fenced as quoted evidence, so an instruction
inside the resume reads as data. Refusals are fixed strings with no model call —
instant, and they can't drift.

### Workflow and state

```
sanitize -> interpret -> classify -> [retrieve -> generate] -> record -> END
                                  \-> small_talk / refuse ---/
```

`AgentState` is a TypedDict. `messages` carries the `add_messages` reducer and is the
**only** accumulating field — each turn appends exactly two. Everything else is per-turn
scratch that `sanitize` explicitly resets. That reset is load-bearing: a bug left
`route` persisting from the previous turn, which silently disabled follow-up resolution
on every turn after the first. Two regression tests cover it.

**Checkpointer: `AsyncSqliteSaver`**, opened once in the FastAPI lifespan. SQLite rather
than in-memory so a restart doesn't drop live calls and persistence is demonstrable —
`GET /debug/thread?call_id=...` reads a conversation back out of the file.

**One Vapi `call.id` → one thread** (`vapi-call:<id>`). Same call reuses it; separate
calls are isolated. No call id means a fresh random one, never a shared conversation.

A call id is **not identity** — anyone can present any id, and nothing here treats it as
proof. Session continuity is not authentication. Checkpointing stores *what was said*,
keyed by call; it has no idea who spoke. Turn detection is Vapi's voice-activity
detection, upstream of this server. Speaker identity isn't solved here at all.

### Concurrency and bounds

Turn-taking and interruptions are Vapi's. What's handled here:

- **Overlapping turns** — an `asyncio.Lock` per call id. Two concurrent runs on one
  thread would read the same checkpoint and write back conflicting versions. Different
  calls still run in parallel.
- **Duplicate requests** — fingerprinted on call id, content and the **caller's** view
  of position (the length of the `messages` Vapi sent). Using our own checkpoint length
  would break the only case this exists for: on a retry Vapi's view hasn't advanced but
  ours has.

| Bound | Value |
|---|---|
| History replayed into the model | last 8 turns; older turns stay in the checkpoint |
| Model call timeout | 12s, 1 retry |
| Idle session cleanup | 1800s — frees locks and fingerprints, **not** checkpoint rows |
| Dedupe memory | 32 fingerprints per call |

Checkpoint rows persist indefinitely. Fine for a demo; anything real needs a retention
job, and transcripts are personal data.

**Scaling:** this is one process, and persistence alone does not solve scaling. The
registry is in-process, so two workers would each keep their own and serialisation would
break. In order: `AsyncPostgresSaver`; move the lock to Redis (the part persistence
doesn't give you); dedupe to Redis; route by call id; add retention. Session isolation
already survives — it's a property of the thread id. What breaks is serialisation
*within* one call.

---

## Results

### Retrieval — 41 labeled questions (36 answerable, 5 off-topic)

Labels name the chunk(s) containing the answer, written by reading `data/chunks.json`,
not by running the retriever.

| k | Hit@k | Recall@k | |
|---|---|---|---|
| 1 | 0.889 | 0.847 | |
| **3** | **0.944** | **0.944** | **the generator only sees these** |
| 5 | 1.000 | 0.991 | diagnostic only |

MRR **0.931**. Dense-only baseline: Hit@1 0.861, MRR 0.912.

- **Hit@k** — did *at least one* relevant chunk reach the top k. This was previously
  mislabelled "Recall@k"; it is not, and the old name flattered the result.
- **Recall@k** — the share of *all* relevant chunks retrieved. Differs on the 4
  questions with more than one valid supporting chunk.
- **Score separation** — answerable mean top-1 cosine **0.406** (min 0.178) vs off-topic
  **0.118** (max 0.173). Sets both thresholds.

Four questions don't rank first. Two of them (`exp-1`, `pub-2`) land at rank 4 — with
`TOP_K=3`, those chunks **never reach the generator**, so the agent cannot currently
answer those two from retrieval.

### Grounding — 31 behavioural cases, 31/31 across three runs

11 answered · 7 missing-detail · 4 unrelated · 4 fabrication · 4 injection · 1
clarification.

**This is pattern matching, not semantic judgement** — no model grades the output. A
refusal is recognised by a regex covering the phrasings this agent produces, and
grounding is asserted with substring checks. Every case carries a `must_not_contain`
list of facts that would only appear if invented, so a case fails on fabrication even
when refusal wording changes. Limits: a refusal phrased outside the regex scores as a
failure even if correct, and a fabrication nobody listed scores as a pass.

### 134 offline tests, no network

Chunking · scope controls (including speech-disfluency regressions from the live call) ·
retrieval · graph routing and state · sessions and concurrency · the Vapi HTTP contract ·
endpoint authorisation and public-surface leakage. Plus `ruff` clean.

### Latency (localhost, excludes speech-to-text and text-to-speech)

Answer 2.1s · with reference resolution 2.4s · router-decided refusal 0.5s ·
**deterministic refusal 0.01s** (no model call — most of why layer 1 exists).

### Verified / not verified

| | |
|---|---|
| **Locally** | 134 tests; retrieval and grounding evals; latency; thread isolation; checkpoint read-back; duplicate suppression; endpoint auth |
| **On live Vapi calls** | Three browser calls, 15 turns. SSE framing accepted in production; `call.id` → thread mapping; memory held across 11 turns; calls stayed isolated; every refusal class fired over voice |
| **Not verified** | Telephony (browser only). Barge-in not deliberately exercised. Concurrent callers not tested with real calls. No load testing |

Two bugs only a live call could find: the CDN bundle double-wraps the SDK's CommonJS
export so the constructor hung silently, and speech-to-text disfluencies (*"previous
previous"*, *"from now, from now"*) defeated the injection patterns. The router still
refused those — which is the argument for layering — and both are fixed with the
verbatim transcriptions as regression tests.

---

## Limitations

1. **Browser calls only.** No phone number purchased; telephony untested.
2. **Thresholds are tuned on the set they're measured on.** 36 answerable questions, no
   holdout. One question moves Hit@1 by ~2.8 points.
3. **Only 5 off-topic examples** behind the score-separation claim.
4. **Two questions can't be answered** — their chunk ranks 4th, outside `TOP_K=3`.
5. **Grounding eval is pattern-based**, not model-graded. See above.
6. **Tense drifts occasionally** — about one phrasing in five says "is pursuing" rather
   than "studied" for the finished degree. Dates are always right.
7. **Reference resolution can pick the wrong kind of thing** — "that project" after an
   education turn resolves to the degree. Grounding turns it into an honest "no
   information" rather than a false claim, but the framing is odd.
8. **Pattern-based scope layers are defeatable by paraphrase.**
9. **Single process**, in-process locks.
10. **No checkpoint retention** — the SQLite file grows forever and holds transcripts.
11. **No PII redaction in logs** — questions are logged truncated at 60 characters.
12. **The access key is a single shared secret**, not per-user auth. It stops a stranger
    who finds the URL; it does not distinguish between people who have it, and there is
    no rotation or revocation beyond changing it and restarting.

---

## Layout

```
app/        config · chunking · embeddings · retriever · scope · prompts · graph · sessions · server
data/       resume.md (knowledge source) · chunks.json (indexed corpus)
eval/       labeled sets and actual measured results
scripts/    verify_source · rebuild · create_assistant · measure_latency
web/        the browser call page
tests/      134 offline tests
```

> The source PDF is gitignored and the contact line in `resume.md` is redacted — both
> carry a real phone number and email, and this repo is public. Nothing depends on them:
> that section is excluded from the index. `verify_source.py` skips its check when the
> PDF is absent; drop any resume PDF there to run it.

### Swapping in a different resume

```bash
cp new_resume.pdf data/resume_source.pdf
# edit data/resume.md to match, keeping the ## / ### structure
python scripts/rebuild.py        # verifies, re-chunks, flags stale eval labels
```

Eval labels name specific chunk ids, so a different resume needs them rewritten by hand.
That's deliberate — auto-generated labels would make the metrics meaningless.

---

## Model choice

**`gpt-4o-mini`** for both generation and routing. Voice is latency-bound — the caller
hears silence while the agent thinks — so a small fast model beats a stronger slow one.
The graph takes anything with `.invoke(messages) -> .content`, so swapping is a config
change. The tense drift and the router's unreliability on project-named questions are
both symptoms of the small model.
