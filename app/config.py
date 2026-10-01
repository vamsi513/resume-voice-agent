"""Runtime configuration. Everything tunable lives here so the demo has one place to look."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


@dataclass(frozen=True)
class Settings:
    # --- Generation ---------------------------------------------------------
    # gpt-4o-mini: the credential available for this demo, and the lowest-latency
    # competent option on that account. Voice is latency-bound -- the caller hears
    # silence while we think -- so a small fast model beats a stronger slow one here.
    # repr=False: keeps the key out of tracebacks, logs and /health output.
    openai_api_key: str = field(default=os.getenv("OPENAI_API_KEY", ""), repr=False)
    chat_model: str = os.getenv("CHAT_MODEL", "gpt-4o-mini")
    router_model: str = os.getenv("ROUTER_MODEL", "gpt-4o-mini")
    temperature: float = float(os.getenv("TEMPERATURE", "0.2"))
    max_answer_tokens: int = int(os.getenv("MAX_ANSWER_TOKENS", "160"))

    # --- Retrieval ---------------------------------------------------------
    # text-embedding-3-small, 1536 dimensions. Chosen because the corpus is ~15
    # chunks: the whole index is one batched API call at startup and a single
    # ~60ms query embedding per turn, with no extra dependency (a local
    # sentence-transformers model pulls ~2GB of torch for no measurable gain at
    # this size). See README "Embedding model" for the measured comparison.
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
    embedding_dim: int = 1536
    top_k: int = int(os.getenv("TOP_K", "3"))
    # Dense cosine is fused with a small BM25 signal by weighted RRF. 0.8 beat
    # dense-only on every metric (R@1 .690->.759, MRR .816->.863, R@5 .966->1.000);
    # equal weighting regressed education queries. Swept: --sweep-weight.
    hybrid_retrieval: bool = os.getenv("HYBRID_RETRIEVAL", "true").lower() == "true"
    dense_weight: float = float(os.getenv("DENSE_WEIGHT", "0.8"))
    # A deliberately LOW backstop, not the scope decision. On the eval set the weakest
    # answerable question scores 0.178 and the strongest off-topic one 0.173 -- the
    # distributions nearly touch, so no threshold can separate them reliably. 0.12 sits
    # safely under every answerable question (it never blocks a real one) while still
    # rejecting ~60% of off-topic ones for free. The router and the generator's
    # no-evidence sentinel do the actual scope work. See README "Scope enforcement".
    min_similarity: float = float(os.getenv("MIN_SIMILARITY", "0.12"))
    # The other end of the same measurement. Off-topic questions never scored above
    # 0.173 on the eval set, while answerable ones average 0.406. So a score this high
    # is strong evidence the resume really does cover the question -- enough to overrule
    # the router when it calls something out of scope. This exists because the router is
    # a single LLM call and measurably unreliable on questions that name a project
    # without naming Vamsi ("What did TriageTune find?" was misrouted ~50% of the time).
    # The two signals now check each other instead of the router deciding alone.
    high_confidence_similarity: float = float(os.getenv("HIGH_CONFIDENCE_SIMILARITY", "0.35"))

    # --- Conversation / session -------------------------------------------
    # Bound on history replayed into the model. Keeps prompt size and latency flat
    # over a long call; older turns stay in the checkpoint but are not re-sent.
    max_history_turns: int = int(os.getenv("MAX_HISTORY_TURNS", "8"))
    # Sessions idle longer than this are dropped from the in-process lock/dedupe
    # registry. Checkpoint rows survive -- see README "Retention".
    session_idle_seconds: int = int(os.getenv("SESSION_IDLE_SECONDS", "1800"))
    checkpoint_db: str = os.getenv("CHECKPOINT_DB", str(DATA_DIR / "checkpoints.sqlite"))

    # --- Server ------------------------------------------------------------
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    # Optional shared secret. If set, Vapi must send it as `X-Vapi-Secret`.
    server_secret: str = os.getenv("SERVER_SECRET", "")
    debug_panel: bool = os.getenv("DEBUG_PANEL", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

    # --- Vapi (browser demo only) -----------------------------------------
    # Served to the page so a reviewer can open one link and press Start call without
    # pasting anything. The PUBLIC key is designed to be exposed in client code; the
    # private key is never read here and must never be set in this file.
    vapi_public_key: str = os.getenv("VAPI_PUBLIC_KEY", "")
    vapi_assistant_id: str = os.getenv("VAPI_ASSISTANT_ID", "")


settings = Settings()

GREETING = (
    "Hi, I'm Vamsi's portfolio assistant. "
    "You can ask me about his projects, education, skills or experience."
)

# Exact phrasing the agent uses for the two refusal classes. Tests assert on
# meaning rather than these strings, but keeping them here stops them drifting.
MISSING_INFO_REPLY = "The resume doesn't provide that detail."
OUT_OF_SCOPE_REPLY = "I can help with questions about Vamsi's background and projects."
