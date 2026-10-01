"""Embedding backends behind one small interface.

Default: OpenAI text-embedding-3-small (1536 dims). Chosen on size, not prestige --
see README. Swapping in a local sentence-transformers model means implementing
`Embedder` and nothing else.

Corpus vectors are cached on disk keyed by (model, text), so rebuilding the index
after a resume edit only re-embeds what changed and the test suite never pays for
an API call twice.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Return an L2-normalised (n, dim) float32 matrix."""
        ...


def _l2_normalise(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # Guard the degenerate all-zero row so a bad embedding can't produce NaNs that
    # silently read as "no match" downstream.
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


class OpenAIEmbedder:
    """text-embedding-3-small. One batched call for the corpus, one per query turn."""

    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 1536):
        from openai import OpenAI

        self._client = OpenAI(api_key=api_key)
        self.name = model
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        response = self._client.embeddings.create(model=self.name, input=list(texts))
        # The API returns results in request order, but it carries an explicit index;
        # sort by it rather than trusting order.
        ordered = sorted(response.data, key=lambda d: d.index)
        return _l2_normalise(np.array([d.embedding for d in ordered], dtype=np.float32))


class HashingEmbedder:
    """Deterministic offline stand-in. Used by tests and by `--offline` eval runs.

    Hashed character 4-grams into a fixed space. Far weaker than a real model -- it has
    no synonym sense at all -- so retrieval numbers from this backend are not
    comparable to the real ones and the eval script labels them as such.
    """

    def __init__(self, dim: int = 512):
        self.name = f"hashing-4gram-{dim}"
        self.dim = dim

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        matrix = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            normalised = " ".join(text.lower().split())
            for i in range(max(len(normalised) - 3, 1)):
                gram = normalised[i : i + 4]
                bucket = int.from_bytes(hashlib.md5(gram.encode()).digest()[:4], "big")
                matrix[row, bucket % self.dim] += 1.0
        return _l2_normalise(matrix)


class CachedEmbedder:
    """Disk-memoised wrapper. Cache key includes the model name, so switching models
    cannot silently serve vectors from the previous one."""

    def __init__(self, inner: Embedder, cache_path: Path):
        self._inner = inner
        self._path = cache_path
        self.name = inner.name
        self.dim = inner.dim
        self._cache: dict[str, list[float]] = {}
        if cache_path.exists():
            stored = json.loads(cache_path.read_text())
            if stored.get("model") == self.name:
                self._cache = stored.get("vectors", {})

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:32]

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        missing = [t for t in texts if self._key(t) not in self._cache]
        if missing:
            fresh = self._inner.embed(missing)
            for text, vector in zip(missing, fresh, strict=True):
                self._cache[self._key(text)] = vector.tolist()
            self._flush()
        return np.array([self._cache[self._key(t)] for t in texts], dtype=np.float32)

    def _flush(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._path.write_text(json.dumps({"model": self.name, "vectors": self._cache}))
