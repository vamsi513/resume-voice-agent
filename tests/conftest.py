"""Shared fixtures.

Unit tests run offline against a scripted LLM so they are deterministic and free.
The live-model tests live in test_grounding_live.py and are skipped without a key.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.chunking import parse_resume  # noqa: E402
from app.embeddings import HashingEmbedder  # noqa: E402
from app.retriever import ResumeRetriever  # noqa: E402

RESUME = ROOT / "data/resume.md"


class ScriptedLLM:
    """Returns queued replies in order, recording every prompt it was sent.

    Scripted rather than mocked-per-call because the graph makes a variable number of
    model calls per turn (the rewrite is conditional), so asserting on call counts
    would couple the tests to routing internals.
    """

    def __init__(self, replies: list[str]):
        self._replies = list(replies)
        self.prompts: list[tuple[str, str]] = []

    def invoke(self, messages, **_):
        self.prompts.append((messages[0]["content"], messages[1]["content"]))
        text = self._replies.pop(0) if self._replies else ""

        class Response:
            content = text

        return Response()


@pytest.fixture(scope="session")
def chunks():
    return parse_resume(RESUME)


@pytest.fixture(scope="session")
def offline_retriever():
    return ResumeRetriever.from_resume(RESUME, HashingEmbedder(), hybrid=True, dense_weight=0.8)


@pytest.fixture
def scripted():
    return ScriptedLLM
