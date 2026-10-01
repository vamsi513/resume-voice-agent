"""The knowledge source must chunk on the resume's own structure."""
from __future__ import annotations


def test_one_chunk_per_project(chunks):
    projects = [c for c in chunks if c.section == "Projects"]
    names = {c.entry.split(" - ")[0] for c in projects}
    assert names == {"AgentIQ", "IncidentMemory AI", "MCP Tool-Calling Agent", "TriageTune"}
    # One chunk each: a project's bullets must not be split away from its technology
    # list, because "what did he use in that project" needs both together.
    assert len(projects) == 4
    assert all(c.parts == 1 for c in projects)


def test_one_chunk_per_role(chunks):
    roles = [c for c in chunks if c.section == "Experience"]
    assert len(roles) == 3
    assert all(c.parts == 1 for c in roles)


def test_tech_list_and_bullets_stay_together(chunks):
    agentiq = next(c for c in chunks if c.chunk_id == "projects--agentiq--1")
    assert "Technologies:" in agentiq.text
    assert "LangGraph" in agentiq.text and "Kubernetes" in agentiq.text
    assert "0.91 faithfulness" in agentiq.text


def test_entry_heading_is_embedded_with_the_body(chunks):
    """Dates and employer live in the heading; without them date questions can't match."""
    codecasa = next(c for c in chunks if "machine-learning-intern" in c.chunk_id)
    assert "Codecasa" in codecasa.text
    assert "Aug 2023" in codecasa.text and "Jul 2024" in codecasa.text


def test_contact_details_are_not_indexed(chunks):
    """Marked retrievable:false -- the agent must not be able to read these aloud.

    Asserts on SHAPE rather than on the literal phone number and email. That keeps real
    contact details out of this repository, and it is the stronger check: it fails for
    any phone number or address, not just the two that happen to be on this resume.
    """
    import re

    indexed = " ".join(c.text for c in chunks)
    phone = re.compile(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b")
    email = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]+\b")
    assert not phone.search(indexed), f"a phone number reached the index: {phone.search(indexed).group()}"
    assert not email.search(indexed), f"an email reached the index: {email.search(indexed).group()}"
    assert not any(c.section == "Contact" for c in chunks)


def test_summary_is_not_indexed(chunks):
    """Excluded on evidence: it paraphrases every section and stole rank 1 from six
    questions. Recall@1 0.778 -> 0.889, MRR 0.868 -> 0.931."""
    assert not any(c.section == "Summary" for c in chunks)


def test_location_is_indexed(chunks):
    """Unlike phone and email, where he is based is an ordinary professional fact."""
    location = next(c for c in chunks if c.section == "Location")
    assert "San Jose" in location.text


def test_every_chunk_carries_provenance(chunks):
    for c in chunks:
        assert c.citation and c.chunk_id and c.text.strip()
        assert c.section in c.citation


def test_split_chunks_respect_the_size_budget(chunks):
    from app.chunking import MAX_CHARS

    for c in chunks:
        if c.parts > 1:
            assert len(c.text) <= MAX_CHARS * 1.2


def test_oversized_entry_splits_on_line_boundaries():
    """Guards the splitter itself, since no current entry is large enough to trigger it."""
    from app.chunking import MAX_CHARS, _split_oversized

    body = "\n".join(f"- bullet number {i} " + "x" * 120 for i in range(40))
    parts = _split_oversized(body)
    assert len(parts) > 1
    assert all(len(p) <= MAX_CHARS for p in parts)
    # Nothing lost, nothing duplicated, no line cut in half.
    assert sum(len(p.splitlines()) for p in parts) == len(body.splitlines())
