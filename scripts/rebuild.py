"""Rebuild the knowledge source after the resume changes.

Run this whenever data/resume_source.pdf or data/resume.md is edited:

    python scripts/rebuild.py

It extracts the PDF text for comparison, verifies resume.md against it, re-chunks,
re-embeds, drops the stale embedding cache, and prints the corpus so the result is
reviewable. It does NOT relabel eval/retrieval_set.json -- labels name specific chunk
ids, so a resume with different projects needs those written by hand. The script says
so when it detects chunk ids the eval set does not mention.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main() -> int:
    from app.chunking import parse_resume, write_chunks

    print("1. verifying data/resume.md against data/resume_source.pdf")
    verify = subprocess.run(
        [sys.executable, str(ROOT / "scripts/verify_source.py")], capture_output=True, text=True
    )
    print("   " + verify.stdout.strip().replace("\n", "\n   "))
    if verify.returncode != 0:
        print("\n   Source verification FAILED. Fix resume.md (or declare the rewording")
        print("   in ALLOWED_EDITS) before rebuilding -- the agent must not state a fact")
        print("   the PDF does not contain.")
        return 1

    print("\n2. chunking")
    chunks = parse_resume(ROOT / "data/resume.md")
    write_chunks(chunks, ROOT / "data/chunks.json")
    for c in chunks:
        print(f"   [{len(c.text):5d} ch] {c.chunk_id:<46} {c.citation[:60]}")

    print("\n3. dropping the stale embedding cache")
    cache = ROOT / "data/emb_cache.json"
    if cache.exists():
        cache.unlink()
        print("   removed data/emb_cache.json (it will rebuild on next start)")
    else:
        print("   no cache present")

    print("\n4. checking the eval labels still refer to real chunks")
    eval_path = ROOT / "eval/retrieval_set.json"
    labelled = {
        cid
        for q in json.loads(eval_path.read_text())["questions"]
        for cid in q["relevant_chunks"]
    }
    actual = {c.chunk_id for c in chunks}
    dangling = sorted(labelled - actual)
    unlabelled = sorted(actual - labelled)
    if dangling:
        print(f"   STALE LABELS -> chunks that no longer exist: {dangling}")
    if unlabelled:
        print(f"   UNLABELLED   -> chunks no question covers:  {unlabelled}")
    if not dangling and not unlabelled:
        print("   every labelled chunk exists and every chunk is covered")

    print("\n5. next steps")
    print("   - fix any stale labels above, then: python eval/run_retrieval_eval.py --hybrid --dense-weight 0.8")
    print("   - re-sweep if retrieval looks worse:  python eval/run_retrieval_eval.py --sweep-weight")
    print("   - start the server, then:             python eval/run_grounding_eval.py")
    return 1 if (dangling or unlabelled) else 0


if __name__ == "__main__":
    raise SystemExit(main())
