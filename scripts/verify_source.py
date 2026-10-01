"""Check data/resume.md faithfully represents data/resume_source.pdf.

Guards the one claim the whole demo rests on: the knowledge source contains nothing
the PDF doesn't. Compares normalised bag-of-words per side, ignoring PDF layout noise
(hyphen line-wraps, sentence punctuation).

Run: python scripts/verify_source.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

STOP = {"the", "a", "an", "and", "or", "of", "in", "on", "to", "with", "for", "at",
        "as", "by", "is", "are", "was", "were", "per", "via", "across", "has", "have"}

# Rewordings deliberately introduced in resume.md, each justified. The verifier
# accepts these and nothing else, so a new unlisted term is a real finding.
ALLOWED_EDITS = {
    # Each entry is a deliberate rewording in resume.md, with the reason. The verifier
    # accepts only these, so any other new term is a real finding to review.
    "roughly", "about",             # "~1,371 requests" -> "about 1,371 requests"
    "plus",                         # "BM25 + Qdrant" -> "BM25 plus Qdrant"
    "reciprocal", "rank", "fusion", # expands the PDF's own "RRF" acronym
    "links", "repository",          # labels for the PDF's [Live Demo] / (GitHub) markers
    "source", "code",               # "(GitHub)" -> "Source code is on GitHub"
    "degree", "date",               # "Degree date May 2026" for the PDF's bare "May 2026"
    "contact", "location", "based",  # section headings; Contact is excluded from retrieval
    "seconds", "request", "hours", "week",  # "/" and "s" units written out for speech
    "3.0",                          # "3.0s/request" -> "3.0 seconds per request"
    "phone", "email", "redacted",   # contact line redacted for the public repo
}

# Contact details are intentionally dropped from resume.md because this repository is
# public. They are excluded from the index either way, so their absence is a deliberate
# redaction, not a coverage gap. Recognised by SHAPE rather than listed literally --
# listing them here would put the very values back into a public file.
_CONTACT_SHAPED = re.compile(
    r"^\d{7,}$"                    # a bare phone number once punctuation is stripped
    r"|^[\w.+-]+@"                  # an email local part
    r"|\.(com|net|org|edu|io)$"     # a mail/domain fragment
)


def is_contact_term(token: str) -> bool:
    return bool(_CONTACT_SHAPED.search(token))


def words(text: str) -> set[str]:
    text = text.lower().replace("—", " ").replace("–", " ")
    text = re.sub(r"-\s*\n\s*", "", text)   # rejoin words the PDF hyphen-wrapped
    text = text.replace("/", " ")           # "8 hours/week" == "8 hours per week"
    text = re.sub(r"\s+", " ", text)
    out: set[str] = set()
    for token in re.findall(r"[a-z0-9][a-z0-9+.&-]*", text):
        token = token.strip(".,;:()[]&-")   # drop sentence punctuation
        # Compare hyphen-insensitively: the PDF wraps long model names mid-hyphen,
        # so "ms-marco-minilm" and "ms-marcominilm" are the same term.
        token = token.replace("-", "")
        if len(token) > 2 and token not in STOP:
            out.add(token)
    return out


def main() -> int:
    from pypdf import PdfReader

    source = ROOT / "data/resume_source.pdf"
    if not source.exists():
        # The source PDF is gitignored: it carries a real phone number and email, and
        # this repository is public. Everything else runs without it -- only this
        # fidelity check needs it. Drop your own resume there to re-run the check.
        print("data/resume_source.pdf not present (gitignored: it contains contact details).")
        print("Skipping the fidelity check. Add the PDF locally to run it.")
        return 0

    pdf_text = "\n".join(p.extract_text() or "" for p in PdfReader(source).pages)
    md_text = re.sub(r"<!--.*?-->", "", (ROOT / "data/resume.md").read_text(), flags=re.S)

    pdf_w, md_w = words(pdf_text), words(md_text)
    missing = sorted(w for w in (pdf_w - md_w) if not is_contact_term(w))
    redacted = sorted(w for w in (pdf_w - md_w) if is_contact_term(w))
    unexplained = sorted(md_w - pdf_w - ALLOWED_EDITS)

    print(f"PDF terms: {len(pdf_w)}   resume.md terms: {len(md_w)}")
    print(f"coverage of PDF by resume.md: {1 - len(missing) / len(pdf_w):.1%}"
          f"   ({len(redacted)} contact terms redacted on purpose)")
    if missing:
        print(f"\nIn PDF, absent from resume.md ({len(missing)}): {', '.join(missing)}")
    print(
        f"\nTerms in resume.md with no PDF source, beyond the {len(ALLOWED_EDITS)} "
        f"declared rewordings: {len(unexplained)}"
    )
    if unexplained:
        print(f"  {', '.join(unexplained)}")
        print("  ^ each is a potential invented fact. Review before demoing.")
        return 1
    print("  none -- every substantive term traces back to the PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
