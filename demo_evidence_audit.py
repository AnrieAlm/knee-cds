# demo_evidence_audit.py
#
# Independent proof of corpus provenance for the demo/viva.
#
# This reads straight from case.agentLog in MongoDB — the audit trail
# written by orchestrator.py's _extract_retrieved() — NOT from the
# suggestion text the LLM produced. That distinction is the whole point:
# the LLM might fail to cite something it was shown, or might cite the
# wrong tag. agentLog.retrieved records what search_corpus actually
# returned, for every tool call, regardless of what the model did with it.
#
# It groups every retrieved chunk by corpus category (special-test stub /
# Magee / ACL CPG / meniscal CPG) so you can show, live, that all four
# corpus segments are being reached and are distinguishable by source file.
#
# Run from the repo root (same constraint as bench.py, because of the
# backend.* import):
#
#   python demo_evidence_audit.py <case_id>
#   python demo_evidence_audit.py <case_id> --entry -1   # latest agentLog entry only (default)
#   python demo_evidence_audit.py <case_id> --entry all  # every entry ever logged for this case
#
# ---------------------------------------------------------------------
# IMPORTANT — edit CATEGORY_PATTERNS below to match your real corpus
# filenames. I don't have visibility into corpus/guidelines/ (it's
# gitignored), so these patterns are guesses based on the commented
# SOURCE_TAGS placeholders in ingest.py ("logerstedt_acl_2017.md",
# "logerstedt_meniscal_2018.md") and the tracked corpus/magee_notes/
# filenames. Run this once against a case with a varied query history,
# check the "UNCLASSIFIED" bucket at the bottom, and adjust the patterns
# until nothing meaningful lands there.
#
# Also worth checking before the demo: ingest.py's CORPUS_ROOT is
# corpus/guidelines (non-recursive), but the only corpus content tracked
# in the repo lives under corpus/magee_notes/. If your real local corpus
# really is laid out as corpus/guidelines/{*.md, test_stubs/*.md}, this
# is fine and magee_notes/ is just leftover. If not, ingest.py may not be
# picking up the files you think it is — worth a quick `ls` check before
# Wednesday.
# ---------------------------------------------------------------------

import argparse
import re
import sys
from collections import defaultdict

sys.path.insert(0, ".")  # so `python demo_evidence_audit.py` works from repo root

from backend import store  # noqa: E402


# Ordered: first pattern that matches (case-insensitive substring of the
# filename) wins. Anything matching none of these falls into either
# "Special-test stub" (if it looks like a short atomic filename) or
# "UNCLASSIFIED" (if you should go look at it).
CATEGORY_PATTERNS = [
    ("Magee (Orthopedic Physical Assessment)", ["magee"]),
    ("JOSPT ACL CPG 2017 (Logerstedt et al.)", ["acl_2017", "logerstedt_acl", "_acl_"]),
    ("JOSPT Meniscal CPG 2018 (Logerstedt et al.)", ["meniscal_2018", "logerstedt_meniscal", "meniscal"]),
    ("Ottawa / Pittsburgh decision rule stub", ["ottawa", "pittsburgh"]),
]

# Filenames of known special-test stubs seen so far in this project's
# logs (from evidence_retrieval_scope.json). Add to this list as you see
# new ones show up in the UNCLASSIFIED bucket during testing — every stub
# file is one chunk = one test, so this list should converge quickly.
KNOWN_STUB_NAMES = {
    "lachman.md",
    "anterior_drawer.md",
    "pivot_shift.md",
    "varus_stress.md",
    "sag_sign.md",
}

RESULT_LINE_RE = re.compile(
    r"^\[(?P<source>[^\|\]]+?)(?:\s*\|\s*relevance\s*(?P<score>[\d.]+))?\]:\s*(?P<text>.*)$",
    re.DOTALL,
)


def categorize(source_filename: str) -> str:
    name = source_filename.strip().lower()
    for label, needles in CATEGORY_PATTERNS:
        if any(n in name for n in needles):
            return label
    if name in KNOWN_STUB_NAMES:
        return "Special-test stub"
    # Heuristic: stub files tend to be short, single-concept filenames
    # with no CPG/textbook naming convention in them.
    if len(name.replace(".md", "")) <= 20 and "_20" not in name:
        return "Special-test stub (unconfirmed — check against test_stubs/)"
    return "UNCLASSIFIED — check CATEGORY_PATTERNS"


def parse_retrieved_result(result_text: str):
    """One search_corpus call's raw string -> list of (source, score, snippet)."""
    chunks = result_text.split("\n---\n")
    parsed = []
    for chunk in chunks:
        chunk = chunk.strip()
        if not chunk or chunk == "No relevant evidence found.":
            continue
        m = RESULT_LINE_RE.match(chunk)
        if not m:
            parsed.append(("(unparsed)", None, chunk[:120]))
            continue
        source = m.group("source").strip()
        score = m.group("score")
        text = m.group("text").strip()
        parsed.append((source, float(score) if score else None, text[:160]))
    return parsed


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("case_id", help="Case id (as stored in Mongo / used in URLs)")
    ap.add_argument(
        "--entry", default="-1",
        help="Which agentLog entry to inspect: -1 (latest, default) or 'all'",
    )
    args = ap.parse_args()

    log = store.get_agent_log(args.case_id)
    if not log:
        print(f"No agentLog entries found for case {args.case_id}.")
        print("Run /suggest or /suggest-async on this case first.")
        sys.exit(1)

    entries = log if args.entry == "all" else [log[int(args.entry)]]

    for i, entry in enumerate(entries):
        print("=" * 70)
        print(f"agentLog entry  (logged_at: {entry.get('logged_at', '?')})")
        print("=" * 70)

        by_category = defaultdict(list)
        retrieved = entry.get("retrieved", [])

        if not retrieved:
            print("  (empty — this entry predates _extract_retrieved(), "
                  "or the agent made no tool calls)")
            continue

        for call in retrieved:
            query = call.get("query", "(no query)")
            result_text = call.get("result", "")
            for source, score, snippet in parse_retrieved_result(result_text):
                category = categorize(source)
                by_category[category].append((source, score, query, snippet))

        # Tally line first — the headline number for the slide/screenshot
        print()
        print("Coverage by corpus category:")
        for category in sorted(by_category, key=lambda c: -len(by_category[c])):
            print(f"  {len(by_category[category]):3d} chunks  -  {category}")
        print()

        # Full detail, grouped
        for category, items in sorted(by_category.items()):
            print(f"--- {category} " + "-" * max(1, 50 - len(category)))
            for source, score, query, snippet in items:
                score_str = f"{score:.3f}" if score is not None else "n/a"
                print(f"  source={source:<45} score={score_str:<6} query={query!r}")
                print(f"      \u2192 {snippet}")
            print()


if __name__ == "__main__":
    main()
