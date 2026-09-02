# probe_corpus_coverage.py
#
# Calls retriever.retrieve() directly, bypassing the agent entirely.
# This proves the corpus itself is retrievable across all four
# categories, independent of what queries the LLM happens to choose.
#
# Run from repo root:
#   python probe_corpus_coverage.py

from backend.rag.retriever import retrieve

# One hand-picked query per category you want to prove coverage for.
# If a query comes back empty, that's real signal — either the wording
# doesn't match the corpus phrasing, or that content wasn't ingested.
# Try a couple of phrasings before assuming the latter.
QUERIES = {
    "Special-test stub":              "Lachman test ACL",
    "Magee - observation/gait":       "knee observation standing gait assessment",
    "Magee - active/passive ROM":     "active and passive knee range of motion examination",
    "Magee - ligament stability":     "collateral and cruciate ligament stability testing",
    "Magee - applied anatomy":  "knee joint anatomy tibiofemoral structures",
    "ACL CPG (Logerstedt 2017)":      "ACL injury classification criteria",
    "Meniscal CPG (Logerstedt 2018)": "meniscal tear conservative management",
}

for category, query in QUERIES.items():
    print("=" * 60)
    print(category)
    print(f"query: {query!r}")
    print("=" * 60)

    results = retrieve(query)

    if not results:
        print("  NOTHING retrieved above the relevance floor.\n")
        continue

    for r in results:
        source = r["metadata"].get("source", "unknown")
        score = r["score"]
        snippet = r["text"][:150].replace("\n", " ")
        print(f"  source={source:<45} score={score:.3f}")
        print(f"      -> {snippet}")
    print()
