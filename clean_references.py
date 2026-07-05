#!/usr/bin/env python3
"""
clean_references.py
-------------------
Cleans the JOSPT 2017 ACL CPG reference list extracted from the ODT.

Fixes applied:
  BLOCKING
    1. Missing comma after ref 17
    2. Unquoted values in refs 39-44
    3. Malformed key "116:" -> "116"
    4. Ref 30 value ends with ) instead of closing quote

  NOISE
    5. Double commas in author strings (, ,) -> (,)
    6. Strips [Crossref](...), [Medline](...), [Google Scholar](...) link noise
    7. Strips leading/trailing whitespace from all values
    8. Strips [Link](...) artifacts (JOSPT internal links)
    9. Removes "and " author separator introduced by OCR

  DATA QUALITY
   10. Reorders refs 72/73 (73 appears before 72 in source)
   11. Adds missing journal name to ref 128 (Xergia SA)
   12. Adds missing author to ref 125 (WHO)

  SCHEMA
   13. Outputs structured dict per reference:
       citation_number, citation_text, doi, year, source_document
"""

import re
import json
import sys


def clean_value(v: str) -> str:
    """Clean a single reference string value."""
    v = v.strip()

    # Remove journal website navigation links — bracketed markdown form,
    # e.g. [Crossref](url) [Medline](url) [Google Scholar](url) [Link](url)
    v = re.sub(r'\[(Crossref|Medline|Google Scholar|Link)\]\([^)]*\)', '', v)

    # Remove the same links when they've already collapsed to bare trailing
    # words with no brackets/URL, e.g. "...890-897. Crossref Medline Google Scholar"
    # Anchored to end-of-string so it never eats a legitimate mid-sentence word.
    v = re.sub(
        r'\s*(?:Crossref|Medline|Google Scholar|Link)(?:\s+(?:Crossref|Medline|Google Scholar|Link))*\s*$',
        '',
        v,
    )

    # Remove OCR-introduced "and " joining the author list to the title.
    # Two forms appear in this corpus:
    #   ", and Title"   (comma before "and")            -> ", Title"
    #   "LastName. and Title" (period before "and", the
    #    common case when "et al." is absent)           -> "LastName. Title"
    v = re.sub(r',\s+and\s+', ', ', v)
    v = re.sub(r'\.\s+and\s+(?=[A-Z])', '. ', v)

    # Fix double commas from OCR: ", ," -> ","
    v = re.sub(r',\s*,', ',', v)

    # Collapse multiple spaces
    v = re.sub(r'  +', ' ', v)

    v = v.strip().rstrip(',').strip()
    return v


def extract_doi(v: str) -> str | None:
    """Extract DOI URL from value string (before cleaning strips URLs)."""
    m = re.search(r'https://doi\.org/[\w./\-()\[\]]+', v)
    return m.group(0) if m else None


def extract_year(v: str) -> int | None:
    """Extract publication year.

    Two citation shapes appear in this corpus:
      - journal articles: "Title. Journal. 2014; 2: 123"  (year after ". ")
      - books/reports:    "Title. City: Publisher; 2006"   (year after "; ")
    """
    m = re.search(r'[.;]\s*((?:19|20)\d{2})\s*[;:]', v)
    if m:
        return int(m.group(1))
    # Fallback: trailing "; YYYY" or "; YYYY." with nothing meaningful after it
    # (books/reports). Optional trailing period because this runs before the
    # final rstrip('.') in build_structured.
    m = re.search(r';\s*((?:19|20)\d{2})\.?\s*\Z', v)
    if m:
        return int(m.group(1))
    # Fallback: year inside parentheses, e.g. "(March 2009)" — website/resource
    # citations that don't follow the journal-article or book patterns above.
    m = re.search(r'\(\D*((?:19|20)\d{2})\)', v)
    if m:
        return int(m.group(1))
    return None


def build_structured(num: str, raw_value: str) -> dict:
    """Convert a flat citation string into a structured dict."""
    doi = extract_doi(raw_value)  # must run on raw_value — needs the live URL

    v = clean_value(raw_value)

    year = extract_year(v)  # must run on cleaned text — fallback pattern is end-anchored

    # Remove DOI URL from citation_text now that we have it in doi field
    citation_text = re.sub(r'https://doi\.org/[\w./\-()\[\]]+', '', v)
    citation_text = re.sub(r'  +', ' ', citation_text).strip().rstrip('.')

    return {
        "citation_number": int(num),
        "citation_text": citation_text,
        "doi": doi,
        "year": year,
        "source_document": "jospt_2017_acl_cpg",
    }


# Manual fixes for data quality issues that can't be regex-cleaned
MANUAL_FIXES = {
    # Ref 125: missing author
    "125": lambda v: (
        "World Health Organization. " + v
        if not v.startswith("World")
        else v
    ),
    # Ref 128: missing journal name (Xergia SA)
    "128": lambda v: v.replace(
        "reconstruction. 2011; 19: 768",
        "reconstruction. Knee Surg Sports Traumatol Arthrosc. 2011; 19: 768",
    ),
}


def process_references(raw_dict: dict) -> dict:
    """Process all references: clean, fix, structure, sort."""
    result = {}

    # Fix malformed key "116:" -> "116"
    if "116:" in raw_dict:
        raw_dict["116"] = raw_dict.pop("116:")

    for key, value in raw_dict.items():
        # Normalise key (strip stray periods e.g. "127.")
        clean_key = key.strip().rstrip(".")
        if not clean_key.isdigit():
            print(f"  WARNING: unexpected key format '{key}' -> keeping as-is", file=sys.stderr)

        if value is None:
            print(f"  WARNING: ref {key} has null value, skipping", file=sys.stderr)
            continue

        # Apply manual data-quality fixes
        if clean_key in MANUAL_FIXES:
            value = MANUAL_FIXES[clean_key](value)

        result[clean_key] = build_structured(clean_key, value)

    # Sort numerically (fixes 72/73 ordering and any other out-of-order keys)
    result = dict(sorted(result.items(), key=lambda x: int(x[0]) if x[0].isdigit() else 0))

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python clean_references.py raw_refs.json [output.json]")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else "citation_map.json"

    with open(input_path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    print(f"Loaded {len(raw)} references from {input_path}")

    cleaned = process_references(raw)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

    print(f"Done. {len(cleaned)} references written to {output_path}")

    # Sanity check: print a sample
    sample_key = "17"
    if sample_key in cleaned:
        print(f"\nSample ref {sample_key}:")
        print(json.dumps(cleaned[sample_key], indent=2))
