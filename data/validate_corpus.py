"""Validate a corpus file against the frozen Streaming RAG format.

Usage (from the repo root, venv active):
    python validate_corpus.py data/corpus.json

Checks every entry for the five required fields, non-empty values, unique ids,
and text length within the allowed range. Prints a clear list of problems so a
bad merge fails loudly instead of crashing the backend with a vague error.
"""
import json
import sys

REQUIRED = ("id", "title", "parent_title", "source", "text")
MIN_LEN, MAX_LEN = 200, 2000


def main() -> int:
    path = sys.argv[1] if len(sys.argv) > 1 else "data/corpus.json"
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return 1
    except json.JSONDecodeError as e:
        print(f"Not valid JSON: {e}")
        return 1

    if not isinstance(data, list):
        print("Top level is not a JSON array.")
        return 1

    problems = []
    seen_ids = set()
    for i, entry in enumerate(data):
        for field in REQUIRED:
            if field not in entry or not str(entry.get(field, "")).strip():
                problems.append(f"entry {i}: missing or empty '{field}'")
        eid = entry.get("id")
        if eid in seen_ids:
            problems.append(f"entry {i}: duplicate id '{eid}'")
        seen_ids.add(eid)
        text = entry.get("text", "")
        if not (MIN_LEN <= len(text) <= MAX_LEN):
            problems.append(
                f"entry {i} (id={eid}): text length {len(text)} outside {MIN_LEN}-{MAX_LEN}"
            )

    print(f"{len(data)} entries checked in {path}")
    if problems:
        print(f"{len(problems)} problem(s) found:")
        for p in problems[:40]:
            print("  - " + p)
        if len(problems) > 40:
            print(f"  ... and {len(problems) - 40} more")
        return 1
    print("OK: every entry matches the frozen format.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
