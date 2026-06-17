"""
build_corpus.py
===============
Downloads Wikipedia articles across three topics:
  1. 2024 Paris Olympics
  2. History of music & genres
  3. Space exploration & missions

Splits long articles into 200-300 word chunks with 30-50 word overlap
and saves everything as  data/corpus.json

HOW TO RUN
----------
1. Install the required library (only once):
       pip install wikipedia-api

2. Run the script:
       python build_corpus.py

The file  data/corpus.json  will be created automatically.
"""

import json
import os
import re
import wikipediaapi

# ── Settings ──────────────────────────────────────────────────────────────────
TARGET_CHUNK_WORDS = 250
OVERLAP_WORDS      = 40
MIN_CHUNK_WORDS    = 200
MAX_CHUNK_CHARS    = 1500
MIN_CHUNK_CHARS    = 200   # matches the frozen format and the validator

# ── Articles to download ──────────────────────────────────────────────────────
# Organised by topic so it is easy to add or remove articles later.

ARTICLES = {

    # ── Topic 1: 2024 Paris Olympics ─────────────────────────────────────────
    "2024 Paris Olympics": [
        "2024 Summer Olympics",
        "2024 Summer Olympics opening ceremony",
        "2024 Summer Olympics closing ceremony",
        "2024 Summer Olympics medal table",
        "France at the 2024 Summer Olympics",
        "United States at the 2024 Summer Olympics",
        "Great Britain at the 2024 Summer Olympics",
        "Australia at the 2024 Summer Olympics",
        "China at the 2024 Summer Olympics",
        "Japan at the 2024 Summer Olympics",
        "Kenya at the 2024 Summer Olympics",
        "Jamaica at the 2024 Summer Olympics",
        "Brazil at the 2024 Summer Olympics",
        "Athletics at the 2024 Summer Olympics",
        "Athletics at the 2024 Summer Olympics – Men's 100 metres",
        "Athletics at the 2024 Summer Olympics – Women's 100 metres",
        "Athletics at the 2024 Summer Olympics – Men's marathon",
        "Athletics at the 2024 Summer Olympics – Women's marathon",
        "Swimming at the 2024 Summer Olympics",
        "Swimming at the 2024 Summer Olympics – Men's 100 metre freestyle",
        "Swimming at the 2024 Summer Olympics – Women's 100 metre freestyle",
        "Gymnastics at the 2024 Summer Olympics",
        "Artistic gymnastics at the 2024 Summer Olympics – Women's individual all-around",
        "Artistic gymnastics at the 2024 Summer Olympics – Men's individual all-around",
        "Football at the 2024 Summer Olympics",
        "Basketball at the 2024 Summer Olympics",
        "Volleyball at the 2024 Summer Olympics",
        "Rugby sevens at the 2024 Summer Olympics",
        "Boxing at the 2024 Summer Olympics",
        "Judo at the 2024 Summer Olympics",
        "Tennis at the 2024 Summer Olympics",
        "Cycling at the 2024 Summer Olympics",
        "Rowing at the 2024 Summer Olympics",
        "Skateboarding at the 2024 Summer Olympics",
        "Sport climbing at the 2024 Summer Olympics",
        "Breaking at the 2024 Summer Olympics",
        "Surfing at the 2024 Summer Olympics",
        "Stade de France",
    ],

    # ── Topic 2: History of music & genres ───────────────────────────────────
    "History of music": [
        "History of music",
        "Classical music",
        "Baroque music",
        "Romantic music",
        "Jazz",
        "Blues",
        "Rock music",
        "Rock and roll",
        "Punk rock",
        "Heavy metal music",
        "Electronic music",
        "Hip hop music",
        "Rhythm and blues",
        "Soul music",
        "Reggae",
        "Pop music",
        "Country music",
        "Folk music",
        "Opera",
        "Musical theatre",
        "The Beatles",
        "Michael Jackson",
        "Elvis Presley",
        "David Bowie",
        "Bob Dylan",
        "Beethoven",
        "Wolfgang Amadeus Mozart",
        "Johann Sebastian Bach",
        "Music of Africa",
        "Music of Latin America",
    ],

    # ── Topic 3: Space exploration & missions ────────────────────────────────
    "Space exploration": [
        "Space exploration",
        "History of spaceflight",
        "Apollo program",
        "Apollo 11",
        "Moon landing",
        "International Space Station",
        "Hubble Space Telescope",
        "James Webb Space Telescope",
        "Mars exploration",
        "Mars rovers",
        "Curiosity rover",
        "Perseverance rover",
        "SpaceX",
        "Falcon 9",
        "Starship (spacecraft)",
        "NASA",
        "European Space Agency",
        "Yuri Gagarin",
        "Neil Armstrong",
        "Space Shuttle program",
        "Voyager program",
        "Solar System",
        "Black hole",
        "Milky Way",
        "Exoplanet",
        "Artemis program",
        "Commercial Crew Program",
        "Sputnik 1",
        "Venus",
        "Jupiter",
    ],
}

# ─────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Strip Wikipedia section headings and collapse whitespace.

    Wikipedia's plain text includes lines like '== References ==' and tail
    sections (See also, References, External links) that are noise for
    retrieval. Removing them keeps chunks focused on real content.
    """
    # Cut everything from the first tail section onward.
    for marker in ("\n\nSee also", "\n\nReferences", "\n\nExternal links",
                   "\n\nNotes", "\n\nFurther reading", "\n\nBibliography"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    # Drop heading lines like '== History =='.
    text = re.sub(r"^=+\s*.*?\s*=+$", "", text, flags=re.MULTILINE)
    # Collapse runs of whitespace.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def make_chunks(text: str, parent_title: str, source: str, base_id: str) -> list[dict]:
    """Split an article into chunks that each satisfy the frozen format:
    every text is between MIN_CHUNK_CHARS and MAX_CHUNK_CHARS characters.
    """
    text = clean_text(text)

    # If the whole article is too short to make even one valid chunk, skip it.
    if len(text) < MIN_CHUNK_CHARS:
        return []

    words = text.split()
    raw_chunks: list[str] = []

    if len(words) <= TARGET_CHUNK_WORDS:
        raw_chunks.append(" ".join(words))
    else:
        start = 0
        step = TARGET_CHUNK_WORDS - OVERLAP_WORDS
        while start < len(words):
            end = min(start + TARGET_CHUNK_WORDS, len(words))
            raw_chunks.append(" ".join(words[start:end]))
            if end == len(words):
                break
            start += step

    # Enforce the character window on every chunk.
    fixed: list[str] = []
    for ct in raw_chunks:
        ct = ct.strip()
        if len(ct) > MAX_CHUNK_CHARS:
            ct = ct[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0]  # cut on a word boundary
        if len(ct) < MIN_CHUNK_CHARS:
            # Fold a too-short piece into the previous chunk when possible,
            # otherwise drop it. Never emit a chunk under the floor.
            if fixed:
                merged = (fixed[-1] + " " + ct).strip()
                fixed[-1] = merged[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] if len(merged) > MAX_CHUNK_CHARS else merged
            continue
        fixed.append(ct)

    # Final safety: if merging left the last chunk over the cap, trim it.
    fixed = [c[:MAX_CHUNK_CHARS].rsplit(" ", 1)[0] if len(c) > MAX_CHUNK_CHARS else c for c in fixed]
    fixed = [c for c in fixed if len(c) >= MIN_CHUNK_CHARS]

    chunks = []
    for i, ct in enumerate(fixed, start=1):
        title_suffix = f" (part {i})" if len(fixed) > 1 and i > 1 else ""
        chunks.append({
            "id": f"{base_id}_c{i}",
            "title": parent_title + title_suffix,
            "parent_title": parent_title,
            "source": source,
            "text": ct,
        })
    return chunks


def main():
    print("=== Corpus builder: Olympics + Music + Space ===\n")

    wiki = wikipediaapi.Wikipedia(
        language="en",
        user_agent="CorpusBuilder/1.0 (university-project)"
    )

    corpus: list[dict] = []
    seen_ids: set[str] = set()
    global_index = 1

    for topic, titles in ARTICLES.items():
        print(f"\n📂  Topic: {topic}")
        print("-" * 45)

        for title in titles:
            print(f"  [{global_index:03d}] Fetching: {title}")
            page = wiki.page(title)

            if not page.exists():
                print(f"        ⚠  Not found, skipping.")
                global_index += 1
                continue

            base_id = f"doc_{global_index:03d}"
            chunks  = make_chunks(page.text.strip(), title, page.fullurl, base_id)

            for chunk in chunks:
                if chunk["id"] in seen_ids:
                    chunk["id"] += "_dup"
                seen_ids.add(chunk["id"])
                corpus.append(chunk)

            print(f"        ✓  {len(chunks)} chunk(s)  (total: {len(corpus)})")
            global_index += 1

    # ── Save corpus ───────────────────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)
    output_path = os.path.join("data", "corpus.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(corpus, f, ensure_ascii=False, indent=2)

    print(f"\n✅  Done!  {len(corpus)} entries saved to  {output_path}")

    if len(corpus) < 120:
        print("⚠  Fewer than 120 entries — add more articles.")
    elif len(corpus) < 200:
        print("ℹ  Good for Day-6 checkpoint! Add a few more to reach 200 for the final.")
    else:
        print("🎉  200+ entries — ready for the final hand-in!")

    # ── Save example questions ────────────────────────────────────────────────
    questions = [
        # Olympics
        "Where were the 2024 Summer Olympics held?",
        "Which country topped the 2024 Olympics medal table?",
        "Who won the men's 100 metres at the 2024 Olympics?",
        "Who won the women's gymnastics all-around at Paris 2024?",
        "Which new sport made its debut at the 2024 Olympics?",
        "Who won gold in basketball at the 2024 Olympics?",
        # Music
        "When did jazz originate?",
        "What are the main characteristics of baroque music?",
        "Who invented rock and roll?",
        "What is the origin of reggae music?",
        "Which country does bossa nova come from?",
        "What albums made Michael Jackson famous?",
        # Space
        "Who was the first human to walk on the Moon?",
        "What is the James Webb Space Telescope used for?",
        "How long has the International Space Station been in orbit?",
        "What did the Perseverance rover discover on Mars?",
        "Who founded SpaceX?",
        "What was the first artificial satellite ever launched?",
        # Trick questions (corpus should NOT answer these)
        "Who won the 2026 FIFA World Cup?",
        "What is the population of Mars?",
    ]

    q_path = os.path.join("data", "example_questions.txt")
    with open(q_path, "w", encoding="utf-8") as f:
        f.write("Example questions — Olympics + Music + Space corpus\n")
        f.write("=" * 52 + "\n\n")
        f.write("-- Questions the corpus SHOULD answer --\n\n")
        for i, q in enumerate(questions[:18], 1):
            f.write(f"{i:02d}. {q}\n")
        f.write("\n-- Questions the corpus should NOT answer (no info in corpus) --\n\n")
        for i, q in enumerate(questions[18:], 19):
            f.write(f"{i:02d}. {q}\n")

    print(f"📋  20 example questions saved to  {q_path}")


if __name__ == "__main__":
    main()
