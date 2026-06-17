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

def make_chunks(text: str, parent_title: str, source: str, base_id: str) -> list[dict]:
    words = text.split()

    if len(words) <= MIN_CHUNK_WORDS:
        chunk_text = text.strip()[:MAX_CHUNK_CHARS]
        return [{
            "id": f"{base_id}_c1",
            "title": parent_title,
            "parent_title": parent_title,
            "source": source,
            "text": chunk_text,
        }]

    chunks = []
    start = 0
    chunk_index = 1

    while start < len(words):
        end = min(start + TARGET_CHUNK_WORDS, len(words))
        chunk_text = " ".join(words[start:end]).strip()

        if len(chunk_text) > MAX_CHUNK_CHARS:
            chunk_text = chunk_text[:MAX_CHUNK_CHARS]

        if len(words[start:end]) < MIN_CHUNK_WORDS and chunks:
            prev = chunks[-1]
            merged = prev["text"] + " " + chunk_text
            if len(merged) <= MAX_CHUNK_CHARS:
                prev["text"] = merged
            break

        title_suffix = f" (part {chunk_index})" if chunk_index > 1 else ""
        chunks.append({
            "id": f"{base_id}_c{chunk_index}",
            "title": parent_title + title_suffix,
            "parent_title": parent_title,
            "source": source,
            "text": chunk_text,
        })

        chunk_index += 1
        start += TARGET_CHUNK_WORDS - OVERLAP_WORDS
        if start >= len(words):
            break

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
