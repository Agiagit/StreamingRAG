# Work Package 2 — Corpus

---

## What this package does

This package builds the document collection that the backend searches through.
It downloads Wikipedia articles across three topic areas, splits long articles into smaller chunks, and saves everything as a single `data/corpus.json` file that the backend (Work Package 1) loads at startup.

**Topics covered:**
- 🏅 2024 Paris Olympics (events, athletes, countries, sports)
- 🎵 History of music and genres (jazz, classical, rock, hip-hop, artists...)
- 🚀 Space exploration (missions, planets, agencies, astronauts...)

---

## Folder structure

```
data/
├── corpus.json            ← the final output (200+ entries)
└── example_questions.txt  ← 20 demo questions for the evaluation team
build_corpus.py            ← the script that generates corpus.json
README.md                  ← this file
```

---

## How to run the script

### 1. Make sure you have Python 3.12

Check your version by running:
```bash
python --version
```

If you need to install it: https://www.python.org/downloads/

### 2. Install the required library

```bash
pip install wikipedia-api
```

Only needs to be done once.

### 3. Run the script

```bash
python build_corpus.py
```

The script will print its progress article by article and tell you how many entries were created. It takes a few minutes to run because it downloads ~98 articles from Wikipedia.

### 4. Check the output

When it finishes you will find:
- `data/corpus.json` — the corpus file ready for the backend
- `data/example_questions.txt` — example questions for Work Package 3

---

## Corpus format

Every entry in `corpus.json` follows the format agreed in section 1.3 of the project spec:

```json
{
  "id":           "doc_001_c1",
  "title":        "2024 Summer Olympics (part 1)",
  "parent_title": "2024 Summer Olympics",
  "source":       "https://en.wikipedia.org/wiki/2024_Summer_Olympics",
  "text":         "...roughly 200–300 words of content..."
}
```

### Field rules
| Field | Description |
|---|---|
| `id` | Unique identifier. Format: `doc_NNN_cN`. No two entries may share an id. |
| `title` | Human-readable name. Chunks of the same article include a part number. |
| `parent_title` | Title of the original article. All chunks of one article share this. |
| `source` | Wikipedia URL of the original article. |
| `text` | The chunk content. Between 200 and 1500 characters (hard limit). |

### Chunking rules (from spec section 1.3)
- Each chunk is **200–300 words**, with a hard ceiling of **1500 characters**
- Neighbouring chunks overlap by **30–50 words** so facts at boundaries are not lost
- Short articles that already fit under the limit become a single entry

---

## Corpus statistics (after running the script)

| Topic | Articles | Approx. entries |
|---|---|---|
| 2024 Paris Olympics | 38 | ~80 |
| History of music | 30 | ~70 |
| Space exploration | 30 | ~70 |
| **Total** | **~98** | **~220** |

> Actual numbers may vary slightly depending on article length on Wikipedia.

---

## Example questions

`data/example_questions.txt` contains 20 questions split into two groups:

- **18 questions the corpus should answer** (used by Work Package 3 for evaluation)
- **2 trick questions the corpus cannot answer** (useful for the live demo)

Sample questions:
- *Where were the 2024 Summer Olympics held?*
- *Who won the men's 100 metres at the 2024 Olympics?*
- *When did jazz originate?*
- *Who founded SpaceX?*
- *Who was the first human to walk on the Moon?*

---

## Adding more articles

To add articles, open `build_corpus.py` and find the `ARTICLES` dictionary near the top. Add the exact Wikipedia article title to the relevant topic list:

```python
"Space exploration": [
    ...
    "Mars",          # ← add a new title here
    "Europa (moon)", # ← and here
],
```

Then run the script again. The file will be regenerated from scratch.

---

## Checking the output manually

After running the script, you can quickly verify the output is valid by running:

```bash
python -c "
import json
with open('data/corpus.json', encoding='utf-8') as f:
    data = json.load(f)
print(f'Total entries: {len(data)}')
ids = [d[\"id\"] for d in data]
print(f'Duplicate ids: {len(ids) - len(set(ids))}')
too_long = [d[\"id\"] for d in data if len(d[\"text\"]) > 1500]
print(f'Entries over 1500 chars: {len(too_long)}')
empty = [d[\"id\"] for d in data if not d[\"text\"].strip()]
print(f'Empty text fields: {len(empty)}')
"
```

A healthy corpus shows 0 duplicates, 0 entries over 1500 chars, and 0 empty fields.

---

## Dependencies

```
wikipedia-api
```

All dependencies are listed in the shared `requirements.txt` at the root of the repository.

---
