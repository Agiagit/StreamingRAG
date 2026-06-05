# Streaming RAG: Project Specification and Work Packages

A live, predictive retrieval system: as the user types, the system continuously searches the document collection and, once it is confident enough, answers before the user finishes. Typing more interrupts the answer and starts the search again.

This document defines what each of the four of us builds, what we must agree on before we split up, and what "done" looks like. The goal is that we can work in parallel for most of the two weeks and have the parts fit together at the end.

---

## How to read this document

Two kinds of statements appear throughout:

- **Requirement (must):** a rule that has to be met or the parts will not fit together. These are not optional.
- **Suggestion:** a recommended way to do it. How you implement it inside your own package is your choice, as long as the requirements are met.

The idea is to set clear boundaries between the packages while leaving everyone free to solve their own part their own way.

---

# Part 1: Shared foundations (agree on these on day 1)

This is the most important section. If we all agree on these before we start coding, the four parts will combine cleanly. If we skip this, we will waste the last three days trying to glue mismatched pieces together.

## 1.1 Technology and versions

**Requirements:**

- Python 3.12 for everyone. This is the version we use: it has full, stable support across torch, sentence-transformers, transformers, and fastapi on Windows, where newer versions like 3.14 still have gaps in the machine learning library wheels.
- The embedding model is `all-MiniLM-L6-v2` (the same one from our RAG project). This is fixed because the corpus and the backend both depend on it.
- The answer-generating model is `SmolLM2-360M-Instruct`, run locally (the same one from our RAG project). It is small enough to run on CPU, which keeps the backend portable across machines without needing a GPU or any API key. Because the model sits behind the `/answer` endpoint, it can be swapped for a stronger model later without changing anything else.
- Every package keeps its dependencies listed in a shared `requirements.txt` so anyone can recreate the environment.

**Suggestion:** each person works in their own virtual environment but installs from the same `requirements.txt` so we are all on the same library versions.

## 1.2 Repository structure

**Requirement:** we use one shared repository with this folder layout, so each person owns a clear area and we do not overwrite each other's work.

```
streaming_rag/
├── backend/            ← Work Package 1 owns this
│   └── main.py and supporting files
├── frontend/           ← Work Package 4 owns this
│   └── index.html and supporting files
├── data/               ← Work Package 2 owns this
│   └── corpus.json
├── evaluation/         ← Work Package 3 owns this
│   ├── test_queries.json
│   └── results (spreadsheet)
├── requirements.txt    ← shared
└── README.md           ← shared
```

Because each person works mostly inside their own folder, we rarely edit the same file, which keeps version control simple.

## 1.3 The corpus format (the data contract for the corpus)

This is the exact shape of the document collection. The backend reads this file, so the format is fixed.

This is the exact shape of the document collection. The backend reads this file, so the format is fixed.

**We chunk long articles.** The embedding model can only read about the first 380 words of any text, and it compresses whatever it reads into a single vector, so a long article would be both partly ignored and blurry. To avoid this, long articles are split into smaller pieces called chunks, and each chunk becomes its own entry in the corpus. The backend does not change at all; it simply treats each chunk as a normal entry. All of the chunking work happens here, during corpus preparation.

**Requirement:** the corpus is a single file `data/corpus.json`, encoded in UTF-8, containing a JSON array of entry objects. Each object has exactly these five fields, and none of them may be empty:

```json
[
  {
    "id": "doc_042_c2",
    "title": "Athletics at the 2024 Olympics (part 2)",
    "parent_title": "Athletics at the 2024 Olympics",
    "source": "https://en.wikipedia.org/wiki/Athletics_at_the_2024_Summer_Olympics",
    "text": "the second roughly 300-word slice of the article..."
  },
  {
    "id": "doc_043_c1",
    "title": "Swimming at the 2024 Olympics (part 1)",
    "parent_title": "Swimming at the 2024 Olympics",
    "source": "...",
    "text": "..."
  }
]
```

**Field rules (all required):**

- `id`: a unique identifier. For a chunk, use the parent's id plus a chunk number, like `doc_042_c1`, `doc_042_c2`. No two entries may share an id.
- `title`: a short human-readable name for the entry. For a chunk, include which part it is, like `Athletics at the 2024 Olympics (part 2)`.
- `parent_title`: the title of the original article this entry came from. If several chunks come from one article, they all share the same `parent_title`. For a short article that was not split, set `parent_title` to the same value as `title`.
- `source`: where the text came from (a URL or a reference). All chunks of one article share the same source.
- `text`: the chunk content. See the length rule below.

**The text length rule (important):** each `text` must be between **200 and 1500 characters** (roughly 30 to 250 words), with 2000 characters as a hard ceiling that must never be exceeded. This is the reason chunking exists: a 1000-word article does not fit, so it is split into three or four entries that each fit comfortably and stay focused.

**How to chunk (the corpus person owns this):**

- Aim for **200 to 300 words per chunk**.
- Let neighboring chunks **overlap by about 30 to 50 words**. This matters because a fact sitting right on a chunk boundary would otherwise be cut in half and matched poorly; the overlap keeps it whole in at least one chunk.
- Split on **paragraph or sentence boundaries**, never in the middle of a sentence.
- A short article that already fits under the limit does not need splitting. It becomes a single entry whose `parent_title` equals its `title`.

## 1.4 The API contract (the frontend and backend fit together here)

An API is simply the set of messages the frontend sends to the backend and the replies it gets back. This is the single most important agreement in the project, because it is where the frontend and backend meet. Once this is fixed, the frontend person and the backend person can work completely independently.

The backend provides two endpoints.

**Endpoint 1: search as you type.** The frontend calls this repeatedly while the user types.

Request (frontend sends this):

```json
POST /retrieve
{ "text": "who won the 100m", "k": 5 }
```

Response (backend sends this back):

```json
{
  "query": "who won the 100m",
  "results": [
    { "doc_id": "doc_042_c2", "title": "Athletics at the 2024 Olympics (part 2)", "parent_title": "Athletics at the 2024 Olympics", "score": 0.83, "snippet": "first 200 characters of the chunk..." },
    { "doc_id": "doc_017_c1", "title": "...", "parent_title": "...", "score": 0.71, "snippet": "..." }
  ],
  "top1_score": 0.83,
  "top2_score": 0.71,
  "confidence": 0.78,
  "decision": "WAIT"
}
```

Each result now carries a `parent_title` alongside its own `title`, because results are chunks and several chunks can come from the same article. The frontend uses `parent_title` to show one article name even when two of its chunks are retrieved (see Work Package 4).

The `decision` field is always one of three words: `"WAIT"`, `"SUGGEST"`, or `"COMMIT"`. The backend decides this, not the frontend. This keeps the frontend simple: it just reacts to whatever decision it is told.

**Endpoint 2: get the answer.** The frontend calls this once, when it is time to answer (when the decision becomes `COMMIT`, or when the user presses Enter).

Request:

```json
POST /answer
{ "query": "who won the 100m", "doc_ids": ["doc_042_c2", "doc_017_c1"] }
```

Response:

```json
{
  "answer": "The men's 100m at the 2024 Olympics was won by...",
  "sources": [ { "doc_id": "doc_042_c2", "title": "Athletics at the 2024 Olympics" } ]
}
```

The `doc_ids` sent here are chunk ids. When building the answer, the backend may optionally gather the other chunks that share the same `parent_title`, so the model sees the fuller article rather than a single slice. That is a refinement the backend owner can add later; it does not change this contract.

**Requirements:**

- These two endpoints, with exactly these field names, are fixed. Neither side may rename a field without telling the other.
- The frontend must work using only these two messages. It never needs to know how retrieval or generation works inside.

## 1.5 Sample data so the frontend can start immediately

So the frontend person does not have to wait for the real backend, here is a small mock the frontend can use from day one. The frontend person saves these example replies and builds a tiny fake backend (or just hardcodes them) that returns them. When the real backend is ready, nothing in the frontend needs to change.

**Requirement:** the frontend is developed against these sample replies first, so frontend and backend progress in parallel.

The sample replies are exactly the JSON shown in section 1.4. The frontend person should prepare three versions to test the three states: one reply with `"decision": "WAIT"`, one with `"decision": "SUGGEST"`, and one with `"decision": "COMMIT"`, so the interface can be built and tested for all three cases without a real backend.

## 1.6 Shared vocabulary

So we all mean the same thing:

- **Confidence:** a number between 0 and 1 saying how sure the system is that it has found the right documents.
- **WAIT:** not confident yet, keep showing search results but do not answer.
- **SUGGEST:** fairly confident, may show a tentative hint, but not a full answer yet.
- **COMMIT:** confident enough to generate and show a full answer.
- **Interrupt:** the user types more after an answer started, which cancels the answer and goes back to searching.

## 1.7 One practical gotcha to decide on day 1

If the frontend and backend run as two separate web servers during development, browsers block the connection between them by default (this is called a CORS restriction). Two ways to avoid losing time on this:

- **Simplest:** the backend serves the frontend file directly (the same way our prototype already does), so they share one address and there is no blocking.
- **Or:** the backend explicitly allows the connection with a few lines of configuration.

Decide which on day one. The first option is recommended because it is one less thing to configure.

---

# Part 2: The four work packages

Each package below lists who owns it, the goal in plain language, the requirements that must be met, suggestions for how to approach it, what it needs from others, and what "done" means at the day-6 checkpoint and at the final hand-in.

---

## Work Package 1: The Backend (the system)

**Owner:** Stefan (the only one currently familiar with this stack).

**Goal in plain language:** build the engine. It takes the text the user has typed so far, finds the most relevant documents, decides how confident it is, and when confident enough, produces an answer. It offers this to the frontend through the two endpoints in section 1.4.

**Requirements (must be met):**

- Implements both endpoints from section 1.4 exactly as specified.
- Loads `data/corpus.json` once at startup and embeds all entries once, not on every request. Entries are chunks, but the backend treats each one as a normal unit; no chunking logic is needed in the backend.
- `/retrieve` returns the top `k` entries with their similarity scores and `parent_title`, plus `top1_score`, `top2_score`, `confidence`, and a `decision` of `WAIT`, `SUGGEST`, or `COMMIT`.
- `/answer` runs the local `SmolLM2-360M-Instruct` model and returns an answer that uses only the provided entries, and says it does not know if they do not contain the answer.
- The model loads once at startup, not on every request, the same way the corpus is embedded once.
- A single `/retrieve` call returns fast enough to feel live (target under 150 milliseconds).

**Suggested implementation:**

- Start from the existing prototype, which already does the retrieval and a rough confidence number.
- Pass each entry's `parent_title` straight through into the results so the frontend can collapse chunks of the same article.
- A simple starting rule for the decision, to be tuned later by Work Package 3: COMMIT if `top1_score` is at least 0.55 and the gap to the second result is at least 0.08; SUGGEST if `top1_score` is at least 0.45; otherwise WAIT. These numbers are placeholders that Work Package 3 will refine.
- Keep the confidence and decision logic in one clearly named place, so the tuning person knows exactly where the numbers live.
- Optional refinement for `/answer`: gather the other chunks sharing the same `parent_title` as a retrieved chunk, so the model sees the fuller article instead of a single slice. Add this only after the basic version works.

**Consumes from others:** the corpus from Work Package 2 (uses a small sample corpus until the real one is ready). Tuning values from Work Package 3 later.

**Provides to others:** the two endpoints, which Work Package 4 builds against, and the running system, which Work Package 3 evaluates.

**Definition of done, day 6 (checkpoint):**

- Both endpoints run locally and return correctly shaped replies.
- `/retrieve` produces real results and a real decision from a sample corpus.
- `/answer` produces a real answer from the local SmolLM2 model.
- It does not need to be tuned, pretty, or deployed.

**Definition of done, final:**

- The decision logic has a conservative and an aggressive setting that can be switched.
- The thresholds have been tuned using the data from Work Package 3.
- The real corpus from Work Package 2 is loaded.
- The frontend connects to it and the whole loop works end to end.

---

## Work Package 2: The Corpus

**Owner:** a teammate. Needs basic care and organization, no special tech stack knowledge.

**Goal in plain language:** build the collection of documents the system searches, and split any long articles into properly sized chunks. The quality of this collection decides whether the live search looks impressive or broken, so this is a genuinely important job, not busywork. The chunking is the core technical task here, and it is accessible: it is careful text preparation, not coding.

**Requirements (must be met):**

- Produces `data/corpus.json` in exactly the format defined in section 1.3, including the `parent_title` field on every entry.
- Long articles are split into chunks following the chunking rules in section 1.3 (200 to 300 words per chunk, 30 to 50 words of overlap, split on sentence or paragraph boundaries).
- Every entry obeys the field rules and the length rule (200 to 1500 characters of text).
- All ids are unique, and chunks from one article share the same `parent_title` and `source`. No empty fields. Valid JSON that opens without errors.
- Target **200 entries**, with an absolute minimum of **120** for the project to feel substantial. These entries can be whole short articles or chunks of longer ones; what matters for the search is the number of entries.
- The articles share one coherent topic area (for example the 2024 Olympics) so that searching is meaningful, but contain enough internal variety that different questions pull up different entries.

**Suggested implementation:**

- Wikipedia articles are a good source. Longer articles get chunked; short ones can stay whole.
- The splitting can be done with a small script (split on paragraphs, then group paragraphs up to roughly 300 words, carrying a little overlap) or carefully by hand for a smaller corpus. Either is fine as long as the rules are met.
- Check a few entries by hand to confirm the text is clean (no leftover formatting junk, no chunks that are mostly empty, no sentences cut in half at the start or end).
- Optional but valuable: for each article, think of one example question it should answer, and keep a list. This helps Work Package 3 build its test queries and helps the demo.

**Consumes from others:** only the agreed format from section 1.3. Otherwise fully independent and can start on day 1.

**Provides to others:** the corpus file that Work Package 1 loads, and example questions that help Work Package 3.

**Definition of done, day 6 (checkpoint):**

- A first version of `corpus.json` with at least 120 valid entries in the correct format, with long articles already chunked and `parent_title` filled in.
- A short list of about 15 example questions the corpus should be able to answer.

**Definition of done, final:**

- A cleaned, checked corpus of around 200 entries, no duplicate ids, all within the length limit, all chunks correctly tagged with their `parent_title`.
- A curated list of demo questions, grouped by what each one shows (one that the system locks onto quickly, one that stays unclear until the last word, one that the system should refuse because the answer is not in the corpus).

---

## Work Package 3: Evaluation and Tuning

**Owner:** a teammate. Needs basic spreadsheet or basic Python skills and clear thinking, no special tech stack knowledge.

**Goal in plain language:** measure how well the system decides when to answer, and find the threshold numbers that make it behave well. This produces the results and charts that turn a nice demo into a credible project, which is exactly what the professor will look for.

**Requirements (must be met):**

- Defines a written rule for judging a decision: what counts as answering at the right moment, what counts as answering too early (before the question was clear), and what counts as answering too late.
- Tests the system against a fixed set of questions and records, for each one, at what point during typing the system committed and whether that commit was correct.
- Compares at least two settings of the thresholds (a cautious one and an eager one) and reports the difference.
- Delivers recommended threshold numbers back to Work Package 1.

**Suggested implementation:**

- Keep results in a simple spreadsheet: one row per test question, columns for the typed text at the moment of commit, whether it was correct, and the confidence at that point.
- Use the example questions from Work Package 2 plus some deliberately tricky ones.
- A couple of simple charts at the end (for example, how often each setting answered too early) are enough to tell the story.

**Consumes from others:** the test queries from Work Package 2, and a running backend from Work Package 1 to test against. The rubric and the question list can be prepared before the backend is ready.

**Provides to others:** the recommended threshold values for Work Package 1, and the evaluation results for the presentation.

**Definition of done, day 6 (checkpoint):**

- The judging rule is written down clearly.
- At least 10 example evaluations have been done by hand against the rough backend, enough to show the method works.

**Definition of done, final:**

- A full results table across all demo questions.
- A comparison of the cautious and eager settings with a short conclusion.
- One or two simple charts.
- A clear recommendation of which threshold values to use, handed to Work Package 1.

---

## Work Package 4: The Frontend (the interface)

**Owner:** a teammate. Needs basic comfort with editing a web page; the logic is kept deliberately small and a clear spec plus the sample data make it approachable.

**Goal in plain language:** build the page the user actually sees and types into. As the user types, it shows the live search results and a confidence indicator, and when the system decides to answer, it shows the answer. Typing again interrupts the answer.

**Requirements (must be met):**

- A web page with a text input. As the user types, it calls `/retrieve` and displays the returned documents and the confidence value.
- It waits briefly after each keystroke before calling (debouncing), so it does not fire on every single letter.
- A newer reply must never be overwritten by a slower older reply (cancel or ignore outdated replies).
- When the reply's `decision` is `COMMIT`, it calls `/answer` and shows the answer, marked clearly as the system's response, with the sources.
- If the user types more after an answer has appeared, the answer is cleared and the page goes back to showing live search results.
- When two or more results share the same `parent_title` (they are chunks of the same article), the displayed list shows that article once rather than repeating the title. This keeps the results readable.
- It works using only the two endpoints in section 1.4 and nothing else.

**Suggested implementation:**

- Plain HTML with plain JavaScript, no framework and no build step, is the lowest-friction path and matches the prototype. A framework is allowed if the owner prefers one, but it is not needed.
- Build against the sample replies from section 1.5 first, so you do not depend on the backend being ready. Prepare the three reply versions (WAIT, SUGGEST, COMMIT) to test all three on-screen states.
- For collapsing chunks, group the results by `parent_title` before drawing them, and show each article once (you can keep the highest score among its chunks).
- Show confidence as something visual, for example a bar that fills up, rather than just a number.

**Consumes from others:** only the API contract from section 1.4 and the sample data from section 1.5. Independent after day 1.

**Provides to others:** the interface that everything is demonstrated through.

**Definition of done, day 6 (checkpoint):**

- A page where typing triggers live calls and displays the returned documents and a confidence indicator.
- It correctly reacts to all three decisions using the sample data, even if not yet connected to the real backend.

**Definition of done, final:**

- Connected to the real backend.
- The full flow works on screen: typing shows live results, the answer appears on commit, and typing again interrupts and returns to searching.
- The page looks clean and is presentable.

---

# Part 3: Day-1 checklist and timeline

## Agree on all of this on day 1, together, before splitting up

- [ ] The Python version everyone uses.
- [ ] The repository is created with the folder structure from section 1.2.
- [ ] The corpus format from section 1.3 is confirmed.
- [ ] The API contract from section 1.4 is confirmed and the sample replies from section 1.5 are written down and shared.
- [ ] The CORS decision from section 1.7 is made.
- [ ] Each person knows which package they own.

Once this checklist is complete, everyone can work in their own folder largely without blocking each other.

## Day 6: the checkpoint

By the halfway session we want a rough but working prototype plus visible progress from each person. The targets are the day-6 definitions of done above. In the session, each person shows their own piece: the backend owner demonstrates the live search and a generated answer, the corpus owner shows the document collection and demo questions, the evaluation owner shows the judging rule and first measurements, and the frontend owner shows the interface reacting to the three states.

## Days 7 to 12: integration and polish

A safe order for combining the parts, lowest risk first:

1. Load the real corpus into the backend (day 7 or 8). This is the easiest merge.
2. Connect the frontend to the real backend in place of the sample data.
3. Run the full evaluation against the integrated system and feed the recommended thresholds back into the backend.
4. Apply the tuned thresholds and polish the interface.
5. Reserve the final day or two for rehearsing the demo and writing the report, not for building new features.

## The one rule that protects the deadline

The project must always be demonstrable without its fanciest parts. The core loop is: type, see live results, get an answer, interrupt by typing more. Anything beyond that (the cautious-versus-eager comparison, visual polish, any later additions) is a bonus. If a bonus feature is fighting you in the last days, cut it and present a clean working core. A simple system that works always beats an ambitious one that half-works in a live demo.
