# StreamingRAG

A live, predictive retrieval system based on RAG: as the user types, the system continuously searches the document collection and, once it is confident enough, answers before the user finishes.

While standard RAG is request and response - you type a full question, press Enter, and then it retrieves and answers - this system does the retrieval continuously, on every keystroke, and decides on its own when it has seen enough to answer.

## How it works

1. **Search as you type.** Each partial query is turned into a vector by an embedding model and compared against the document collection. The most similar documents come back ranked by score. This runs on every keystroke and is fast.
2. **Decide when to answer.** A confidence value is computed from the retrieval scores. While confidence is low the system keeps searching; once it crosses a threshold, it commits.
3. **Generate the answer.** On commit, the committed query and its top documents are passed to a small local language model, which writes an answer grounded in those documents.

The retrieval and the decision use only a small embedding model and simple math, so they are quick and run anywhere. Only the final answer step uses a language model, and it runs once per committed query.

## Project structure

```
streaming_rag/
├── backend/            FastAPI app: retrieval, confidence, generation
├── data/
│   └── corpus.json     the document collection (chunked articles)
├── requirements.txt    shared Python dependencies
└── README.md
```

## Tech stack

- **Python 3.12**
- **FastAPI** and **uvicorn** for the backend
- **sentence-transformers** with `all-MiniLM-L6-v2` for embeddings and retrieval (runs on CPU)
- **SmolLM2-360M-Instruct** run locally for answer generation (small enough for CPU, no API key, swappable for a larger model later)
- Plain HTML and JavaScript for the frontend, served by the backend so there is no cross-origin setup

## Work packages

The project is split into four parts that can be developed in parallel against a fixed data and API contract:

1. **Backend** the retrieval engine, the confidence and commit logic, the generation, and the two endpoints.
2. **Corpus** the document collection, including splitting longer articles into properly sized chunks.
3. **Evaluation** measuring when the system decides to answer and tuning the thresholds.
4. **Frontend** the page the user types into, showing live results, confidence, and the answer.

## Documentation

The full specification, including the corpus format, the API contract between frontend and backend, and the detailed requirements and definition of done for each work package, is in `sample_and_spec/streaming_rag_spec.md`. Mock API responses for frontend development without a running backend are in `sample_and_spec/sample_responses.json`.

## How to run

Requires Python 3.12.

```
# 1. Create and activate a virtual environment (Windows shown)
python -m venv .venv
.venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Start the server from the repo root
uvicorn backend.main:app
```

The first start downloads the two models from Hugging Face (a few hundred MB); after that everything runs locally and offline. When startup finishes, the log shows which corpus file was loaded and how many entries it contains.

Open `http://127.0.0.1:8000` for the radar visualization: type a question, watch blips appear as articles are retrieved, and see the answer appear automatically when the system commits. The developer debug page is at `http://127.0.0.1:8000/debug.html`.

## Building the corpus

The corpus is built from Wikipedia articles by build_corpus.py, which downloads a set of articles across three topics (2024 Paris Olympics, history of music, and space exploration), splits long articles into overlapping chunks, and writes the result to data/corpus.json in the frozen corpus format. It also writes data/example_questions.txt with sample queries the corpus should and should not be able to answer.

This is a build-time step only. You do not need to run it to run the project; the backend reads the finished data/corpus.json. Run it only when you want to regenerate or extend the corpus.

### Dependency

The script needs the wikipedia-api package, which is not part of the runtime requirements because the backend never imports it. Install it on its own before building:

```bash
uv pip install wikipedia-api
```

Note the name mismatch: the package is installed as wikipedia-api (with a hyphen) but imported in the script as wikipediaapi (one word). This is normal for this package and not a typo.

### Usage

From the project root, with the virtual environment active:

bashuv pip install wikipedia-api      # once, if not already installed
python build_corpus.py            # writes data/corpus.json
python validate_corpus.py data/corpus.json   # confirm the output matches the format

Every entry in the generated corpus has a text field between 200 and 1500 characters, the five required fields (id, title, parent_title, source, text), and a unique id. Run validate_corpus.py after building to confirm; if it reports problems, fix the listed entries before using the corpus with the backend.

## Corpus selection

The backend defaults to `data/corpus.json`. To use a different corpus, set the `CORPUS_PATH` environment variable or edit the default in `backend/retrieval.py`:

```
set CORPUS_PATH=path\to\other_corpus.json   # Windows
uvicorn backend.main:app
```

### Tuning the commit decision

All confidence weights and WAIT/SUGGEST/COMMIT thresholds live in one place: the constants at the top of `backend/confidence.py`.

##Evaluation and Redommended Settings

We evaluated the decision-making logic (WAIT, SUGGEST, COMMIT) of the Streaming RAG system to find the optimal balance between speed and factual accuracy.

Using a custom Python script (run_evaluation.py), we tested the system against various standard and trick questions, comparing two configurations:

    Cautious Setting (Threshold 0.55): Highly accurate, but often waited too long even when the question was complete (too_late).
    Eager Setting (Threshold 0.40): Very fast, but easily confused by complex queries, triggering before the full context was clear (too_early).

Results & Recommendation: The visual data analysis (evaluation_chart.png) shows that neither extreme is ideal. For a fluid yet accurate user experience, we recommend a balanced middle ground: setting COMMIT_TOP1_THRESHOLD to 0.48 and COMMIT_MARGIN_THRESHOLD to 0.06.

## Optional: GPU acceleration (NVIDIA)

The backend runs on CPU by default, which works on any machine. If you have an
NVIDIA GPU, answer generation is much faster with a CUDA build of torch:

1. Get the CUDA index URL for your setup from https://pytorch.org/get-started/locally
2. Reinstall the SAME torch version with the CUDA build, forcing the swap:

       uv pip install torch --index-url <cuda-index-url> --reinstall-package torch

3. Verify:

       python -c "import torch; print(torch.cuda.is_available())"   # expect True
       python -c "import backend.main"                              # expect no error

The startup log will then show the generation model on `cuda`. This is optional
and not required to run or develop the project.
