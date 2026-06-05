# StreamingRAG

A live, predictive retrieval system based on RAG: as the user types, the system continuously searches the document collection and, once it is confident enough, answers before the user finishes.

This is a group final project for our Machine Learning course. It builds on our earlier work: a small GPT trained from scratch and a basic Retrieval-Augmented Generation (RAG) prototype.

## What makes it different

Standard RAG is request and response: you type a full question, press Enter, and then it retrieves and answers. This system does the retrieval continuously, on every keystroke, and decides on its own when it has seen enough to answer. The novel part is the timing of the answer, not the retrieval itself.

## How it works

1. **Search as you type.** Each partial query is turned into a vector by an embedding model and compared against the document collection. The most similar documents come back ranked by score. This runs on every keystroke and is fast.
2. **Decide when to answer.** A confidence value is computed from the retrieval scores. While confidence is low the system keeps searching; once it crosses a threshold, it commits.
3. **Generate the answer.** On commit, the committed query and its top documents are passed to a small local language model, which writes an answer grounded in those documents.

The retrieval and the decision use only a small embedding model and simple math, so they are quick and run anywhere. Only the final answer step uses a language model, and it runs once per committed query.

## Project structure

```
streaming_rag/
├── backend/            FastAPI app: retrieval, confidence, generation
├── frontend/           the web interface (served by the backend)
├── data/
│   └── corpus.json     the document collection (chunked articles)
├── evaluation/         tests and tuning for the commit decision
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

The full specification, including the corpus format, the API contract between frontend and backend, and the detailed requirements and definition of done for each work package, is in `streaming_rag_spec.md`. Mock API responses for frontend development without a running backend are in `sample_responses.json`.

## Status

Early development. Setup and run instructions will be added here once the backend skeleton is in place.
