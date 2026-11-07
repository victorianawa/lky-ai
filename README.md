# Ask LKY

> Explore leadership, history, economics, and geopolitics.

Ask LKY is a Retrieval-Augmented Generation (RAG) chatbot built for the **"What Would Lee Kuan Yew Do?"** challenge. It answers questions in a pragmatic, direct voice inspired by the documented views of Lee Kuan Yew, grounded in passages retrieved from a curated document corpus.

**It is an AI historical simulation, not the real Lee Kuan Yew.** It has no personal memories or private thoughts, and for events after his death in 2015 it labels answers as hypothetical extrapolation.

- Demo: _add Streamlit Community Cloud URL here_
- Screenshots: _add to `screenshots/` and link here_

## Features

- Multi-turn chat, follow-up-aware retrieval, clear-conversation button
- Custom RAG pipeline (no LangChain): extraction, cleaning, chunking, embedding, ChromaDB, retrieval, prompt assembly
- Top-3 retrieved passages shown under every answer (file, page, category, chunk ID, text); sources come from retrieval metadata, never from model output
- Persistent AI-simulation disclaimer; weak-retrieval warning
- Persona prompt with no-invented-quotes, no-personal-memories and post-2015 hypothetical rules
- Evaluation script (retrieval, hit rate, support, citations, abstention)
- Add documents later by dropping files into `data/` and re-running ingestion

## Tech stack

Python · Streamlit · Google Gemini via `google-genai` · Sentence Transformers (`all-MiniLM-L6-v2`) · ChromaDB · PyPDF · python-dotenv · Git/GitHub · Streamlit Community Cloud

## Architecture and RAG workflow

```
data/ (PDF/TXT/MD) → extract text (per page) → clean → chunk (~900 chars, 150 overlap)
  → embed (MiniLM, normalised) → ChromaDB (cosine)
                                     ▲
User question → (+ previous user turn if short follow-up) → embed → top-3 chunks
  → prompt (persona rules + numbered context + recent history) → Gemini → answer + sources → UI
```

1. **Ingestion** (`ingest.py`): walks `data/`, extracts text per page, skips unreadable files with a warning.
2. **Cleaning**: rejoins hyphenated line breaks, removes bare page numbers and repeated headers/footers, drops near-empty pages (flags likely scanned PDFs).
3. **Chunking**: sentence-aware, ~900 characters with ~150 overlap, within a page so page metadata stays accurate. MiniLM truncates around 256 tokens, so larger chunks would lose text.
4. **Storage**: persistent ChromaDB collection (cosine). Chunk IDs are deterministic (`file::p3::c2`), so re-ingesting upserts rather than duplicates.
5. **Retrieval** (`chatbot.py`): embeds the query, returns top 3 with distances. If the best distance exceeds `WEAK_DISTANCE` (default 0.65, an untuned starting value), the model is told the evidence is weak.
6. **Generation**: Gemini with a persona/grounding system prompt, retries with backoff on 429/5xx.
7. **Hallucination mitigation**: grounding rules, no-quote rule, weak-evidence branch, metadata-driven citations, evaluation checks.

## Project structure

| Path | Responsibility |
|---|---|
| `app.py` | Streamlit UI, session state, disclaimer, source expanders, builds the index on first run |
| `ingest.py` | Document loading, cleaning, chunking, embedding, ChromaDB storage (CLI and importable) |
| `chatbot.py` | Retrieval, prompt construction, Gemini call, error handling |
| `evaluate.py` | Test questions, metrics, writes `evaluation_results.md/.json` |
| `check_model.py` | Lists Gemini models available to your key and tests `GEMINI_MODEL` |
| `data/<category>/` | `speeches`, `interviews`, `articles`, `memoir_excerpts`, `historical_records` (category = folder name) |
| `data/sources.csv` | Optional manifest: `filename,title,url,notes`; doubles as a provenance/permission log |
| `chroma_db/` | Generated index (gitignored) |

## Installation

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # then edit .env
```

Python 3.11–3.13 was used for development (tested on 3.13).

## Environment configuration

```
GEMINI_API_KEY=your_api_key        # from Google AI Studio
GEMINI_MODEL=<model id>            # pick one from check_model.py output
```

Never commit `.env`. Verify the model is available to your key before running the app:

```bash
python check_model.py
```

Model names change over time, so none is hardcoded.

## Adding documents and ingesting

Place only documents you are permitted to use in the category folders, and log each in `data/sources.csv`. Copyrighted books (e.g. memoirs) must not be redistributed in a public repo; keep them local or use short permitted excerpts.

```bash
python ingest.py            # incremental
python ingest.py --rebuild  # wipe and rebuild the index
```

## Running

```bash
streamlit run app.py
```

## Evaluation

```bash
python evaluate.py --retrieval-only   # no Gemini calls
python evaluate.py                    # full run
```

Edit `expected_sources` and `keywords` in `evaluate.py` to match the documents in your corpus before running. Questions without expected sources are reported as n/a for source hit rate rather than guessed.

Metrics (10 in-corpus questions across history, economics, leadership, geopolitics, education, governance, technology, plus 2 out-of-scope):

1. Retrieval relevance: mean similarity of top-3 chunks
2. Source hit rate and keyword hit rate
3. Answer support: Gemini-as-judge (an LLM judgment, not ground truth)
4. Citation correctness: every `[n]` refers to a retrieved passage
5. Abstention: out-of-scope questions acknowledge insufficient evidence (regex check; imperfect)

Results are written to `evaluation_results.md`. _Paste the real results here after running._

## Deployment (Streamlit Community Cloud)

```bash
git init
git add .
git commit -m "Ask LKY"
git branch -M main
git remote add origin https://github.com/victorianawa/lky-ai.git
git push -u origin main
```

1. On share.streamlit.io, choose **Create app**, select the repo, branch `main`, main file `app.py`.
2. Under **Advanced settings**, choose Python 3.11 or 3.12 and paste secrets:
   ```toml
   GEMINI_API_KEY = "your_key"
   GEMINI_MODEL = "your_model_id"
   ```
3. Deploy. The Cloud filesystem is ephemeral, so `chroma_db/` is not committed; `app.py` rebuilds the index from the committed `data/` on first start (cached afterwards). This requires your permitted documents to be in the repo.

**Troubleshooting**
- *Out of memory / slow install*: `torch` is heavy. Retry the deploy, or pin a CPU-only wheel in `requirements.txt`.
- *sqlite3 version error from ChromaDB*: add `pysqlite3-binary` to `requirements.txt` and put `__import__("pysqlite3"); import sys; sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")` at the top of `app.py`.
- *404 model error*: re-run `check_model.py` and update `GEMINI_MODEL`.
- *429 errors*: free-tier quota; wait and retry.
- *Empty knowledge base*: no files under `data/` in the repo.

## Limitations

- Quality depends entirely on the corpus; a small corpus gives thin answers.
- PDFs without a text layer are skipped (no OCR).
- Retrieval is vector-only top-3 with a heuristic distance threshold; follow-up handling is a simple query concatenation.
- The persona is a model's interpretation; it can still be wrong or overstate the sources.
- The evaluation set is small and the LLM judge is imperfect.

## Future improvements

Hybrid BM25 + vector search, reranking, LLM query rewriting, streaming responses, OCR, hosted vector DB, larger corpus, conversation export.
