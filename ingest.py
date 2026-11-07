"""Ingestion pipeline: load -> extract -> clean -> chunk -> embed -> store in ChromaDB.

Usage:
    python ingest.py            # incremental (upserts, deterministic chunk IDs)
    python ingest.py --rebuild  # wipe the collection first

Also importable: `ingest(rebuild=False)` is called by app.py when the index is missing.
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter
from pathlib import Path

import chromadb
from pypdf import PdfReader

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_DIR = BASE_DIR / "chroma_db"
MANIFEST = DATA_DIR / "sources.csv"  # optional: filename,title,url,notes
COLLECTION = "ask_lky"
EMBED_MODEL = "all-MiniLM-L6-v2"
SUPPORTED = {".pdf", ".txt", ".md"}

CHUNK_SIZE = 900      # characters; MiniLM truncates at ~256 tokens (~1000 chars)
CHUNK_OVERLAP = 150
MIN_PAGE_CHARS = 40   # pages below this are treated as empty / likely scanned

_model = None


def get_embedder():
    """Load the sentence-transformers model once (heavy: ~90MB on first use)."""
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(EMBED_MODEL)
    return _model


def embed(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    vecs = get_embedder().encode(
        texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False
    )
    return vecs.tolist()


def get_collection(db_dir: Path = DB_DIR):
    client = chromadb.PersistentClient(path=str(db_dir))
    return client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})


# ---------------------------------------------------------------- loading ----
def load_manifest() -> dict[str, dict]:
    """Optional sources.csv doubles as the permission/provenance log."""
    if not MANIFEST.exists():
        return {}
    with open(MANIFEST, newline="", encoding="utf-8") as f:
        return {row["filename"]: row for row in csv.DictReader(f) if row.get("filename")}


def extract_pages(path: Path) -> list[tuple[int, str]]:
    """Return [(page_number, text)]. Page is 0 for non-paged formats (TXT/MD)."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")
            except Exception:
                raise ValueError("PDF is encrypted")
        return [(i, page.extract_text() or "") for i, page in enumerate(reader.pages, start=1)]
    return [(0, path.read_text(encoding="utf-8", errors="replace"))]


# --------------------------------------------------------------- cleaning ----
def clean_text(text: str) -> str:
    text = text.replace("\x00", " ").replace("­", "")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)            # rejoin hyphenated line breaks
    text = re.sub(r"(?m)^\s*(page\s*)?\d{1,4}\s*$", "", text, flags=re.I)  # bare page numbers
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)            # single newline -> space
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def strip_repeated_lines(pages: list[tuple[int, str]]) -> list[tuple[int, str]]:
    """Drop short lines (headers/footers) that repeat on >=40% of pages of a PDF."""
    if len(pages) < 5:
        return pages
    counts: Counter = Counter()
    for _, text in pages:
        for line in {l.strip() for l in text.splitlines() if 0 < len(l.strip()) < 80}:
            counts[line] += 1
    threshold = max(3, int(0.4 * len(pages)))
    repeated = {line for line, c in counts.items() if c >= threshold}
    if not repeated:
        return pages
    return [(n, "\n".join(l for l in text.splitlines() if l.strip() not in repeated))
            for n, text in pages]


# --------------------------------------------------------------- chunking ----
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Greedy sentence-aware chunking with character overlap."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n\n", text) if s.strip()]
    # Hard-split any sentence longer than `size` so nothing exceeds the embedder window.
    parts: list[str] = []
    for s in sentences:
        while len(s) > size:
            cut = s.rfind(" ", 0, size)
            cut = cut if cut > size // 2 else size
            parts.append(s[:cut].strip())
            s = s[cut:].strip()
        if s:
            parts.append(s)

    chunks, current = [], ""
    for s in parts:
        if current and len(current) + 1 + len(s) > size:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            tail = tail[tail.find(" ") + 1:] if " " in tail else tail  # start on a word boundary
            current = f"{tail} {s}".strip()
        else:
            current = f"{current} {s}".strip()
    if current:
        chunks.append(current)
    return chunks


# ------------------------------------------------------------------ main -----
def build_records(data_dir: Path = DATA_DIR) -> tuple[list[dict], list[str]]:
    """Walk data/, return (chunk records, warnings). Bad files are skipped, not fatal."""
    manifest = load_manifest()
    records, warnings = [], []
    files = sorted(p for p in data_dir.rglob("*") if p.is_file() and p.suffix.lower() in SUPPORTED)

    for path in files:
        rel = path.relative_to(data_dir)
        category = rel.parts[0] if len(rel.parts) > 1 else "uncategorized"
        meta = manifest.get(path.name, {})
        try:
            pages = extract_pages(path)
        except Exception as e:
            warnings.append(f"SKIPPED {rel}: {e}")
            continue
        if path.suffix.lower() == ".pdf":
            pages = strip_repeated_lines(pages)

        kept = 0
        for page_no, raw in pages:
            text = clean_text(raw)
            if len(text) < MIN_PAGE_CHARS:
                continue
            kept += 1
            for ci, chunk in enumerate(chunk_text(text)):
                records.append({
                    "id": f"{path.stem}::p{page_no}::c{ci}",
                    "text": chunk,
                    "metadata": {
                        "source": path.name,
                        "page": page_no,  # 0 = not paged (TXT/MD); Chroma disallows None
                        "category": category,
                        "chunk_id": f"{path.stem}::p{page_no}::c{ci}",
                        "title": meta.get("title") or path.stem.replace("_", " "),
                        "url": meta.get("url") or "",
                    },
                })
        if kept == 0:
            warnings.append(f"WARNING {rel}: no extractable text (scanned PDF? OCR is not supported)")
    if not files:
        warnings.append(f"No documents found under {data_dir}. Add PDF/TXT/MD files to data/<category>/.")
    return records, warnings


def ingest(rebuild: bool = False, data_dir: Path = DATA_DIR, db_dir: Path = DB_DIR) -> dict:
    records, warnings = build_records(data_dir)
    for w in warnings:
        print(w, file=sys.stderr)

    client = chromadb.PersistentClient(path=str(db_dir))
    if rebuild:
        try:
            client.delete_collection(COLLECTION)
        except Exception:
            pass
    col = client.get_or_create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    if records:
        batch = 128
        total = len(records)
        for i in range(0, total, batch):
            part = records[i:i + batch]
            end = min(i + batch, total)
            print(f"Embedding and indexing chunks {i + 1}-{end} of {total}...")
            col.upsert(
                ids=[r["id"] for r in part],
                documents=[r["text"] for r in part],
                embeddings=embed([r["text"] for r in part]),
                metadatas=[r["metadata"] for r in part],
            )
    return {"chunks_processed": len(records), "chunks_in_db": col.count(), "warnings": warnings}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Build the Ask LKY vector index.")
    ap.add_argument("--rebuild", action="store_true", help="delete and recreate the collection")
    args = ap.parse_args()
    result = ingest(rebuild=args.rebuild)
    print(f"Processed {result['chunks_processed']} chunks; collection now holds {result['chunks_in_db']}.")
