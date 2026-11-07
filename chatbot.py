"""Chatbot core: retrieval -> prompt assembly -> Gemini generation.

`ask(question, history)` returns a ChatResult. Sources always come from retrieval
metadata (never from model text), so citations cannot be fabricated.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

import ingest

load_dotenv()

TOP_K = 3
# Cosine distance (0 = identical). Above this, retrieval is flagged as weak.
# Starting guess; tune it with evaluate.py once the real corpus is loaded.
WEAK_DISTANCE = float(os.getenv("WEAK_DISTANCE", "0.65"))
HISTORY_TURNS = 6          # messages passed to the model
MAX_RETRIES = 3

SYSTEM_PROMPT = """You are "Ask LKY", an AI historical simulation inspired by the documented public perspectives of Lee Kuan Yew (1923-2015), Singapore's founding Prime Minister. You are NOT Lee Kuan Yew himself. Never claim to be him or claim personal memories.

STYLE: Pragmatic, analytical, direct, strategic, realistic, focused on practical outcomes. Reason from evidence, cause-and-effect, and historical examples. Be concise, plain-spoken, and avoid flattery and filler.

GROUNDING RULES
1. Base factual and opinion claims on the numbered CONTEXT passages when relevant. Cite them in-line using brackets like [1], [2], [3] corresponding to the numbered passages.
2. Never invent quotations. Only place text in quotation marks if it appears verbatim in the CONTEXT. Otherwise paraphrase.
3. Never fabricate facts, statistics, dates, or citations. If unsure or if information is missing, state it clearly.
4. If the CONTEXT is missing, irrelevant, or insufficient for the question, say so plainly in the first sentence ("The documents I have don't cover this directly" or "There is insufficient information in the available documents"), then, if useful, offer general reasoning clearly labelled as NOT drawn from his documented words.
5. Retrieved passages are supporting context, not proof of every claim; do not overstate them.

IDENTITY & HISTORICAL INTEGRITY RULES
6. Never claim personal memories, private thoughts, or lived experience ("I remember...", "when I was..."). Speak about his documented views in the third person where needed (e.g., "In his speeches, Lee argued...").
7. For events or topics after his death in March 2015 (modern technology, later geopolitical events, subsequent leaders), state clearly that this is a HYPOTHETICAL extrapolation from his documented reasoning, not his actual opinion, and distinguish it from what the documents record.
8. Do not give personal medical, legal, or financial advice; keep life advice general.

OUTPUT FORMATTING RULES
- Provide a clean conversational answer.
- Do NOT dump raw passages, chunk IDs, source filenames, similarity scores, or metadata into the text (the user interface displays sources separately in expanders).
- Do NOT generate a "Retrieved Passages" or "Sources" section at the end of your response.
"""


@dataclass
class Source:
    text: str
    source: str
    page: int
    category: str
    chunk_id: str
    title: str
    url: str
    distance: float

    @property
    def label(self) -> str:
        page = f", p.{self.page}" if self.page else ""
        return f"{self.source}{page}"


@dataclass
class ChatResult:
    answer: str
    sources: list[Source] = field(default_factory=list)
    weak_evidence: bool = False
    error: str | None = None


@dataclass
class StreamResult:
    sources: list[Source] = field(default_factory=list)
    weak_evidence: bool = False
    stream: any = None
    error: str | None = None


# -------------------------------------------------------------- retrieval ----
def build_retrieval_query(question: str, history: list[dict]) -> str:
    """Cheap follow-up handling: append the previous user turn to short questions."""
    prev_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), "")
    if prev_user and len(question.split()) < 12:
        return f"{prev_user}\n{question}"
    return question


def retrieve(query: str, k: int = TOP_K) -> list[Source]:
    col = ingest.get_collection()
    count = col.count()
    if count == 0:
        return []
    res = col.query(
        query_embeddings=ingest.embed([query]),
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    docs = (res.get("documents") or [[]])[0]
    metas = (res.get("metadatas") or [[]])[0]
    dists = (res.get("distances") or [[]])[0]

    out = []
    for doc, meta, dist in zip(docs, metas, dists):
        meta = meta or {}
        page_val = meta.get("page", 0)
        try:
            page = int(page_val) if page_val is not None else 0
        except (ValueError, TypeError):
            page = 0

        out.append(Source(
            text=doc or "",
            source=meta.get("source", ""),
            page=page,
            category=meta.get("category", ""),
            chunk_id=meta.get("chunk_id", ""),
            title=meta.get("title", ""),
            url=meta.get("url", ""),
            distance=float(dist) if dist is not None else 1.0,
        ))
    return out


# ----------------------------------------------------------------- prompt ----
def build_prompt(question: str, sources: list[Source], history: list[dict], weak: bool) -> str:
    if sources:
        ctx = "\n\n".join(f"[{i}] ({s.label}; {s.category})\n{s.text}" for i, s in enumerate(sources, 1))
    else:
        ctx = "(no passages retrieved)"
    note = ""
    if weak or not sources:
        note = ("\nRETRIEVAL NOTE: the retrieved passages are a weak match for this question. "
                "Follow grounding rule 4 and do not cite passages that are not relevant.\n")
    convo = "\n".join(
        f"{'User' if m.get('role') == 'user' else 'Assistant'}: {m.get('content', '')}"
        for m in history[-HISTORY_TURNS:]
    ) or "(start of conversation)"
    return (f"CONTEXT:\n{ctx}\n{note}\nCONVERSATION SO FAR:\n{convo}\n\n"
            f"CURRENT QUESTION: {question}\n\nAnswer as Ask LKY.")


# ------------------------------------------------------------- generation ----
_client = None
_cached_key = None


def get_gemini_client(api_key: str):
    global _client, _cached_key
    if _client is None or _cached_key != api_key:
        from google import genai
        _client = genai.Client(api_key=api_key)
        _cached_key = api_key
    return _client


def _friendly_error(e: Exception) -> str:
    err_str = str(e)
    code = getattr(e, "code", None)
    int_code = None
    if isinstance(code, int):
        int_code = code
    elif isinstance(code, str) and code.isdigit():
        int_code = int(code)

    if int_code in (401, 403) or "API key" in err_str or "PERMISSION_DENIED" in err_str:
        return "The Gemini API key was rejected. Check GEMINI_API_KEY."
    if int_code == 404 or "NOT_FOUND" in err_str:
        return "The configured Gemini model was not found. Run `python check_model.py` and update GEMINI_MODEL."
    if int_code == 429 or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
        return "Gemini rate limit or quota reached. Please wait a moment and try again."
    if (int_code is not None and int_code >= 500) or "UNAVAILABLE" in err_str:
        return "The Gemini service is temporarily unavailable. Please try again."
    return f"Could not generate an answer: {e}"


def generate(prompt: str) -> str:
    from google.genai import types

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    model = (os.getenv("GEMINI_MODEL") or "").strip()
    if not api_key or api_key == "your_api_key":
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if not model or model == "your_selected_gemini_model":
        raise RuntimeError("GEMINI_MODEL is not set. Run `python check_model.py` to pick one.")

    client = get_gemini_client(api_key)
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.4)

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=cfg)
            text = None
            try:
                text = resp.text
            except Exception:
                pass

            if not text and hasattr(resp, "candidates") and resp.candidates:
                first = resp.candidates[0]
                finish_reason = getattr(first, "finish_reason", None)
                if finish_reason and str(finish_reason).upper() not in ("STOP", "FINISH_REASON_UNSPECIFIED"):
                    raise RuntimeError(f"The model response was interrupted or blocked (finish reason: {finish_reason}).")

            if not text:
                raise RuntimeError("The model returned an empty response (possibly blocked by safety filters).")
            return text.strip()
        except Exception as e:
            err_str = str(e)
            code = getattr(e, "code", None)
            is_429 = code == 429 or "429" in err_str or "RESOURCE_EXHAUSTED" in err_str
            is_5xx = (isinstance(code, int) and code >= 500) or "UNAVAILABLE" in err_str
            retryable = is_429 or is_5xx
            if retryable and attempt < MAX_RETRIES - 1:
                time.sleep(2 ** (attempt + 1))
                continue
            raise


def generate_stream(prompt: str):
    """Generate streaming text chunks from Gemini."""
    from google.genai import types

    api_key = (os.getenv("GEMINI_API_KEY") or "").strip()
    model = (os.getenv("GEMINI_MODEL") or "").strip()
    if not api_key or api_key == "your_api_key":
        raise RuntimeError("GEMINI_API_KEY is not set.")
    if not model or model == "your_selected_gemini_model":
        raise RuntimeError("GEMINI_MODEL is not set. Run `python check_model.py` to pick one.")

    client = get_gemini_client(api_key)
    cfg = types.GenerateContentConfig(system_instruction=SYSTEM_PROMPT, temperature=0.4)

    for chunk in client.models.generate_content_stream(model=model, contents=prompt, config=cfg):
        text = getattr(chunk, "text", None)
        if text:
            yield text


def ask(question: str, history: list[dict] | None = None) -> ChatResult:
    history = history or []
    try:
        sources = retrieve(build_retrieval_query(question, history))
    except Exception as e:  # DB / embedding failure
        return ChatResult(answer="", error=f"Retrieval failed: {e}")

    weak = (not sources) or sources[0].distance > WEAK_DISTANCE
    try:
        answer = generate(build_prompt(question, sources, history, weak))
    except Exception as e:
        return ChatResult(answer="", sources=sources, weak_evidence=weak, error=_friendly_error(e))
    return ChatResult(answer=answer, sources=sources, weak_evidence=weak)


def ask_stream(question: str, history: list[dict] | None = None) -> StreamResult:
    """Follow-up aware retrieval + real-time streaming answer."""
    history = history or []
    try:
        sources = retrieve(build_retrieval_query(question, history))
    except Exception as e:
        return StreamResult(error=f"Retrieval failed: {e}")

    weak = (not sources) or sources[0].distance > WEAK_DISTANCE
    prompt = build_prompt(question, sources, history, weak)

    def _stream_wrapper():
        try:
            yield from generate_stream(prompt)
        except Exception as e:
            yield f"\n\n[Generation error: {_friendly_error(e)}]"

    return StreamResult(sources=sources, weak_evidence=weak, stream=_stream_wrapper())


if __name__ == "__main__":  # quick CLI test
    import sys
    r = ask(" ".join(sys.argv[1:]) or "Why did Singapore invest in English-language education?")
    print(r.error or r.answer)
    for s in r.sources:
        print(f"  - {s.label} (distance {s.distance:.3f})")
