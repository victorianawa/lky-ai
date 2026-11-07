"""Basic evaluation for Ask LKY. Results are whatever the run produces; nothing is hardcoded.

Usage:
    python evaluate.py                    # retrieval + generation + LLM-judge support check
    python evaluate.py --retrieval-only   # no Gemini calls (retrieval metrics only)

Before running, edit `expected_sources` (filename substrings of the documents YOU added to data/)
and `keywords` (terms the relevant passages should contain) for each question. A question with an
empty `expected_sources` is excluded from the source hit rate and reported as n/a.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import chatbot

# in_corpus=False marks out-of-scope questions used to test abstention.
QUESTIONS = [
    {"id": 1, "category": "Singapore history", "in_corpus": True,
     "q": "Why did Singapore leave Malaysia in 1965, and how did leaders respond?",
     "expected_sources": ["1965_independence_speech", "1965_auckland_university_speech", "lee_kuan_yew_biography"], "keywords": ["malaysia", "separation", "independence"]},
    {"id": 2, "category": "Singapore history", "in_corpus": True,
     "q": "What were the biggest challenges Singapore faced after independence?",
     "expected_sources": ["1965_independence_speech", "lky_economic_development_speech", "lee_kuan_yew_biography"], "keywords": ["independence", "survival", "unemployment"]},
    {"id": 3, "category": "Economic development", "in_corpus": True,
     "q": "How did Singapore attract foreign investment and multinational companies?",
     "expected_sources": ["lky_economic_development_speech", "1996_asian_wall_street_journal_interview"], "keywords": ["investment", "multinational", "foreign"]},
    {"id": 4, "category": "Economic development", "in_corpus": True,
     "q": "Why does Singapore stress economic growth and jobs?",
     "expected_sources": ["lky_economic_development_speech", "1959_city_hall_speech"], "keywords": ["growth", "jobs", "economy"]},
    {"id": 5, "category": "Leadership", "in_corpus": True,
     "q": "What qualities make an effective leader?",
     "expected_sources": ["2010_seth_mydans_interview", "lee_kuan_yew_biography"], "keywords": ["leader", "leadership", "character"]},
    {"id": 6, "category": "Geopolitics", "in_corpus": True,
     "q": "How should a small country manage relations with big powers like China and the US?",
     "expected_sources": ["lky_regional_foreign_policy", "1965_canberra_speech"], "keywords": ["china", "united states", "america"]},
    {"id": 7, "category": "Geopolitics", "in_corpus": True,
     "q": "What is Singapore's approach to relations with its neighbours?",
     "expected_sources": ["1965_auckland_university_speech", "lky_regional_foreign_policy", "1965_letter_to_menzies"], "keywords": ["malaysia", "indonesia", "neighbour"]},
    {"id": 8, "category": "Education", "in_corpus": True,
     "q": "Why is education, including bilingual policy, important to national development?",
     "expected_sources": ["lee_kuan_yew_biography", "1996_asian_wall_street_journal_interview"], "keywords": ["education", "english", "bilingual", "school"]},
    {"id": 9, "category": "Governance", "in_corpus": True,
     "q": "What is the role of the rule of law and clean government in Singapore's success?",
     "expected_sources": ["1959_city_hall_speech", "lee_kuan_yew_biography"], "keywords": ["corruption", "law", "government"]},
    {"id": 10, "category": "Modern technology (post-2015)", "in_corpus": True,
     "q": "What might he think about artificial intelligence and its effect on jobs?",
     "expected_sources": ["lky_economic_development_speech", "lee_kuan_yew_biography"], "keywords": ["technology", "jobs", "skills"], "expect_hypothetical": True},
    {"id": 11, "category": "Out-of-scope (abstention)", "in_corpus": False,
     "q": "What did he say about the 2022 FIFA World Cup final?", "expected_sources": [], "keywords": []},
    {"id": 12, "category": "Out-of-scope (abstention)", "in_corpus": False,
     "q": "Give me a recipe for chicken rice.", "expected_sources": [], "keywords": []},
]

ABSTAIN_RE = re.compile(
    r"(don'?t|do not|doesn'?t|does not) (directly )?(cover|address|contain|have)|not (directly )?covered|"
    r"insufficient|no (relevant )?(information|passages|material)|can'?t find|cannot find|"
    r"not drawn from|outside (the|my) (scope|documents)", re.I)
HYPO_RE = re.compile(r"hypothetic|extrapolat|after (his|lee'?s) death|not (his )?(actual|documented)", re.I)


def judge_support(answer: str, sources) -> tuple[bool | None, str]:
    """LLM-as-judge (Gemini): are the answer's key claims supported by the passages?
    This is a model judgment, not ground truth."""
    ctx = "\n\n".join(f"[{i}] {s.text}" for i, s in enumerate(sources, 1))
    prompt = (
        "You are a strict evaluator. Given PASSAGES and an ANSWER, decide whether the answer's main "
        "factual/opinion claims are supported by the passages. General reasoning that is explicitly "
        "labelled as not from the documents does not count against support.\n"
        'Reply with ONLY JSON: {"supported": true|false, "reason": "<one sentence>"}\n\n'
        f"PASSAGES:\n{ctx}\n\nANSWER:\n{answer}"
    )
    try:
        raw = chatbot.generate(prompt)
        match = re.search(r"\{[^{}]*\}", raw, re.S)
        if match:
            data = json.loads(match.group(0))
            return bool(data.get("supported")), str(data.get("reason", ""))
        return None, f"judge failed: no JSON object in response: {raw[:80]}"
    except Exception as e:
        return None, f"judge failed: {e}"


def run(retrieval_only: bool, delay: float) -> dict:
    rows = []
    for item in QUESTIONS:
        sources = chatbot.retrieve(item["q"])
        row = {"id": item["id"], "category": item["category"], "question": item["q"],
               "in_corpus": item["in_corpus"],
               "retrieved": [{"label": s.label, "distance": round(s.distance, 3)} for s in sources]}
        row["avg_similarity"] = round(sum(1 - s.distance for s in sources) / len(sources), 3) if sources else None
        row["top1_distance"] = round(sources[0].distance, 3) if sources else None
        row["weak_flag"] = (not sources) or sources[0].distance > chatbot.WEAK_DISTANCE

        exp = item["expected_sources"]
        row["source_hit"] = (any(e.lower() in s.source.lower() for s in sources for e in exp) if exp else None)
        blob = " ".join(s.text.lower() for s in sources)
        row["keyword_hit"] = (any(k in blob for k in item["keywords"]) if item["keywords"] else None)

        if not retrieval_only:
            res = chatbot.ask(item["q"], [])
            row["error"] = res.error
            row["answer"] = res.answer
            if not res.error:
                cited = {int(n) for n in re.findall(r"\[(\d+)\]", res.answer)}
                row["citations_valid"] = all(1 <= n <= len(res.sources) for n in cited)
                row["n_citations"] = len(cited)
                row["abstained"] = bool(ABSTAIN_RE.search(res.answer))
                if item.get("expect_hypothetical"):
                    row["hypothetical_labelled"] = bool(HYPO_RE.search(res.answer))
                if item["in_corpus"] and res.sources:
                    time.sleep(delay)
                    row["supported"], row["judge_reason"] = judge_support(res.answer, res.sources)
            time.sleep(delay)
        rows.append(row)
        print(f"done {item['id']}/{len(QUESTIONS)}")
    return {"weak_distance_threshold": chatbot.WEAK_DISTANCE, "retrieval_only": retrieval_only, "rows": rows}


def _rate(vals) -> str:
    vals = [v for v in vals if v is not None]
    return f"{sum(vals)}/{len(vals)} ({100 * sum(vals) / len(vals):.0f}%)" if vals else "n/a"


def summarize(report: dict) -> str:
    rows = report["rows"]
    inc = [r for r in rows if r["in_corpus"]]
    oos = [r for r in rows if not r["in_corpus"]]
    sims = [r["avg_similarity"] for r in inc if r["avg_similarity"] is not None]
    lines = [
        "# Evaluation results", "",
        f"Generated by `evaluate.py` (weak-evidence distance threshold = {report['weak_distance_threshold']}). "
        f"{len(inc)} in-corpus questions, {len(oos)} out-of-scope. Small sample: treat as indicative, not conclusive.", "",
        "## Summary", "", "| Metric | Result |", "|---|---|",
        f"| Mean similarity of top-3 chunks (1 - cosine distance), in-corpus | {sum(sims) / len(sims):.3f} |" if sims else "| Mean similarity | n/a |",
        f"| Source hit rate (expected source in top 3; only questions with expected sources set) | {_rate([r['source_hit'] for r in inc])} |",
        f"| Keyword hit rate (any expected keyword in retrieved text) | {_rate([r['keyword_hit'] for r in inc])} |",
        f"| Out-of-scope questions flagged weak by retrieval | {_rate([r['weak_flag'] for r in oos])} |",
    ]
    if not report["retrieval_only"]:
        lines += [
            f"| Answers judged supported by passages (LLM judge, not ground truth) | {_rate([r.get('supported') for r in inc])} |",
            f"| Answers whose [n] citations all point to retrieved passages | {_rate([r.get('citations_valid') for r in inc])} |",
            f"| Out-of-scope questions where the answer acknowledged insufficient evidence | {_rate([r.get('abstained') for r in oos])} |",
            f"| Post-2015 question labelled as hypothetical | {_rate([r.get('hypothetical_labelled') for r in rows])} |",
        ]
    lines += ["", "## Per question", "", "| # | Category | Top-1 dist | Weak | Src hit | Kw hit | Supported | Cites ok | Abstained |",
              "|---|---|---|---|---|---|---|---|---|"]
    f = lambda v: "n/a" if v is None else ("yes" if v is True else "no" if v is False else str(v))
    for r in rows:
        lines.append(f"| {r['id']} | {r['category']} | {f(r['top1_distance'])} | {f(r['weak_flag'])} | "
                     f"{f(r['source_hit'])} | {f(r['keyword_hit'])} | {f(r.get('supported'))} | "
                     f"{f(r.get('citations_valid'))} | {f(r.get('abstained'))} |")
    errs = [r for r in rows if r.get("error")]
    if errs:
        lines += ["", f"**{len(errs)} question(s) failed during generation** (see JSON for details)."]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval-only", action="store_true")
    ap.add_argument("--delay", type=float, default=4.0, help="seconds between Gemini calls (rate limits)")
    args = ap.parse_args()
    report = run(args.retrieval_only, args.delay)
    Path("evaluation_results.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    Path("evaluation_results.md").write_text(summarize(report), encoding="utf-8")
    print("Wrote evaluation_results.md and evaluation_results.json")
