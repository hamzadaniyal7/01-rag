"""The eval harness -- the part that turns tuning into engineering.

Without this file, "did chunk_size=256 help?" is a vibe. With it, it's a
number. Two layers, deliberately:

RETRIEVAL (free, deterministic, no API calls)
  recall@k      did any chunk from an expected source make the top-k?
  mrr           how high was the first correct source ranked?
  Run this constantly -- it is the fast loop for chunking/embedding changes.

ANSWER (needs the model)
  must_contain  did the answer state the facts it had to state?
  citation_ok   every marker resolves to a real source, and at least one of
                the expected sources is cited
  uncited       factual sentences with no marker (groundedness smell test)
  abstention    did it correctly say "I don't know" for unanswerable questions?
  judge         optional LLM grader: faithfulness / completeness / citations

The abstention cases are not filler. A RAG system that answers everything is
worse than useless -- a confident wrong answer costs more than no answer.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from ragcite.generate import uncited_sentences
from ragcite.pipeline import Rag
from ragcite.prompts import JUDGE_SCHEMA, JUDGE_SYSTEM
from ragcite.types import Answer


@dataclass(slots=True)
class EvalCase:
    question: str
    # Substrings of the expected source path(s), e.g. "refund-policy.md".
    expect_sources: list[str] = field(default_factory=list)
    # Strings the answer must contain (case-insensitive).
    must_contain: list[str] = field(default_factory=list)
    # True for questions the corpus cannot answer: the right answer is "I don't know".
    expect_abstain: bool = False
    note: str = ""

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        return cls(
            question=data["question"],
            expect_sources=list(data.get("expect_sources", [])),
            must_contain=list(data.get("must_contain", [])),
            expect_abstain=bool(data.get("expect_abstain", False)),
            note=data.get("note", ""),
        )


@dataclass(slots=True)
class CaseResult:
    question: str
    retrieved_sources: list[str]
    recall: float | None
    reciprocal_rank: float | None
    answered: bool = False
    abstained: bool = False
    abstention_correct: bool | None = None
    must_contain_hits: float | None = None
    citation_ok: bool | None = None
    dangling_markers: list[int] = field(default_factory=list)
    uncited_count: int | None = None
    judge: dict[str, Any] | None = None
    answer: str = ""
    latency_s: float = 0.0


def load_cases(path: str | Path) -> list[EvalCase]:
    """Read a JSONL eval set (one JSON object per line, `#` comments allowed)."""
    cases: list[EvalCase] = []
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(EvalCase.from_dict(json.loads(line)))
    return cases


def evaluate(
    rag: Rag,
    cases: list[EvalCase],
    *,
    answer: bool = True,
    judge: bool = False,
    k: int | None = None,
) -> dict[str, Any]:
    results: list[CaseResult] = []
    for case in cases:
        started = time.perf_counter()
        hits = rag.search(case.question, k=k)
        sources = [h.chunk.doc_path for h in hits]
        recall, rr = _retrieval_metrics(case, sources)

        result = CaseResult(
            question=case.question,
            retrieved_sources=_unique(sources),
            recall=recall,
            reciprocal_rank=rr,
        )

        if answer:
            ans = rag.ask(case.question, k=k)
            _score_answer(case, ans, result)
            if judge:
                result.judge = _judge(rag, case, ans)

        result.latency_s = round(time.perf_counter() - started, 3)
        results.append(result)

    return {
        "config": rag.cfg.to_dict(),
        "n_cases": len(results),
        "metrics": _aggregate(results),
        "cases": [asdict(r) for r in results],
    }


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #
def _retrieval_metrics(case: EvalCase, sources: list[str]) -> tuple[float | None, float | None]:
    if not case.expect_sources:
        return None, None
    matched_any = False
    rr = 0.0
    for rank, source in enumerate(sources, 1):
        if any(exp.lower() in source.lower() for exp in case.expect_sources):
            matched_any = True
            rr = 1.0 / rank
            break
    return (1.0 if matched_any else 0.0), rr


def _score_answer(case: EvalCase, ans: Answer, result: CaseResult) -> None:
    result.answered = True
    result.abstained = ans.abstained
    result.answer = ans.text
    result.dangling_markers = ans.dangling_markers
    result.abstention_correct = ans.abstained == case.expect_abstain

    if case.expect_abstain:
        # Nothing else applies: there is no right citation for an unanswerable
        # question, and a model that invents one is already penalised above.
        return

    if case.must_contain:
        hits = sum(1 for needle in case.must_contain if needle.lower() in ans.text.lower())
        result.must_contain_hits = hits / len(case.must_contain)

    cited_paths = [c.path.lower() for c in ans.citations]
    expected_cited = (
        any(exp.lower() in p for exp in case.expect_sources for p in cited_paths)
        if case.expect_sources
        else bool(cited_paths)
    )
    result.citation_ok = bool(cited_paths) and not ans.dangling_markers and expected_cited
    result.uncited_count = len(uncited_sentences(ans.text))


def _judge(rag: Rag, case: EvalCase, ans: Answer) -> dict[str, Any] | None:
    excerpts = "\n\n".join(f"[{i}] {h.chunk.text}" for i, h in enumerate(ans.hits, 1))
    user = (
        f"Question: {case.question}\n\n"
        f"Excerpts given to the system:\n{excerpts}\n\n"
        f"Answer to grade:\n{ans.text}"
    )
    try:
        result = rag.llm.complete(JUDGE_SYSTEM, user, max_tokens=1024, json_schema=JUDGE_SCHEMA)
        return json.loads(result.text)
    except Exception as exc:  # noqa: BLE001 - a judge failure must not fail the run
        return {"error": str(exc)}


def _aggregate(results: list[CaseResult]) -> dict[str, Any]:
    def mean(values: list[float]) -> float | None:
        return round(statistics.fmean(values), 4) if values else None

    answerable = [r for r in results if r.citation_ok is not None]
    judged = [r.judge for r in results if r.judge and "error" not in r.judge]

    metrics: dict[str, Any] = {
        "recall_at_k": mean([r.recall for r in results if r.recall is not None]),
        "mrr": mean([r.reciprocal_rank for r in results if r.reciprocal_rank is not None]),
        "latency_s_mean": mean([r.latency_s for r in results]),
    }
    if any(r.answered for r in results):
        metrics["must_contain"] = mean(
            [r.must_contain_hits for r in results if r.must_contain_hits is not None]
        )
        metrics["citation_ok"] = mean([1.0 if r.citation_ok else 0.0 for r in answerable])
        metrics["abstention_correct"] = mean(
            [
                1.0 if r.abstention_correct else 0.0
                for r in results
                if r.abstention_correct is not None
            ]
        )
        metrics["uncited_sentences_total"] = sum(r.uncited_count or 0 for r in results)
        metrics["dangling_citations_total"] = sum(len(r.dangling_markers) for r in results)
    if judged:
        for key in ("faithfulness", "completeness", "citation_quality"):
            metrics[f"judge_{key}"] = mean([float(j[key]) for j in judged if key in j])
    return metrics


def _unique(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def render_report(report: dict[str, Any], verbose: bool = False) -> str:
    lines = [f"cases: {report['n_cases']}", ""]
    width = max(len(k) for k in report["metrics"]) if report["metrics"] else 10
    for key, value in report["metrics"].items():
        lines.append(f"  {key:<{width}}  {value if value is not None else '-'}")
    if verbose:
        lines.append("")
        for case in report["cases"]:
            flag = "ok " if case.get("citation_ok") in (True, None) else "BAD"
            lines.append(f"[{flag}] {case['question']}")
            if case.get("answer"):
                lines.append(f"       {' '.join(case['answer'].split())[:160]}")
            if case.get("dangling_markers"):
                lines.append(f"       dangling markers: {case['dangling_markers']}")
    return "\n".join(lines)


def save_report(report: dict[str, Any], directory: str | Path = "eval/reports") -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return path
