"""Phase 3-1: gemini-2.5-flash difficulty calibration via RAG.

For each reviewed item we run gemini `attempts` times with the question + top-k
chunks from our index. Each attempt is graded against the gold answer using a
numeric (±relative-tolerance) or exact-match rule. We then label items:

  too_easy : all attempts correct
  too_hard : no attempts correct
  accepted : 1..(attempts-1) correct
"""
from __future__ import annotations

import json
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .common import get_logger, require_env

log = get_logger("calibrator")

PROMPT_TEMPLATE = """당신은 한국어 재무 공시 문서 QA 전문가입니다.
주어진 evidence를 바탕으로 질문에 답하세요.

Question: {question}

Evidence:
{evidence_text}

답변을 단일 값 또는 항목으로 간결하게 제시하고, reasoning step을 포함하세요.
마지막 줄은 반드시 다음 형식으로 끝내세요:
최종답: <단일 값 또는 항목>
"""

FINAL_ANSWER_RE = re.compile(r"최종답\s*[:：]\s*(.+?)\s*$", re.MULTILINE)


# ---- answer normalization & comparison ----

_NUM_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def parse_number(s: str) -> float | None:
    """Strip commas + unit suffix and try to parse a Korean numeric expression."""
    if s is None:
        return None
    m = _NUM_RE.search(str(s).replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


def grade(predicted: str, gold: Any, *, rel_tol: float) -> bool:
    """Numeric ±rel_tol comparison, else exact-match on normalized strings."""
    if predicted is None or predicted == "":
        return False
    gnum = parse_number(gold) if not isinstance(gold, (int, float)) else float(gold)
    pnum = parse_number(predicted)
    if gnum is not None and pnum is not None:
        if gnum == 0:
            return abs(pnum) < rel_tol
        return abs(pnum - gnum) / abs(gnum) <= rel_tol
    # EM fallback
    return normalize_text(predicted) == normalize_text(gold)


# ---- token cost estimation ----

@dataclass
class CalibCost:
    input_tokens: int = 0
    output_tokens: int = 0
    @property
    def usd(self) -> float:
        return self._usd
    _usd: float = 0.0


# ---- main runner ----

@dataclass
class AttemptResult:
    raw: str
    predicted: str | None
    correct: bool
    input_tokens: int = 0
    output_tokens: int = 0


@dataclass
class ItemCalibration:
    item_id: str
    task_type: str
    classification: str  # too_easy | accepted | too_hard
    attempts: list[AttemptResult] = field(default_factory=list)
    retrieved_chunk_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "item_id": self.item_id,
            "task_type": self.task_type,
            "classification": self.classification,
            "attempts": [a.__dict__ for a in self.attempts],
            "retrieved_chunk_ids": self.retrieved_chunk_ids,
        }


class Calibrator:
    def __init__(
        self,
        index: "RagIndex",  # forward-typed to keep this import lazy too
        *,
        model_name: str,
        attempts: int = 3,
        rel_tol: float = 0.01,
        top_k: int = 5,
        input_price_per_1m: float = 0.30,
        output_price_per_1m: float = 2.50,
        api_key: str | None = None,
        provider: str = "gemini",
    ) -> None:
        self.provider = provider
        self.model_name = model_name
        if provider == "gemini":
            import google.generativeai as genai  # lazy: avoids hard dep for offline tests
            self._genai = genai
            genai.configure(api_key=api_key or require_env("GOOGLE_API_KEY"))
            self.model = genai.GenerativeModel(model_name)
            self._openai = None
        elif provider == "openai":
            from openai import OpenAI
            self._openai = OpenAI(api_key=api_key or require_env("OPENAI_API_KEY"))
            self._genai = None
            self.model = None
        else:
            raise ValueError(f"Unknown provider: {provider}")
        self.attempts = attempts
        self.rel_tol = rel_tol
        self.top_k = top_k
        self.index = index
        self.input_price = input_price_per_1m
        self.output_price = output_price_per_1m
        self._total_in = 0
        self._total_out = 0
        # Guards the shared token counters when calibrate() runs across threads.
        self._lock = threading.Lock()

    def estimate_usd(self) -> float:
        return (
            self._total_in / 1_000_000 * self.input_price
            + self._total_out / 1_000_000 * self.output_price
        )

    @staticmethod
    def _extract_final(text: str) -> str | None:
        matches = FINAL_ANSWER_RE.findall(text)
        if matches:
            return matches[-1].strip()
        # Fallback: last non-empty line
        for line in reversed(text.splitlines()):
            if line.strip():
                return line.strip()
        return None

    def _call_llm(self, prompt: str) -> tuple[str, int, int]:
        """Provider-agnostic LLM call. Returns (raw_text, input_tokens, output_tokens)."""
        if self.provider == "gemini":
            resp = self.model.generate_content(prompt)
            raw = (resp.text or "").strip() if hasattr(resp, "text") else ""
            in_tok = out_tok = 0
            meta = getattr(resp, "usage_metadata", None)
            if meta is not None:
                in_tok = getattr(meta, "prompt_token_count", 0) or 0
                out_tok = getattr(meta, "candidates_token_count", 0) or 0
            return raw, in_tok, out_tok
        # openai
        resp = self._openai.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = (resp.choices[0].message.content or "").strip()
        usage = resp.usage
        in_tok = getattr(usage, "prompt_tokens", 0) or 0
        out_tok = getattr(usage, "completion_tokens", 0) or 0
        return raw, in_tok, out_tok

    def _run_once(self, prompt: str) -> AttemptResult:
        for retry in range(4):
            try:
                raw, in_tok, out_tok = self._call_llm(prompt)
                pred = self._extract_final(raw)
                with self._lock:
                    self._total_in += in_tok
                    self._total_out += out_tok
                return AttemptResult(raw=raw, predicted=pred, correct=False,
                                     input_tokens=in_tok, output_tokens=out_tok)
            except Exception as e:
                # Bounded exponential backoff (cap 8s) — uncapped growth just
                # stalls the run on a persistently failing endpoint.
                wait = min(2 ** retry, 8)
                log.warning("%s call failed (retry %d): %s — sleeping %ds",
                            self.provider, retry, e, wait)
                time.sleep(wait)
        return AttemptResult(raw="", predicted=None, correct=False)

    def calibrate(self, item: dict) -> ItemCalibration:
        # Retrieve top_k chunks for *this question*, with company/year filter.
        where = {"$and": [
            {"company": item["source_company"]},
            {"year": int(str(item.get("source_report", "0000"))[:4]) if str(item.get("source_report", ""))[:4].isdigit() else 0},
        ]}
        hits = self.index.query(item["question"], top_k=self.top_k, where=where)
        if not hits:
            hits = self.index.query(item["question"], top_k=self.top_k)
        evidence_text = "\n\n".join(
            f"[chunk={h['chunk_id']}]\n{h['text']}" for h in hits
        )
        prompt = PROMPT_TEMPLATE.format(question=item["question"], evidence_text=evidence_text)

        attempts: list[AttemptResult] = []
        n_correct = 0
        for _ in range(self.attempts):
            a = self._run_once(prompt)
            a.correct = grade(a.predicted or "", item["gold_answer"], rel_tol=self.rel_tol)
            if a.correct:
                n_correct += 1
            attempts.append(a)

        if n_correct == self.attempts:
            cls = "too_easy"
        elif n_correct == 0:
            cls = "too_hard"
        else:
            cls = "accepted"

        return ItemCalibration(
            item_id=item["id"],
            task_type=item.get("task_type", "?"),
            classification=cls,
            attempts=attempts,
            retrieved_chunk_ids=[h["chunk_id"] for h in hits],
        )

    def calibrate_many(
        self,
        items: list[dict],
        *,
        max_workers: int = 4,
        on_result: Callable[[ItemCalibration, dict], None] | None = None,
    ) -> list[ItemCalibration]:
        """Calibrate items concurrently and return results in input order.

        Each item still runs its `attempts` LLM calls sequentially inside
        `calibrate()`; parallelism is across items, which overlaps the dominant
        cost (LLM API round-trips). `max_workers` bounds concurrency for rate
        limits. `on_result(calibration, item)` is invoked from the *calling*
        thread as each result completes — safe for incremental saving without
        extra locking — while the shared token counters are guarded internally.
        """
        if max_workers <= 1:
            results = []
            for it in items:
                cr = self.calibrate(it)
                if on_result:
                    on_result(cr, it)
                results.append(cr)
            return results

        results_by_id: dict[str, ItemCalibration] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = {ex.submit(self.calibrate, it): it for it in items}
            for fut in as_completed(futures):
                it = futures[fut]
                cr = fut.result()
                results_by_id[cr.item_id] = cr
                if on_result:  # runs in the main thread → save is sequential & safe
                    on_result(cr, it)
        return [results_by_id[it["id"]] for it in items if it["id"] in results_by_id]
