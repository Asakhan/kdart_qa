"""Phase 2: generate draft K-DART-QA items via the Anthropic API (Claude).

Per task-type, the generator:
  1. Selects evidence chunks from the RAG index using type-specific queries +
     metadata filters. Iterates over (company, year) so distribution stays
     diverse and the per-bucket cap from config is respected.
  2. Prompts Claude with the chunk text and a strict JSON schema. The system
     prompt enforces *evidence-bounded* generation — no outside knowledge.
  3. Parses the returned JSON and self-verifies:
        - reasoning_hops matches len(reasoning_steps)
        - every evidence_quote appears as a substring of some referenced chunk
        - referenced evidence_chunks all exist in the input set
  4. On verification failure, regenerates up to `max_attempts` times before
     discarding the slot and moving on (logged to logs/).

Anthropic credentials are read from ANTHROPIC_API_KEY only.
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from typing import Any, Iterable

import anthropic

from .common import get_logger, require_env
from .rag_index import RagIndex

log = get_logger("question_generator")

# ---- task-type definitions ----

@dataclass
class TaskTypeSpec:
    code: str            # T1..T5
    label: str           # human-readable
    query_seeds: list[str]
    section_filters: list[str]  # accepted section ids; empty = any
    needs_multi_year: bool = False
    needs_footnote: bool = False
    min_chunks: int = 1
    max_chunks: int = 3
    description: str = ""


TASK_TYPES: dict[str, TaskTypeSpec] = {
    "T1": TaskTypeSpec(
        code="T1",
        label="단일 수치 추출",
        query_seeds=["매출액 영업이익 당기순이익", "요약재무정보 자산총계 자본총계"],
        section_filters=["재무_1_요약", "재무_2_연결재무", "사업의내용_5_매출"],
        min_chunks=1, max_chunks=2,
        description="evidence에 명시된 단일 수치를 그대로 묻는 문항.",
    ),
    "T2": TaskTypeSpec(
        code="T2",
        label="비율 계산 (증감률·비중)",
        query_seeds=["전년 대비 증감률 매출", "영업이익률 영업이익 매출"],
        section_filters=["재무_1_요약", "재무_2_연결재무", "사업의내용_5_매출"],
        min_chunks=1, max_chunks=3,
        description="두 수치로부터 비율/증감률을 1~2단계로 계산하는 문항.",
    ),
    "T3": TaskTypeSpec(
        code="T3",
        label="다중 사업부 비교",
        query_seeds=["사업부문별 매출 비중", "부문별 매출 영업이익 구성"],
        section_filters=["사업의내용_5_매출", "재무_2_연결재무"],
        min_chunks=1, max_chunks=3,
        description="사업부문 표에서 비교·최대/최소를 묻는 문항.",
    ),
    "T4": TaskTypeSpec(
        code="T4",
        label="다단계 조건 추론",
        query_seeds=[
            "사업부문별 매출 영업이익 연구개발 R&D",
            "부문별 영업이익률 연구개발비",
        ],
        section_filters=[
            "사업의내용_5_매출",
            "사업의내용_7_연구개발",
            "재무_2_연결재무",
        ],
        min_chunks=2, max_chunks=4,
        description="필터 조건(예: 이익률 10% 이상)과 정렬 조건(예: 최대)을 결합하는 ≥3단계 문항.",
    ),
    "T5": TaskTypeSpec(
        code="T5",
        label="별표·각주 참조",
        query_seeds=["(주1) (주2) 주석 단서 조정", "각주 별표 *"],
        section_filters=["재무_1_요약", "재무_2_연결재무"],
        needs_footnote=True,
        min_chunks=1, max_chunks=2,
        description="표 하단의 주석(주1, 주2, *)을 적용하여 답을 도출하는 문항.",
    ),
}


# ---- prompt ----

SYSTEM_PROMPT = """당신은 한국 금융감독원 전자공시시스템(DART) 사업보고서를 기반으로
다단계 수치 추론 QA 데이터셋(K-DART-QA)을 만드는 도구입니다.

규칙:
1. evidence로 주어진 청크에 명시된 사실만 사용하세요. 외부 지식 사용 금지.
2. 정답은 단일 값(수치 + 단위) 또는 단일 항목명이어야 합니다. 자유 형식 답변 금지.
3. 정답이 시간이 지나도 변하지 않는 사실이어야 합니다 (보고서에 고정된 수치).
4. evidence_quotes에는 청크에서 정답 도출에 실제로 사용한 문장을 **정확한 문자열로** 인용하세요.
5. reasoning_steps는 자연어로 작성하고, reasoning_hops는 그 길이와 일치해야 합니다.
6. 단순히 표에서 셀 값을 옮겨 적는 것은 T1에서만 허용됩니다. T2~T5는 반드시 계산/필터/정렬 단계가 포함되어야 합니다.
7. 한국어 재무 용어가 모호하면 추측하지 말고 generation_notes에 명시하세요.

출력은 반드시 다음 형식의 JSON 단일 객체여야 합니다. 코드 펜스나 설명 텍스트 없이 JSON만 출력하세요."""


USER_PROMPT_TEMPLATE = """[과제 유형] {task_code} — {task_label}
{task_description}

[목표 회사·년도] {company}, {year}

[사용 가능한 evidence 청크들]
{chunks_block}

위 evidence만 사용하여 다음 JSON 스키마에 맞는 문항 1개를 작성하세요.
evidence_chunks에는 실제로 답 도출에 사용한 chunk_id만 포함하세요.

{{
  "question": "한국어 질문",
  "gold_answer": "단일 값 또는 항목명",
  "gold_answer_unit": "억원|%|항목명 등",
  "evidence_chunks": ["{first_chunk_id}"],
  "evidence_quotes": ["..."],
  "reasoning_steps": ["1. ...", "2. ..."],
  "reasoning_hops": 2,
  "generation_notes": "어떤 추론을 의도했는지 메모"
}}"""


# ---- generator ----

@dataclass
class GenerationStats:
    attempts: dict[str, int] = field(default_factory=dict)
    successes: dict[str, int] = field(default_factory=dict)
    discards: dict[str, int] = field(default_factory=dict)
    discard_reasons: list[str] = field(default_factory=list)


class QuestionGenerator:
    def __init__(
        self,
        index: RagIndex,
        *,
        model: str,
        max_tokens: int = 4096,
        max_attempts: int = 3,
        max_per_company_year: int = 5,
        api_key: str | None = None,
        seed: int = 42,
    ) -> None:
        self.index = index
        self.client = anthropic.Anthropic(api_key=api_key or require_env("ANTHROPIC_API_KEY"))
        self.model = model
        self.max_tokens = max_tokens
        self.max_attempts = max_attempts
        self.max_per_company_year = max_per_company_year
        self._rng = random.Random(seed)
        self._issued_ids: set[str] = set()
        self.stats = GenerationStats()

    # ---- evidence selection ----

    def _select_evidence(
        self,
        spec: TaskTypeSpec,
        *,
        company: str,
        year: int,
    ) -> list[dict]:
        """Pull candidate chunks for one (task_type, company, year) tuple."""
        seeds = list(spec.query_seeds)
        self._rng.shuffle(seeds)
        seen: dict[str, dict] = {}
        for seed_q in seeds:
            base_where: dict[str, Any] = {"company": company}
            # Note: chroma needs $and for multi-key + $in
            if spec.section_filters:
                where: dict[str, Any] = {
                    "$and": [
                        {"company": company},
                        {"year": year},
                        {"section": {"$in": spec.section_filters}},
                    ]
                }
            else:
                where = {"$and": [{"company": company}, {"year": year}]}
            hits = self.index.query(seed_q, top_k=8, where=where)
            for h in hits:
                seen.setdefault(h["chunk_id"], h)
        candidates = list(seen.values())

        if spec.needs_footnote:
            footnote_re = re.compile(r"\(주\s*\d\)|주\d\)|\*[^*]")
            candidates = [c for c in candidates if footnote_re.search(c["text"])]

        if not candidates:
            return []

        # Prefer chunks with tables for numeric questions.
        candidates.sort(key=lambda c: (not c["metadata"].get("has_table"), c["distance"] or 0))
        picked = candidates[: spec.max_chunks]
        if len(picked) < spec.min_chunks:
            return []
        return picked

    # ---- prompt + call ----

    @staticmethod
    def _format_chunks_block(chunks: list[dict]) -> str:
        lines = []
        for c in chunks:
            meta = c["metadata"]
            head = (
                f"--- chunk_id={c['chunk_id']} | section={meta.get('section')} "
                f"| has_table={meta.get('has_table')} ---"
            )
            lines.append(head)
            lines.append(c["text"])
            lines.append("")
        return "\n".join(lines)

    def _call_claude(self, prompt: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        # concatenate text blocks (modern SDK returns content blocks)
        parts = []
        for block in resp.content:
            if getattr(block, "type", None) == "text":
                parts.append(block.text)
        return "".join(parts).strip()

    @staticmethod
    def _extract_json(s: str) -> dict | None:
        s = s.strip()
        # strip markdown fences if model still added them
        if s.startswith("```"):
            s = re.sub(r"^```(json)?\s*", "", s)
            s = re.sub(r"\s*```$", "", s)
        # locate the first balanced { ... }
        start = s.find("{")
        if start < 0:
            return None
        depth = 0
        for i in range(start, len(s)):
            if s[i] == "{":
                depth += 1
            elif s[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError as e:
                        log.warning("JSON decode failed: %s", e)
                        return None
        return None

    # ---- verification ----

    @staticmethod
    def _verify(payload: dict, chunks: list[dict]) -> tuple[bool, str]:
        required = {
            "question", "gold_answer", "gold_answer_unit",
            "evidence_chunks", "evidence_quotes",
            "reasoning_steps", "reasoning_hops",
        }
        missing = required - set(payload.keys())
        if missing:
            return False, f"missing fields: {missing}"
        if not isinstance(payload["reasoning_steps"], list) or not payload["reasoning_steps"]:
            return False, "reasoning_steps empty or not a list"
        if int(payload["reasoning_hops"]) != len(payload["reasoning_steps"]):
            return False, "reasoning_hops != len(reasoning_steps)"
        if not isinstance(payload["evidence_quotes"], list) or not payload["evidence_quotes"]:
            return False, "evidence_quotes empty"
        chunk_by_id = {c["chunk_id"]: c for c in chunks}
        for cid in payload["evidence_chunks"]:
            if cid not in chunk_by_id:
                return False, f"referenced chunk_id {cid} not in provided evidence"
        ref_texts = " \n".join(chunk_by_id[cid]["text"] for cid in payload["evidence_chunks"])
        for quote in payload["evidence_quotes"]:
            if not isinstance(quote, str) or not quote.strip():
                return False, "empty evidence quote"
            # Be lenient about whitespace: normalize spaces before substring check.
            norm_quote = " ".join(quote.split())
            norm_ref = " ".join(ref_texts.split())
            if norm_quote not in norm_ref:
                return False, f"quote not found in evidence: {quote[:60]}…"
        if not isinstance(payload["gold_answer"], (str, int, float)):
            return False, "gold_answer must be a single value"
        return True, "ok"

    # ---- one-shot ----

    def generate_one(
        self,
        spec: TaskTypeSpec,
        *,
        company: str,
        year: int,
        report: str,
        slot_index: int,
    ) -> dict | None:
        self.stats.attempts[spec.code] = self.stats.attempts.get(spec.code, 0) + 1
        chunks = self._select_evidence(spec, company=company, year=year)
        if not chunks:
            log.info("No evidence for %s %s %s — skipping", spec.code, company, year)
            self.stats.discards[spec.code] = self.stats.discards.get(spec.code, 0) + 1
            self.stats.discard_reasons.append(f"{spec.code}|{company}|{year}|no_evidence")
            return None

        prompt = USER_PROMPT_TEMPLATE.format(
            task_code=spec.code,
            task_label=spec.label,
            task_description=spec.description,
            company=company,
            year=year,
            chunks_block=self._format_chunks_block(chunks),
            first_chunk_id=chunks[0]["chunk_id"],
        )

        last_err = ""
        for attempt in range(1, self.max_attempts + 1):
            log.info("Generating %s/%s (%s %s, attempt %d)", spec.code, slot_index, company, year, attempt)
            try:
                raw = self._call_claude(prompt)
            except Exception as e:
                last_err = f"api_error:{e}"
                log.warning("Claude call failed (attempt %d): %s", attempt, e)
                continue
            payload = self._extract_json(raw)
            if not payload:
                last_err = "json_parse_failed"
                continue
            ok, reason = self._verify(payload, chunks)
            if not ok:
                last_err = reason
                log.info("Verification failed (attempt %d): %s", attempt, reason)
                continue
            item_id = f"DART_{spec.code}_{slot_index:03d}"
            self._issued_ids.add(item_id)
            self.stats.successes[spec.code] = self.stats.successes.get(spec.code, 0) + 1
            return {
                "id": item_id,
                "question": payload["question"],
                "gold_answer": payload["gold_answer"],
                "gold_answer_unit": payload["gold_answer_unit"],
                "evidence_chunks": payload["evidence_chunks"],
                "evidence_quotes": payload["evidence_quotes"],
                "reasoning_steps": payload["reasoning_steps"],
                "reasoning_hops": int(payload["reasoning_hops"]),
                "task_type": spec.code,
                "source_company": company,
                "source_report": report,
                "source_section": chunks[0]["metadata"].get("section"),
                "source_pages": None,
                "generation_notes": payload.get("generation_notes", ""),
            }

        log.warning("Discarding %s slot %d after %d attempts: %s",
                    spec.code, slot_index, self.max_attempts, last_err)
        self.stats.discards[spec.code] = self.stats.discards.get(spec.code, 0) + 1
        self.stats.discard_reasons.append(f"{spec.code}|slot{slot_index}|{last_err}")
        return None

    # ---- full pass ----

    def generate_all(
        self,
        distribution: dict[str, int],
        *,
        company_year_pool: list[tuple[str, int, str]],
    ) -> list[dict]:
        """Generate items across all task types until each quota is filled."""
        # Round-robin the pool so distribution stays diverse.
        self._rng.shuffle(company_year_pool)
        per_cy_count: dict[tuple[str, int], int] = {}
        results: list[dict] = []

        for code, quota in distribution.items():
            spec = TASK_TYPES[code]
            produced = 0
            attempt_budget = quota * (self.max_attempts + 2)  # leave headroom for skips
            pool_idx = 0
            tries = 0
            while produced < quota and tries < attempt_budget:
                tries += 1
                if not company_year_pool:
                    break
                company, year, report = company_year_pool[pool_idx % len(company_year_pool)]
                pool_idx += 1
                if per_cy_count.get((company, year), 0) >= self.max_per_company_year:
                    continue
                slot_index = produced + 1
                item = self.generate_one(
                    spec, company=company, year=year,
                    report=report, slot_index=slot_index,
                )
                if item is not None:
                    results.append(item)
                    per_cy_count[(company, year)] = per_cy_count.get((company, year), 0) + 1
                    produced += 1
            if produced < quota:
                log.warning("Quota for %s unmet: produced %d / %d", code, produced, quota)
        return results
