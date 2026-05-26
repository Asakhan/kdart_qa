#!/usr/bin/env python3
"""Phase 2.5 — human review CLI.

Per-item walk through Phase-2 drafts with A/E/R/D/N/P/S/? actions, evidence
highlighting, autosave, and a changes.log audit trail.

Usage:
  python scripts/05_review_session.py                      # first-pass review
  python scripts/05_review_session.py --filter T4          # only T4 items
  python scripts/05_review_session.py --review-mode second-pass
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rich.console import Console  # noqa: E402
from rich.markdown import Markdown  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.prompt import Confirm, Prompt  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.question_generator import (  # noqa: E402
    TASK_TYPES,
    QuestionGenerator,
)
from src.rag_index import OpenAIEmbedder, RagIndex  # noqa: E402
from src.review_helper import ReviewSession, diff_two_passes  # noqa: E402

log = get_logger("review_session", "phase25_review.log")
console = Console()

DART_VIEWER_URL = "https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--filter", dest="type_filter", choices=list(TASK_TYPES.keys()),
                   help="Only walk items of the given task type.")
    p.add_argument("--review-mode", choices=["first-pass", "second-pass"], default="first-pass",
                   help="In second-pass mode, the previous reviewed file is copied to *_first.json "
                        "and a fresh review is started; on exit a diff is written.")
    p.add_argument("--drafts", default=None, help="Override drafts JSON path.")
    return p.parse_args()


def load_rcept_map(manifest_path: Path) -> dict[tuple[str, int], str]:
    if not manifest_path.exists():
        return {}
    rows = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        (r["company"], r["year"]): r.get("rcept_no", "")
        for r in rows
        if r.get("rcept_no")
    }


def load_chunk_index(chunks_dir: Path) -> dict[str, dict]:
    """Read all chunk JSONLs into an id-keyed dict for evidence lookup."""
    out: dict[str, dict] = {}
    for path in sorted(chunks_dir.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out[row["chunk_id"]] = row
    return out


def highlight_quotes(chunk_text: str, quotes: list[str]) -> Text:
    """Return a rich Text with evidence_quotes highlighted yellow."""
    text = Text(chunk_text)
    for q in quotes:
        if not q:
            continue
        # Highlight first occurrence of each (case-sensitive, normalized spaces don't matter visually).
        start = 0
        while True:
            idx = chunk_text.find(q, start)
            if idx < 0:
                break
            text.stylize("bold yellow on grey15", idx, idx + len(q))
            start = idx + max(len(q), 1)
    return text


def render_item(
    state, *, position: int, total: int, chunk_index: dict[str, dict],
    rcept_map: dict[tuple[str, int], str],
) -> None:
    item = state.item
    header = (
        f"[ 검수 진행: {position + 1}/{total} | 유형: {item['task_type']} | "
        f"ID: {item['id']} | 상태: {state.status} ]"
    )
    console.rule(header)

    console.print(Panel(Text(item["question"], style="bold white"), title="QUESTION"))
    gold = f"{item['gold_answer']}"
    if item.get("gold_answer_unit"):
        gold += f"  ({item['gold_answer_unit']})"
    console.print(Panel(Text(gold, style="bold green"), title="GOLD ANSWER"))

    steps = "\n".join(item.get("reasoning_steps", []))
    console.print(Panel(Text(steps), title=f"REASONING STEPS (hops={item.get('reasoning_hops')})"))

    src_line = (
        f"출처: {item.get('source_company')} {item.get('source_report')} "
        f"section={item.get('source_section')}"
    )
    rcept = rcept_map.get((item.get("source_company"), int(str(item.get("source_report", "0_x"))[:4])
                           if str(item.get("source_report", ""))[:4].isdigit() else 0))
    if rcept:
        src_line += f"\nDART: {DART_VIEWER_URL.format(rcept_no=rcept)}"
    console.print(Panel(Text(src_line), title="SOURCE"))

    for i, cid in enumerate(item.get("evidence_chunks", []), start=1):
        chunk = chunk_index.get(cid)
        if not chunk:
            console.print(Panel(Text(f"(chunk {cid} missing from data/chunks)"),
                                title=f"Chunk {i} — MISSING"))
            continue
        title = (f"Chunk {i} — {cid} | section={chunk.get('section')} "
                 f"| has_table={chunk.get('has_table')}")
        body = highlight_quotes(chunk["text"], item.get("evidence_quotes", []))
        console.print(Panel(body, title=title))

    if item.get("generation_notes"):
        console.print(Panel(Text(item["generation_notes"], style="italic dim"),
                            title="GENERATION NOTES"))
    if state.discard_reason:
        console.print(Panel(Text(state.discard_reason, style="red"),
                            title="이전 폐기 사유"))
    if state.regenerate_reason:
        console.print(Panel(Text(state.regenerate_reason, style="cyan"),
                            title="재생성 사유"))


def print_help() -> None:
    table = Table(title="검수 단축키")
    table.add_column("키"); table.add_column("동작"); table.add_column("설명")
    table.add_row("A", "Accept", "초안을 그대로 채택")
    table.add_row("E", "Edit",   "$EDITOR로 JSON 직접 수정")
    table.add_row("R", "Regenerate", "동일 evidence로 Claude 재생성 (사유 입력)")
    table.add_row("D", "Discard", "폐기 (사유 입력)")
    table.add_row("N", "Next",   "다음 문항 (저장 없음)")
    table.add_row("P", "Prev",   "이전 문항")
    table.add_row("J", "Jump",   "특정 ID 또는 인덱스로 이동")
    table.add_row("S", "Save & Quit", "저장 후 종료")
    table.add_row("?", "Help",   "이 도움말 표시")
    console.print(table)


def maybe_regenerate(session: ReviewSession, cfg: dict, chunk_index: dict[str, dict]) -> None:
    s = session.current()
    if not s:
        return
    reason = Prompt.ask("재생성 사유")
    code = s.item["task_type"]
    spec = TASK_TYPES.get(code)
    if not spec:
        console.print(f"[red]Unknown task_type {code}[/]")
        return

    # Reconstruct an evidence list from the chunks the previous draft used.
    evidence: list[dict] = []
    for cid in s.item.get("evidence_chunks", []):
        chunk = chunk_index.get(cid)
        if chunk:
            evidence.append({
                "chunk_id": cid,
                "text": chunk["text"],
                "metadata": chunk,
                "distance": None,
            })
    if not evidence:
        console.print("[red]원본 evidence가 data/chunks에 없습니다. 재생성 불가.[/]")
        return

    # Spin up a generator that bypasses re-selection — call its prompt/verify pipeline directly.
    index = RagIndex(
        persist_dir=project_path(cfg["paths"]["index"]),
        collection_name=cfg["rag"]["collection_name"],
        embedder=OpenAIEmbedder(model=cfg["rag"]["model"]),
    )
    gen = QuestionGenerator(
        index=index,
        model=cfg["generation"]["drafter_model"],
        max_tokens=cfg["generation"]["drafter_max_tokens"],
        max_attempts=cfg["generation"]["max_regenerate_attempts"],
        max_per_company_year=cfg["generation"]["max_per_company_year"],
    )
    from src.question_generator import USER_PROMPT_TEMPLATE
    prompt = USER_PROMPT_TEMPLATE.format(
        task_code=spec.code,
        task_label=spec.label,
        task_description=spec.description + f"\n\n[재생성 사유: {reason}]",
        company=s.item["source_company"],
        year=int(str(s.item["source_report"])[:4]) if str(s.item.get("source_report", ""))[:4].isdigit() else 0,
        chunks_block=gen._format_chunks_block(evidence),
        first_chunk_id=evidence[0]["chunk_id"],
    )
    raw = gen._call_openai(prompt)
    payload = gen._extract_json(raw)
    if not payload:
        console.print("[red]OpenAI 응답을 JSON으로 파싱하지 못했습니다.[/]")
        return
    ok, why = gen._verify(payload, evidence)
    if not ok:
        console.print(f"[red]자체 검증 실패: {why}[/]")
        return
    new_item = dict(s.item)  # preserve id, source meta
    for k in (
        "question", "gold_answer", "gold_answer_unit",
        "evidence_chunks", "evidence_quotes",
        "reasoning_steps", "generation_notes",
    ):
        if k in payload:
            new_item[k] = payload[k]
    new_item["reasoning_hops"] = int(payload["reasoning_hops"])
    session.mark_regenerated(reason, new_item)
    console.print("[green]Regenerate 완료. 새 초안이 표시됩니다.[/]")


def render_stats(session: ReviewSession) -> None:
    s = session.stats()
    console.rule("검수 세션 종료")
    console.print(f"누적 검수: {sum(c for _, c in s['by_type_progress'].values()) - s['by_status'].get('pending', 0)}/{s['total']}")
    for status in ("accepted", "edited", "discarded", "pending"):
        cnt = s["by_status"].get(status, 0)
        pct = (cnt / s["total"] * 100) if s["total"] else 0.0
        console.print(f"  {status:<10}: {cnt:>3}  ({pct:.1f}%)")
    console.print("유형별 진행:")
    for code, (done, tot) in sorted(s["by_type_progress"].items()):
        console.print(f"  {code}: {done}/{tot}")
    console.print(f"저장 위치: {s['reviewed_path']}")


def main() -> int:
    args = parse_args()
    cfg = load_config()

    drafts_path = Path(args.drafts) if args.drafts else project_path(cfg["paths"]["drafts"]) / cfg["generation"]["draft_filename"]
    reviewed_dir = ensure_dir(project_path(cfg["paths"]["reviewed"]))
    reviewed_path = reviewed_dir / cfg["review"]["reviewed_filename"]
    changes_log = reviewed_dir / cfg["review"]["changes_log"]

    if not drafts_path.exists():
        console.print(f"[red]Drafts file not found: {drafts_path}[/]")
        return 1

    second_pass = args.review_mode == "second-pass"
    if second_pass and reviewed_path.exists():
        first_pass_archive = reviewed_dir / "kdart_qa_reviewed_first.json"
        if not first_pass_archive.exists():
            reviewed_path.replace(first_pass_archive)
            console.print(f"[yellow]1차 검수본을 {first_pass_archive.name}으로 보존했습니다.[/]")

    session = ReviewSession(
        drafts_path=drafts_path,
        reviewed_path=reviewed_path,
        changes_log=changes_log,
        type_filter=args.type_filter,
    )
    chunk_index = load_chunk_index(project_path(cfg["paths"]["chunks"]))
    rcept_map = load_rcept_map(project_path(cfg["paths"]["raw"]) / "manifest.json")

    if not session.states:
        console.print("[yellow]No items to review (check --filter or run Phase 2 first).[/]")
        return 0

    while True:
        state = session.current()
        if not state:
            break
        render_item(state, position=session.cursor, total=len(session.states),
                    chunk_index=chunk_index, rcept_map=rcept_map)
        choice = Prompt.ask(
            "Action [A]ccept [E]dit [R]egenerate [D]iscard [N]ext [P]rev [J]ump [S]ave&Quit [?]",
            choices=["a", "e", "r", "d", "n", "p", "j", "s", "?"],
            default="n",
            show_choices=False,
        ).lower()
        if choice == "a":
            session.accept()
            session.goto_next_pending()
        elif choice == "e":
            ok, note = session.edit_with_editor()
            console.print(f"[green]Edit saved.[/] {note}" if ok else f"[red]Edit aborted.[/] {note}")
        elif choice == "r":
            maybe_regenerate(session, cfg, chunk_index)
        elif choice == "d":
            reason = Prompt.ask("폐기 사유 (필수)")
            if reason.strip():
                session.discard(reason)
                session.goto_next_pending()
            else:
                console.print("[red]사유가 비어 있어 폐기 취소.[/]")
        elif choice == "n":
            session.move(1)
        elif choice == "p":
            session.move(-1)
        elif choice == "j":
            target = Prompt.ask("이동할 ID 또는 인덱스(1-base)").strip()
            if target.isdigit():
                session.cursor = max(0, min(len(session.states) - 1, int(target) - 1))
            else:
                for i, st in enumerate(session.states):
                    if st.item.get("id") == target:
                        session.cursor = i
                        break
        elif choice == "s":
            session.save()
            break
        elif choice == "?":
            print_help()

    render_stats(session)

    if second_pass:
        first_pass_archive = reviewed_dir / "kdart_qa_reviewed_first.json"
        if first_pass_archive.exists():
            diff_path = reviewed_dir / "second_pass_diff.log"
            n = diff_two_passes(first_pass_archive, reviewed_path, diff_path)
            console.print(f"[cyan]2차 검수 차이 {n}건 → {diff_path}[/]")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        console.print("\n[yellow]중단됨. 마지막 액션까지의 진행은 이미 저장되었습니다.[/]")
        raise SystemExit(130)
