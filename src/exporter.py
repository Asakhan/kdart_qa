"""Phase 3-2: assemble the final K-DART-QA from reviewed + calibration outputs."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from .common import ensure_dir, get_logger

log = get_logger("exporter")


def load_calibration(results_path: Path) -> dict[str, dict]:
    rows = json.loads(results_path.read_text(encoding="utf-8"))
    return {r["item_id"]: r for r in rows}


def load_reviewed(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if "items" in data:
        return data["items"]
    if "states" in data:
        return [s["item"] for s in data["states"] if s.get("status") in {"accepted", "edited"}]
    raise ValueError("Unrecognized reviewed file shape")


def select_accepted(items: list[dict], calib: dict[str, dict]) -> list[dict]:
    """Keep items whose calibration verdict is 'accepted'."""
    out = []
    for it in items:
        cr = calib.get(it["id"])
        if cr and cr["classification"] == "accepted":
            enriched = dict(it)
            enriched["calibration"] = {
                "classification": cr["classification"],
                "n_correct": sum(1 for a in cr["attempts"] if a["correct"]),
                "n_attempts": len(cr["attempts"]),
                "retrieved_chunk_ids": cr["retrieved_chunk_ids"],
            }
            out.append(enriched)
    return out


def write_jsonl(items: list[dict], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def write_parquet(items: list[dict], path: Path) -> None:
    # nested fields are kept as-is via pyarrow's struct/list mapping
    df = pd.DataFrame(items)
    df.to_parquet(path, index=False)


# ---- changes.log parsing for "edit rate" ----

_LOG_LINE = re.compile(
    r"^(?P<ts>\S+ \S+)\s+\|\s+(?P<id>\S+)\s+\|\s+(?P<action>\S+)\s+\|\s*(?P<note>.*)$"
)


def edit_rate_from_log(log_path: Path, n_total: int) -> dict[str, float | int]:
    if not log_path.exists() or n_total == 0:
        return {"accept": 0, "edit": 0, "regenerate": 0, "discard": 0, "edit_rate": 0.0}
    counts: Counter = Counter()
    for line in log_path.read_text(encoding="utf-8").splitlines():
        m = _LOG_LINE.match(line.strip())
        if m:
            counts[m.group("action").lower()] += 1
    return {
        "accept": counts.get("accept", 0),
        "edit": counts.get("edit", 0),
        "regenerate": counts.get("regenerate", 0),
        "discard": counts.get("discard", 0),
        "edit_rate": counts.get("edit", 0) / n_total,
    }


# ---- statistics.md ----

def build_statistics_md(
    final_items: list[dict],
    *,
    reviewed_count: int,
    calib_summary: dict[str, int],
    edits: dict[str, float | int],
) -> str:
    types = Counter(i["task_type"] for i in final_items)
    hops = Counter(i["reasoning_hops"] for i in final_items)
    cys = Counter((i["source_company"], str(i.get("source_report", ""))) for i in final_items)
    avg_ev = (
        sum(len(i["evidence_chunks"]) for i in final_items) / max(len(final_items), 1)
    )

    out: list[str] = []
    out.append("# K-DART-QA Statistics\n")
    out.append(f"- Final items: **{len(final_items)}**")
    out.append(f"- Reviewed items considered: {reviewed_count}")
    out.append(f"- Calibration pass-rate: {calib_summary.get('accepted', 0)}/{sum(calib_summary.values())}"
               f"  (too_easy={calib_summary.get('too_easy', 0)}, too_hard={calib_summary.get('too_hard', 0)})")
    out.append(f"- Mean evidence_chunks per item: {avg_ev:.2f}")
    out.append("")

    out.append("## Type breakdown")
    out.append("| Type | Count |")
    out.append("| --- | ---: |")
    for code in sorted(types):
        out.append(f"| {code} | {types[code]} |")
    out.append("")

    out.append("## reasoning_hops distribution")
    out.append("| Hops | Count |")
    out.append("| ---: | ---: |")
    for h in sorted(hops):
        out.append(f"| {h} | {hops[h]} |")
    out.append("")

    out.append("## Company × report distribution")
    out.append("| Company | Report | Count |")
    out.append("| --- | --- | ---: |")
    for (company, report), c in sorted(cys.items()):
        out.append(f"| {company} | {report} | {c} |")
    out.append("")

    out.append("## Review action log")
    out.append(f"- Accept: {edits['accept']}")
    out.append(f"- Edit: {edits['edit']} (edit rate {edits['edit_rate']:.1%})")
    out.append(f"- Regenerate: {edits['regenerate']}")
    out.append(f"- Discard: {edits['discard']}")
    return "\n".join(out) + "\n"


def summarize_calibration(calib: dict[str, dict]) -> dict[str, int]:
    out: Counter = Counter()
    for r in calib.values():
        out[r["classification"]] += 1
    return dict(out)
