#!/usr/bin/env python3
"""Work 3.5: export K-DART-QA in FinQA's item schema (single JSON array).

Evidence text is embedded in every item (pre_text / post_text / table) so the
downstream experiment grades with no retrieval, no ChromaDB, and no API calls.
This script itself also runs fully offline from data/final/*.jsonl +
data/chunks/*.jsonl — no index build, no embeddings, no LLM.

    python scripts/09_export_finqa_format.py
    python scripts/09_export_finqa_format.py --input data/final/kdart_qa_finqa_matched.jsonl

Output name is auto-derived:
    kdart_qa_gpt_quota.jsonl       → kdart_qa_finqa_format.json
    kdart_qa_finqa_matched.jsonl   → kdart_qa_finqa_matched_finqa_format.json
    <other>.jsonl                  → <other>_finqa_format.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.common import ensure_dir, get_logger, load_config, project_path  # noqa: E402
from src.finqa_adapter import convert_dataset, load_chunk_texts, validate  # noqa: E402

log = get_logger("export_finqa", "phase4_export_finqa.log")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=None,
                   help="Input K-DART JSONL (default: data/final/kdart_qa_gpt_quota.jsonl).")
    p.add_argument("--out", default=None, help="Override output JSON path.")
    p.add_argument("--use-index", action="store_true",
                   help="Fall back to ChromaDB for chunk ids missing from data/chunks "
                        "(needs a built index; otherwise stays fully offline).")
    return p.parse_args()


def derive_out(in_path: Path, final_dir: Path) -> Path:
    if in_path.name == "kdart_qa_gpt_quota.jsonl":
        return final_dir / "kdart_qa_finqa_format.json"
    return final_dir / f"{in_path.stem}_finqa_format.json"


def load_items(path: Path) -> list[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def main() -> int:
    args = parse_args()
    cfg = load_config()
    final_dir = ensure_dir(project_path(cfg["paths"]["final"]))
    chunks_dir = project_path(cfg["paths"]["chunks"])

    in_path = Path(args.input) if args.input else final_dir / "kdart_qa_gpt_quota.jsonl"
    if not in_path.exists():
        log.error("Input not found: %s", in_path)
        return 1
    out_path = Path(args.out) if args.out else derive_out(in_path, final_dir)

    items = load_items(in_path)
    log.info("Loaded %d K-DART items from %s", len(items), in_path)

    chunk_lookup = load_chunk_texts(chunks_dir)
    log.info("Loaded %d chunk texts from %s (offline)", len(chunk_lookup), chunks_dir)

    index_get = None
    if args.use_index:
        from src.rag_index import RagIndex, build_embedder
        index = RagIndex(
            persist_dir=project_path(cfg["paths"]["index"]),
            collection_name=cfg["rag"]["collection_name"],
            embedder=build_embedder(cfg["rag"]),
        )

        def index_get(cid: str):  # noqa: F811
            got = index.collection.get(ids=[cid])
            docs = got.get("documents") or []
            return docs[0] if docs else None

    converted = convert_dataset(items, chunk_lookup, index_get=index_get)

    report = validate(converted)
    log.info("Validation: %d items | %d with table | %d non-rectangular | %d missing exe_ans",
             report["n"], report["with_table"], report["non_rectangular_tables"],
             report["missing_exe_ans"])
    for issue in report["issues"][:20]:
        log.warning("  issue: %s", issue)
    if report["issues"]:
        log.warning("  ...(%d issues total)", len(report["issues"]))

    out_path.write_text(json.dumps(converted, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("Wrote %d FinQA-schema items → %s", len(converted), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
