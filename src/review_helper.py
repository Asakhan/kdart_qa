"""Phase 2.5: state + helpers for the human review CLI.

The review CLI is a per-item walk over Phase-2 drafts. This module owns:
  * loading/saving the reviewed JSON
  * the append-only changes.log
  * external-editor invocation
  * a "second pass" diff helper for re-review sessions
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .common import ensure_dir, get_logger

log = get_logger("review_helper")

REVIEW_STATUSES = {"pending", "accepted", "edited", "discarded"}


@dataclass
class ReviewState:
    """Persistent review state for a single item."""
    item: dict
    status: str = "pending"   # pending | accepted | edited | discarded
    discard_reason: str | None = None
    regenerate_reason: str | None = None
    reviewer_notes: str = ""
    last_touched: str | None = None  # ISO timestamp

    def to_dict(self) -> dict:
        return {
            "item": self.item,
            "status": self.status,
            "discard_reason": self.discard_reason,
            "regenerate_reason": self.regenerate_reason,
            "reviewer_notes": self.reviewer_notes,
            "last_touched": self.last_touched,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ReviewState":
        return cls(
            item=d["item"],
            status=d.get("status", "pending"),
            discard_reason=d.get("discard_reason"),
            regenerate_reason=d.get("regenerate_reason"),
            reviewer_notes=d.get("reviewer_notes", ""),
            last_touched=d.get("last_touched"),
        )


class ReviewSession:
    def __init__(
        self,
        *,
        drafts_path: Path,
        reviewed_path: Path,
        changes_log: Path,
        type_filter: str | None = None,
    ) -> None:
        self.drafts_path = drafts_path
        self.reviewed_path = reviewed_path
        self.changes_log = changes_log
        ensure_dir(reviewed_path.parent)
        ensure_dir(changes_log.parent)

        drafts = json.loads(drafts_path.read_text(encoding="utf-8"))
        prior = self._load_prior_state()
        self.states: list[ReviewState] = []
        prior_by_id = {p["item"]["id"]: p for p in prior}
        for draft in drafts:
            if type_filter and draft.get("task_type") != type_filter:
                continue
            if draft["id"] in prior_by_id:
                self.states.append(ReviewState.from_dict(prior_by_id[draft["id"]]))
            else:
                self.states.append(ReviewState(item=draft))

        self.cursor = self._first_pending_index()

    # ---- IO ----

    def _load_prior_state(self) -> list[dict]:
        if not self.reviewed_path.exists():
            return []
        try:
            data = json.loads(self.reviewed_path.read_text(encoding="utf-8"))
            return data.get("states", data)  # support both wrapper and bare-list formats
        except json.JSONDecodeError as e:
            log.warning("Could not parse prior reviewed file (%s); starting fresh", e)
            return []

    def save(self) -> None:
        payload = {
            "saved_at": datetime.now().isoformat(timespec="seconds"),
            "states": [s.to_dict() for s in self.states],
            # convenience: also emit the accepted/edited items as a flat list
            "items": [s.item for s in self.states if s.status in {"accepted", "edited"}],
        }
        # atomic write
        tmp = self.reviewed_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.reviewed_path)

    def log_change(self, item_id: str, action: str, note: str = "") -> None:
        line = f"{datetime.now():%Y-%m-%d %H:%M:%S} | {item_id} | {action} | {note}\n"
        with self.changes_log.open("a", encoding="utf-8") as f:
            f.write(line)

    # ---- navigation ----

    def _first_pending_index(self) -> int:
        for i, s in enumerate(self.states):
            if s.status == "pending":
                return i
        return 0

    def current(self) -> ReviewState | None:
        if not self.states:
            return None
        return self.states[self.cursor]

    def move(self, delta: int) -> None:
        if not self.states:
            return
        self.cursor = max(0, min(len(self.states) - 1, self.cursor + delta))

    def goto_next_pending(self) -> None:
        n = len(self.states)
        for k in range(1, n + 1):
            idx = (self.cursor + k) % n
            if self.states[idx].status == "pending":
                self.cursor = idx
                return

    # ---- actions ----

    def accept(self) -> None:
        s = self.current()
        if not s:
            return
        s.status = "accepted"
        s.last_touched = datetime.now().isoformat(timespec="seconds")
        self.log_change(s.item["id"], "ACCEPT")
        self.save()

    def discard(self, reason: str) -> None:
        s = self.current()
        if not s:
            return
        s.status = "discarded"
        s.discard_reason = reason
        s.last_touched = datetime.now().isoformat(timespec="seconds")
        self.log_change(s.item["id"], "DISCARD", reason)
        self.save()

    def edit_with_editor(self) -> tuple[bool, str]:
        """Open the current item in $EDITOR; return (changed, diff_summary)."""
        s = self.current()
        if not s:
            return False, ""
        editor = os.environ.get("EDITOR") or shutil.which("vim") or shutil.which("nano")
        if not editor:
            return False, "no $EDITOR available"
        before = json.dumps(s.item, ensure_ascii=False, indent=2, sort_keys=True)
        with tempfile.NamedTemporaryFile("w+", suffix=".json", encoding="utf-8", delete=False) as tf:
            tf.write(before)
            tmp_path = Path(tf.name)
        try:
            subprocess.call([editor, str(tmp_path)])
            after = tmp_path.read_text(encoding="utf-8")
            try:
                parsed = json.loads(after)
            except json.JSONDecodeError as e:
                return False, f"JSON parse failed; edit rolled back ({e})"
            changed_fields = _diff_fields(s.item, parsed)
            s.item = parsed
            s.status = "edited"
            s.last_touched = datetime.now().isoformat(timespec="seconds")
            note = "; ".join(changed_fields) if changed_fields else "no field changed"
            self.log_change(parsed.get("id", "?"), "EDIT", note)
            self.save()
            return True, note
        finally:
            tmp_path.unlink(missing_ok=True)

    def mark_regenerated(self, reason: str, new_item: dict) -> None:
        s = self.current()
        if not s:
            return
        s.item = new_item
        s.status = "pending"   # needs another look
        s.regenerate_reason = reason
        s.last_touched = datetime.now().isoformat(timespec="seconds")
        self.log_change(new_item.get("id", "?"), "REGENERATE", reason)
        self.save()

    # ---- stats ----

    def stats(self) -> dict[str, Any]:
        total = len(self.states)
        by_status: dict[str, int] = {}
        for s in self.states:
            by_status[s.status] = by_status.get(s.status, 0) + 1
        by_type_progress: dict[str, tuple[int, int]] = {}
        for s in self.states:
            code = s.item.get("task_type", "?")
            done, tot = by_type_progress.get(code, (0, 0))
            tot += 1
            if s.status != "pending":
                done += 1
            by_type_progress[code] = (done, tot)
        return {
            "total": total,
            "by_status": by_status,
            "by_type_progress": by_type_progress,
            "reviewed_path": str(self.reviewed_path),
        }


def _diff_fields(before: dict, after: dict) -> list[str]:
    changed = []
    keys = set(before.keys()) | set(after.keys())
    for k in sorted(keys):
        if before.get(k) != after.get(k):
            changed.append(k)
    return changed


# ---- second-pass diff ----

def diff_two_passes(first_path: Path, second_path: Path, out_log: Path) -> int:
    """Compare statuses + items across two reviewed JSONs and write a delta log."""
    first = json.loads(first_path.read_text(encoding="utf-8"))
    second = json.loads(second_path.read_text(encoding="utf-8"))
    first_by_id = {s["item"]["id"]: s for s in first.get("states", [])}
    second_by_id = {s["item"]["id"]: s for s in second.get("states", [])}
    n_diff = 0
    with out_log.open("w", encoding="utf-8") as f:
        f.write("# second-pass diff\n\n")
        for item_id in sorted(set(first_by_id) | set(second_by_id)):
            a = first_by_id.get(item_id)
            b = second_by_id.get(item_id)
            if not a or not b:
                f.write(f"{item_id}: only in {'first' if a else 'second'}\n")
                n_diff += 1
                continue
            if a["status"] != b["status"]:
                f.write(f"{item_id}: status {a['status']} → {b['status']}\n")
                n_diff += 1
            field_changes = _diff_fields(a["item"], b["item"])
            if field_changes:
                f.write(f"{item_id}: fields changed: {field_changes}\n")
                n_diff += 1
    return n_diff
