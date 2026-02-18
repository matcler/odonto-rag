#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

_VOLATILE_KEYS = {"timestamp", "run_id", "generated_at", "thresholds"}


def _load_json(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"FAIL: file not found: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"FAIL: root must be object in {path}")
    return payload


def _drop_volatile(node: Any) -> Any:
    if isinstance(node, dict):
        out: Dict[str, Any] = {}
        for k, v in node.items():
            if k in _VOLATILE_KEYS:
                continue
            out[k] = _drop_volatile(v)
        return out
    if isinstance(node, list):
        return [_drop_volatile(x) for x in node]
    return node


def _sort_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _sort_known_lists(node: Any) -> Any:
    if isinstance(node, dict):
        out = {k: _sort_known_lists(v) for k, v in node.items()}
        if isinstance(out.get("warnings"), list):
            out["warnings"] = sorted(out["warnings"], key=_sort_key)
        if isinstance(out.get("hard_failures"), list):
            out["hard_failures"] = sorted(out["hard_failures"], key=_sort_key)
        metrics = out.get("metrics")
        if isinstance(metrics, dict):
            source_div = metrics.get("source_diversity")
            if isinstance(source_div, dict) and isinstance(source_div.get("unique_docs_per_slide"), list):
                source_div["unique_docs_per_slide"] = sorted(
                    source_div["unique_docs_per_slide"],
                    key=lambda x: (int((x or {}).get("slide_index", 0) or 0), int((x or {}).get("doc_count", 0) or 0)),
                )
            read = metrics.get("readability")
            if isinstance(read, dict) and isinstance(read.get("table_shapes"), list):
                read["table_shapes"] = sorted(
                    read["table_shapes"],
                    key=lambda x: (
                        int((x or {}).get("slide_index", 0) or 0),
                        str((x or {}).get("asset_id") or ""),
                        int((x or {}).get("rows", 0) or 0),
                        int((x or {}).get("cols", 0) or 0),
                    ),
                )
            cov = metrics.get("coverage")
            if isinstance(cov, dict) and isinstance(cov.get("off_topic_slides"), list):
                cov["off_topic_slides"] = sorted(int(v or 0) for v in cov["off_topic_slides"])
        return out
    if isinstance(node, list):
        return [_sort_known_lists(x) for x in node]
    return node


def normalize_quality(payload: Dict[str, Any]) -> Dict[str, Any]:
    return _sort_known_lists(_drop_volatile(payload))


def normalized_sha256(payload: Dict[str, Any]) -> str:
    norm = normalize_quality(payload)
    raw = json.dumps(norm, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def semantic_diff(old: Dict[str, Any], new: Dict[str, Any]) -> List[str]:
    changes: List[str] = []
    if old.get("deck_id") != new.get("deck_id"):
        changes.append("deck_id changed")
    old_counts = old.get("counts") if isinstance(old.get("counts"), dict) else {}
    new_counts = new.get("counts") if isinstance(new.get("counts"), dict) else {}
    for key in ("slides", "bullets", "evidence_items", "visual_assets", "missing_asset_count_summary"):
        if old_counts.get(key) != new_counts.get(key):
            changes.append(f"counts.{key} changed: {old_counts.get(key)} -> {new_counts.get(key)}")
    old_gate = old.get("gate") if isinstance(old.get("gate"), dict) else {}
    new_gate = new.get("gate") if isinstance(new.get("gate"), dict) else {}
    if bool(old_gate.get("pass_hard")) != bool(new_gate.get("pass_hard")):
        changes.append("gate.pass_hard changed")
    if old_gate.get("hard_failures") != new_gate.get("hard_failures"):
        changes.append("gate.hard_failures changed")
    if old.get("warnings") != new.get("warnings"):
        changes.append("warnings changed")
    return changes


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic semantic diff for quality.json outputs")
    ap.add_argument("old", nargs="?", help="Old quality JSON")
    ap.add_argument("new", nargs="?", help="New quality JSON")
    ap.add_argument("--sha256", default="", help="Print normalized SHA256 for one quality JSON and exit")
    ap.add_argument("--fail-on-diff", action="store_true", help="Exit 1 when semantic or normalized diff is detected")
    args = ap.parse_args()

    if args.sha256:
        payload = _load_json(args.sha256)
        print(normalized_sha256(payload))
        return

    if not args.old or not args.new:
        raise SystemExit("FAIL: provide OLD NEW quality JSON paths (or --sha256 <file>)")

    old_raw = _load_json(args.old)
    new_raw = _load_json(args.new)
    old_norm = normalize_quality(old_raw)
    new_norm = normalize_quality(new_raw)
    old_sha = normalized_sha256(old_raw)
    new_sha = normalized_sha256(new_raw)

    print(f"old_sha256={old_sha}")
    print(f"new_sha256={new_sha}")

    changes = semantic_diff(old_norm, new_norm)
    if not changes and old_norm != new_norm:
        changes.append("normalized quality changed (non-semantic fields)")

    if changes:
        print("DIFF")
        for c in changes:
            print(f"- {c}")
        if args.fail_on_diff:
            raise SystemExit(1)
    else:
        print("NO_DIFF")


if __name__ == "__main__":
    main()
