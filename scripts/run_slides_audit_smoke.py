#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, Optional

import requests


def _load_plan(path: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"FAIL: plan file not found: {path}")
    try:
        payload = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON in {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit("FAIL: plan root must be a JSON object")
    return payload


def _infer_specialty(plan: Dict[str, Any]) -> Optional[str]:
    req = plan.get("request") if isinstance(plan.get("request"), dict) else {}
    specialty = str(req.get("specialty") or "").strip()
    return specialty or None


def _run_cmd(cmd: list[str]) -> None:
    proc = subprocess.run(cmd, text=True)
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def _build_pptx_via_api(
    *,
    api_base: str,
    plan: Dict[str, Any],
    filename: Optional[str],
    evidence_bundle: bool,
    timeout_seconds: int,
) -> Dict[str, Any]:
    url = api_base.rstrip("/") + "/rag/slides/pptx"
    body: Dict[str, Any] = {"slide_plan": plan, "evidence_bundle": evidence_bundle}
    if filename:
        body["filename"] = filename
    try:
        resp = requests.post(url, json=body, timeout=timeout_seconds)
    except Exception as exc:
        raise SystemExit(f"FAIL: cannot reach {url}: {exc}") from exc
    if resp.status_code >= 400:
        raise SystemExit(f"FAIL: pptx endpoint returned {resp.status_code}: {resp.text[:500]}")
    try:
        obj = resp.json()
    except Exception as exc:
        raise SystemExit(f"FAIL: invalid JSON response from pptx endpoint: {exc}") from exc
    if not isinstance(obj, dict):
        raise SystemExit("FAIL: pptx endpoint response must be a JSON object")
    return obj


def _build_pptx_offline(
    *,
    plan: Dict[str, Any],
    out_root: str,
    test_name: str,
    filename: Optional[str],
    evidence_bundle: bool,
) -> Dict[str, Any]:
    repo_root = Path(__file__).resolve().parent.parent
    src_dir = repo_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))

    try:
        from odonto_rag.deck.pptx_builder import build_pptx_from_slide_plan
        from odonto_rag.api.rag_app import (  # type: ignore
            _build_deck_audit_manifest,
            _json_dumps_deterministic,
            _write_evidence_bundle,
        )
    except Exception as exc:
        raise SystemExit(f"FAIL: offline mode import error: {exc}") from exc

    safe_name = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in (test_name or "smoke"))
    safe_name = safe_name.strip("._") or "smoke"
    out_dir = Path(out_root) / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)

    target_filename = (filename or "deck.pptx").strip()
    if not target_filename.lower().endswith(".pptx"):
        target_filename += ".pptx"

    out_path = build_pptx_from_slide_plan(plan, out_dir=str(out_dir), filename=target_filename)
    deck_id = out_path.stem
    audit_payload = _build_deck_audit_manifest(plan, deck_id=deck_id, deck_path=out_path)
    audit_path = out_path.with_suffix(".audit.json")
    audit_path.write_text(_json_dumps_deterministic(audit_payload), encoding="utf-8")

    bundle_path: Optional[Path] = None
    if evidence_bundle:
        bundle_path = _write_evidence_bundle(
            bundle_path=out_path.with_suffix(".evidence.zip"),
            audit_payload=audit_payload,
            slide_plan=plan,
        )

    return {
        "path": str(out_path),
        "audit_path": str(audit_path),
        "evidence_bundle_path": str(bundle_path) if bundle_path else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="One-shot smoke runner: plan grounding check -> /rag/slides/pptx -> S6 audit check."
    )
    ap.add_argument("--plan-json", required=True, help="Path to /rag/slides/plan response JSON")
    ap.add_argument("--api-base", default="http://127.0.0.1:8000", help="API base URL")
    ap.add_argument("--specialty", default="", help="Expected specialty (optional)")
    ap.add_argument("--filename", default="", help="Optional target PPTX filename")
    ap.add_argument("--evidence-bundle", action="store_true", help="Request evidence bundle zip export")
    ap.add_argument("--allow-missing-assets", action="store_true", help="Allow missing assets in audit policy")
    ap.add_argument("--skip-plan-check", action="store_true", help="Skip S5 plan grounding script")
    ap.add_argument("--offline", action="store_true", help="Offline mode: local build without API call")
    ap.add_argument("--out-root", default="out/tests", help="Output root for --offline mode")
    ap.add_argument("--test-name", default="", help="Subfolder name under --out-root for --offline mode")
    ap.add_argument("--timeout", type=int, default=240, help="HTTP timeout seconds for /rag/slides/pptx")
    args = ap.parse_args()

    plan = _load_plan(args.plan_json)
    specialty = args.specialty.strip() or (_infer_specialty(plan) or "")

    if not args.skip_plan_check and specialty:
        _run_cmd(
            [
                sys.executable,
                "scripts/check_slides_plan_grounding.py",
                "--plan-json",
                args.plan_json,
                "--specialty",
                specialty,
            ]
        )
    elif not args.skip_plan_check and not specialty:
        print("WARN: specialty not provided/inferred, skipping S5 specialty plan check")

    if args.offline:
        test_name = args.test_name.strip() or Path(args.plan_json).stem.replace(".plan", "")
        pptx_resp = _build_pptx_offline(
            plan=plan,
            out_root=args.out_root,
            test_name=test_name,
            filename=(args.filename.strip() or "deck.pptx"),
            evidence_bundle=bool(args.evidence_bundle),
        )
    else:
        pptx_resp = _build_pptx_via_api(
            api_base=args.api_base,
            plan=plan,
            filename=(args.filename.strip() or None),
            evidence_bundle=bool(args.evidence_bundle),
            timeout_seconds=max(10, args.timeout),
        )

    pptx_path = str(pptx_resp.get("path") or "").strip()
    audit_path = str(pptx_resp.get("audit_path") or "").strip()
    bundle_path = str(pptx_resp.get("evidence_bundle_path") or "").strip()

    if not pptx_path or not Path(pptx_path).exists():
        raise SystemExit(f"FAIL: missing PPTX output path from endpoint: {pptx_path or '<empty>'}")
    if not audit_path or not Path(audit_path).exists():
        raise SystemExit(f"FAIL: missing audit output path from endpoint: {audit_path or '<empty>'}")
    if args.evidence_bundle and (not bundle_path or not Path(bundle_path).exists()):
        raise SystemExit(f"FAIL: evidence bundle requested but missing: {bundle_path or '<empty>'}")

    audit_cmd = [sys.executable, "scripts/check_slides_audit_pack.py", "--audit-json", audit_path]
    if specialty:
        audit_cmd.extend(["--specialty", specialty])
    if args.allow_missing_assets:
        audit_cmd.append("--allow-missing-assets")
    _run_cmd(audit_cmd)

    print("PASS")
    print(f"pptx={pptx_path}")
    print(f"audit={audit_path}")
    if bundle_path:
        print(f"bundle={bundle_path}")


if __name__ == "__main__":
    main()
