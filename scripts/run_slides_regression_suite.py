#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple


def _run(cmd: List[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, text=True, capture_output=True)


def _fixture_name(path: Path) -> str:
    name = path.name
    suffix = ".plan.json"
    if name.endswith(suffix):
        return name[: -len(suffix)]
    return path.stem


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid JSON root (expected object): {path}")
    return payload


def _hash_for(audit_path: Path) -> str:
    proc = _run([sys.executable, "scripts/diff_audit_json.py", "--sha256", str(audit_path)])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"hash failed: {audit_path}")
    return proc.stdout.strip().splitlines()[-1].strip()


def _quality_hash_for(quality_path: Path) -> str:
    proc = _run([sys.executable, "scripts/diff_quality_json.py", "--sha256", str(quality_path)])
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"quality hash failed: {quality_path}")
    return proc.stdout.strip().splitlines()[-1].strip()


def _run_smoke_and_quality(
    *,
    plan_path: Path,
    out_root: Path,
    test_name: str,
    quality_fail_on_hard: bool,
    quality_strict_missing_assets: bool,
    args: argparse.Namespace,
) -> Tuple[Optional[Path], Optional[Path], Optional[Dict[str, Any]], Optional[str]]:
    smoke_cmd = [
        sys.executable,
        "scripts/run_slides_audit_smoke.py",
        "--plan-json",
        str(plan_path),
        "--offline",
        "--skip-plan-check",
        "--out-root",
        str(out_root),
        "--test-name",
        test_name,
        "--filename",
        "deck.pptx",
    ]
    smoke = _run(smoke_cmd)
    if smoke.returncode != 0:
        return None, None, None, f"smoke failed\n{smoke.stdout}{smoke.stderr}"

    target_dir = out_root / test_name
    audit_path = target_dir / "deck.audit.json"
    if not audit_path.exists():
        return None, None, None, f"missing audit output {audit_path}"

    quality_cmd = [
        sys.executable,
        "scripts/eval_audit_quality.py",
        "--audit-json",
        str(audit_path),
        "--out-dir",
        str(target_dir),
        "--hard-min-evidence-per-bullet",
        str(args.quality_hard_min_evidence_per_bullet),
        "--warn-max-single-evidence-ratio",
        str(args.quality_warn_max_single_evidence_ratio),
        "--warn-max-fallback-asset-ratio",
        str(args.quality_warn_max_fallback_asset_ratio),
        "--warn-bullets-per-slide-cap",
        str(args.quality_warn_bullets_per_slide_cap),
        "--warn-max-slides-over-bullets-cap",
        str(args.quality_warn_max_slides_over_bullets_cap),
        "--warn-long-bullet-chars",
        str(args.quality_warn_long_bullet_chars),
        "--warn-max-long-bullets",
        str(args.quality_warn_max_long_bullets),
    ]
    if quality_strict_missing_assets:
        quality_cmd.append("--strict-missing-assets")

    quality_eval = _run(quality_cmd)
    quality_path = target_dir / "quality.json"
    if not quality_path.exists():
        return None, None, None, f"quality evaluation failed\n{quality_eval.stdout}{quality_eval.stderr}"

    try:
        quality_obj = _load_json(quality_path)
    except Exception as exc:
        return None, None, None, str(exc)

    gate = quality_obj.get("gate") if isinstance(quality_obj.get("gate"), dict) else {}
    pass_hard = bool(gate.get("pass_hard"))
    if quality_fail_on_hard and not pass_hard:
        return audit_path, quality_path, quality_obj, "quality hard gate failed"

    return audit_path, quality_path, quality_obj, None


def _repair_trigger(quality_obj: Dict[str, Any], warning_threshold: int) -> Tuple[bool, List[str]]:
    gate = quality_obj.get("gate") if isinstance(quality_obj.get("gate"), dict) else {}
    reasons: List[str] = []
    if not bool(gate.get("pass_hard")):
        reasons.append("hard_fail")
    warning_count = int(gate.get("warning_count", 0) or 0)
    if warning_threshold >= 0 and warning_count > warning_threshold:
        reasons.append(f"warning_count>{warning_threshold}")
    return bool(reasons), reasons


def _parse_diff_output(text: str) -> Dict[str, Any]:
    old_sha = ""
    new_sha = ""
    status = "UNKNOWN"
    changes: List[str] = []
    for ln in text.splitlines():
        line = ln.strip()
        if line.startswith("old_sha256="):
            old_sha = line.split("=", 1)[1].strip()
        elif line.startswith("new_sha256="):
            new_sha = line.split("=", 1)[1].strip()
        elif line == "NO_DIFF":
            status = "NO_DIFF"
        elif line == "DIFF":
            status = "DIFF"
        elif line.startswith("- "):
            changes.append(line[2:])
    return {
        "status": status,
        "old_sha256": old_sha or None,
        "new_sha256": new_sha or None,
        "changes": changes,
        "raw": text,
    }


def _run_auto_repair(
    *,
    name: str,
    plan_path: Path,
    args: argparse.Namespace,
    quality_fail_on_hard: bool,
    quality_strict_missing_assets: bool,
) -> Tuple[Optional[Path], Optional[Path], Optional[Dict[str, Any]], Optional[str]]:
    fixture_root = Path(args.out_root) / name
    fixture_root.mkdir(parents=True, exist_ok=True)

    before_audit, before_quality, before_quality_obj, before_err = _run_smoke_and_quality(
        plan_path=plan_path,
        out_root=fixture_root,
        test_name="before",
        quality_fail_on_hard=False,
        quality_strict_missing_assets=quality_strict_missing_assets,
        args=args,
    )
    if before_err:
        return None, None, None, f"{name}: before run failed\n{before_err}"

    assert before_audit is not None and before_quality is not None and before_quality_obj is not None
    should_repair, reasons = _repair_trigger(before_quality_obj, args.auto_repair_warning_threshold)

    report: Dict[str, Any] = {
        "fixture": name,
        "mode": args.auto_repair,
        "triggered": should_repair,
        "trigger_reasons": reasons,
        "before": {
            "plan": str(plan_path),
            "audit": str(before_audit),
            "quality": str(before_quality),
            "gate": before_quality_obj.get("gate"),
            "warnings": before_quality_obj.get("warnings"),
        },
        "after": None,
        "repairs": None,
        "audit_diff": None,
        "quality_diff": None,
    }

    if not should_repair:
        report_path = fixture_root / "auto_repair_report.json"
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if quality_fail_on_hard and not bool((before_quality_obj.get("gate") or {}).get("pass_hard")):
            return None, None, None, f"{name}: quality hard gate failed"
        return before_audit, before_quality, before_quality_obj, None

    after_dir = fixture_root / "after"
    after_dir.mkdir(parents=True, exist_ok=True)
    repaired_plan_path = after_dir / "plan.repaired.json"
    repairs_path = after_dir / "repairs.applied.json"

    repair_cmd = [
        sys.executable,
        "scripts/repair_slide_plan.py",
        "--plan-json",
        str(plan_path),
        "--out-plan-json",
        str(repaired_plan_path),
        "--out-repairs-json",
        str(repairs_path),
        "--mode",
        args.auto_repair,
    ]
    if args.repair_max_bullets_per_slide is not None:
        repair_cmd.extend(["--max-bullets-per-slide", str(args.repair_max_bullets_per_slide)])
    if args.repair_max_evidence_per_bullet is not None:
        repair_cmd.extend(["--max-evidence-per-bullet", str(args.repair_max_evidence_per_bullet)])
    if args.repair_long_bullet_chars is not None:
        repair_cmd.extend(["--long-bullet-chars", str(args.repair_long_bullet_chars)])
    if args.repair_downgrade_visual_role_on_no_link is not None:
        repair_cmd.extend([
            "--downgrade-visual-role-on-no-link",
            str(int(bool(args.repair_downgrade_visual_role_on_no_link))),
        ])

    repair_proc = _run(repair_cmd)
    if repair_proc.returncode != 0:
        return None, None, None, f"{name}: repair failed\n{repair_proc.stdout}{repair_proc.stderr}"

    after_audit, after_quality, after_quality_obj, after_err = _run_smoke_and_quality(
        plan_path=repaired_plan_path,
        out_root=fixture_root,
        test_name="after",
        quality_fail_on_hard=False,
        quality_strict_missing_assets=quality_strict_missing_assets,
        args=args,
    )
    if after_err:
        return None, None, None, f"{name}: after run failed\n{after_err}"

    assert after_audit is not None and after_quality is not None and after_quality_obj is not None

    audit_diff_proc = _run([sys.executable, "scripts/diff_audit_json.py", str(before_audit), str(after_audit)])
    quality_diff_proc = _run([sys.executable, "scripts/diff_quality_json.py", str(before_quality), str(after_quality)])

    audit_diff_txt = after_dir / "audit.diff.txt"
    quality_diff_txt = after_dir / "quality.diff.txt"
    audit_diff_txt.write_text(audit_diff_proc.stdout + (audit_diff_proc.stderr or ""), encoding="utf-8")
    quality_diff_txt.write_text(quality_diff_proc.stdout + (quality_diff_proc.stderr or ""), encoding="utf-8")

    repairs_payload: Dict[str, Any] = {}
    if repairs_path.exists():
        try:
            repairs_payload = _load_json(repairs_path)
        except Exception:
            repairs_payload = {}

    report["after"] = {
        "plan": str(repaired_plan_path),
        "audit": str(after_audit),
        "quality": str(after_quality),
        "gate": after_quality_obj.get("gate"),
        "warnings": after_quality_obj.get("warnings"),
    }
    report["repairs"] = repairs_payload
    report["audit_diff"] = _parse_diff_output(audit_diff_proc.stdout + (audit_diff_proc.stderr or ""))
    report["quality_diff"] = _parse_diff_output(quality_diff_proc.stdout + (quality_diff_proc.stderr or ""))

    report_path = fixture_root / "auto_repair_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    after_gate = after_quality_obj.get("gate") if isinstance(after_quality_obj.get("gate"), dict) else {}
    if quality_fail_on_hard and not bool(after_gate.get("pass_hard")):
        return None, None, None, f"{name}: hard gate still failing after repair"

    warning_count_after = int(after_gate.get("warning_count", 0) or 0)
    if args.auto_repair_warning_threshold >= 0 and warning_count_after > args.auto_repair_warning_threshold:
        return None, None, None, (
            f"{name}: warning_count still > threshold after repair "
            f"({warning_count_after} > {args.auto_repair_warning_threshold})"
        )

    return after_audit, after_quality, after_quality_obj, None


def main() -> None:
    ap = argparse.ArgumentParser(description="Run deterministic regression suite (golden plans + golden audit/hash)")
    ap.add_argument("--fixture-dir", default="tests/fixtures/slides", help="Fixture directory")
    ap.add_argument("--out-root", default="out/tests", help="Output root for generated audits")
    ap.add_argument("--fixtures", default="", help="Comma-separated fixture names (default: all *.plan.json)")
    ap.add_argument(
        "--profile",
        choices=["dev", "ci"],
        default="dev",
        help="Execution profile: dev=local permissive defaults, ci=quality strict defaults",
    )
    ap.add_argument("--update-golden", action="store_true", help="Regenerate golden audit files and hashes")
    ap.add_argument("--quality-golden", action="store_true", help="Enable golden diff/hash checks for quality.json")
    ap.add_argument("--quality-fail-on-hard", action="store_true", help="Fail suite if quality evaluator hits hard gates")
    ap.add_argument("--quality-strict-missing-assets", action="store_true", help="Quality hard gate: missing_asset_count must be 0")

    ap.add_argument("--quality-hard-min-evidence-per-bullet", type=int, default=1)
    ap.add_argument("--quality-warn-max-single-evidence-ratio", type=float, default=0.30)
    ap.add_argument("--quality-warn-max-fallback-asset-ratio", type=float, default=0.20)
    ap.add_argument("--quality-warn-bullets-per-slide-cap", type=int, default=7)
    ap.add_argument("--quality-warn-max-slides-over-bullets-cap", type=int, default=2)
    ap.add_argument("--quality-warn-long-bullet-chars", type=int, default=140)
    ap.add_argument("--quality-warn-max-long-bullets", type=int, default=2)

    ap.add_argument(
        "--auto-repair",
        choices=["off", "soft", "hard"],
        default="off",
        help="off=disabled, soft=readability+visual-linking, hard=soft+evidence trim+table compact",
    )
    ap.add_argument(
        "--auto-repair-warning-threshold",
        type=int,
        default=0,
        help="Trigger auto-repair when warning_count > threshold (-1 disables warning-trigger)",
    )
    ap.add_argument("--repair-max-bullets-per-slide", type=int, default=None)
    ap.add_argument("--repair-max-evidence-per-bullet", type=int, default=None)
    ap.add_argument("--repair-long-bullet-chars", type=int, default=None)
    ap.add_argument(
        "--repair-downgrade-visual-role-on-no-link",
        type=int,
        choices=[0, 1],
        default=None,
        help="Override SLIDES_REPAIR_DOWNGRADE_VISUAL_ROLE_ON_NO_LINK",
    )
    args = ap.parse_args()

    if args.profile == "ci":
        args.quality_golden = True
        args.quality_fail_on_hard = True
        args.quality_strict_missing_assets = True

    fixture_dir = Path(args.fixture_dir)
    if not fixture_dir.exists():
        raise SystemExit(f"FAIL: missing fixture dir: {fixture_dir}")

    selected = {x.strip() for x in args.fixtures.split(",") if x.strip()}
    plan_files = sorted(fixture_dir.glob("*.plan.json"))
    if selected:
        plan_files = [p for p in plan_files if _fixture_name(p) in selected]
    if not plan_files:
        raise SystemExit("FAIL: no plan fixtures found")

    hash_file = fixture_dir / "golden_hashes.json"
    stored_hashes: Dict[str, str] = {}
    if hash_file.exists():
        try:
            payload = json.loads(hash_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                stored_hashes = {str(k): str(v) for k, v in payload.items()}
        except Exception:
            stored_hashes = {}

    next_hashes = dict(stored_hashes)

    quality_hash_file = fixture_dir / "golden_quality_hashes.json"
    stored_quality_hashes: Dict[str, str] = {}
    if quality_hash_file.exists():
        try:
            payload = json.loads(quality_hash_file.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                stored_quality_hashes = {str(k): str(v) for k, v in payload.items()}
        except Exception:
            stored_quality_hashes = {}
    next_quality_hashes = dict(stored_quality_hashes)

    errors: List[str] = []
    print(
        f"PROFILE {args.profile}: quality_golden={args.quality_golden} "
        f"quality_fail_on_hard={args.quality_fail_on_hard} "
        f"quality_strict_missing_assets={args.quality_strict_missing_assets} "
        f"auto_repair={args.auto_repair}"
    )

    for plan in plan_files:
        name = _fixture_name(plan)

        if args.auto_repair == "off":
            audit_path, quality_path, quality_obj, run_err = _run_smoke_and_quality(
                plan_path=plan,
                out_root=Path(args.out_root),
                test_name=name,
                quality_fail_on_hard=args.quality_fail_on_hard,
                quality_strict_missing_assets=args.quality_strict_missing_assets,
                args=args,
            )
            if run_err:
                errors.append(f"{name}: {run_err}")
                continue
        else:
            audit_path, quality_path, quality_obj, run_err = _run_auto_repair(
                name=name,
                plan_path=plan,
                args=args,
                quality_fail_on_hard=args.quality_fail_on_hard,
                quality_strict_missing_assets=args.quality_strict_missing_assets,
            )
            if run_err:
                errors.append(run_err)
                continue

        if audit_path is None or quality_path is None or quality_obj is None:
            errors.append(f"{name}: internal error (missing final artifacts)")
            continue

        golden_audit = fixture_dir / f"{name}.golden.audit.json"
        current_hash = _hash_for(audit_path)
        current_quality_hash = _quality_hash_for(quality_path)

        if args.update_golden:
            shutil.copyfile(audit_path, golden_audit)
            next_hashes[name] = current_hash
            if args.quality_golden:
                golden_quality = fixture_dir / f"{name}.golden.quality.json"
                shutil.copyfile(quality_path, golden_quality)
                next_quality_hashes[name] = current_quality_hash
            print(f"UPDATED {name}: audit_hash={current_hash} quality_hash={current_quality_hash}")
            continue

        if not golden_audit.exists():
            errors.append(f"{name}: missing golden audit {golden_audit}")
            continue

        diff = _run(
            [
                sys.executable,
                "scripts/diff_audit_json.py",
                str(golden_audit),
                str(audit_path),
                "--fail-on-diff",
            ]
        )
        if diff.returncode != 0:
            errors.append(f"{name}: audit semantic diff detected\n{diff.stdout}{diff.stderr}")
            continue

        expected_hash = stored_hashes.get(name, "")
        if not expected_hash:
            errors.append(f"{name}: missing expected hash in {hash_file}")
            continue
        if current_hash != expected_hash:
            errors.append(f"{name}: hash mismatch expected={expected_hash} got={current_hash}")
            continue

        if args.quality_golden:
            golden_quality = fixture_dir / f"{name}.golden.quality.json"
            if not golden_quality.exists():
                errors.append(f"{name}: missing golden quality {golden_quality}")
                continue
            qdiff = _run(
                [
                    sys.executable,
                    "scripts/diff_quality_json.py",
                    str(golden_quality),
                    str(quality_path),
                    "--fail-on-diff",
                ]
            )
            if qdiff.returncode != 0:
                errors.append(f"{name}: quality semantic diff detected\n{qdiff.stdout}{qdiff.stderr}")
                continue
            expected_qhash = stored_quality_hashes.get(name, "")
            if not expected_qhash:
                errors.append(f"{name}: missing expected quality hash in {quality_hash_file}")
                continue
            if current_quality_hash != expected_qhash:
                errors.append(
                    f"{name}: quality hash mismatch expected={expected_qhash} got={current_quality_hash}"
                )
                continue

        print(f"PASS {name}: audit_hash={current_hash} quality_hash={current_quality_hash}")

    if args.update_golden:
        hash_file.write_text(json.dumps(dict(sorted(next_hashes.items())), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if args.quality_golden:
            quality_hash_file.write_text(
                json.dumps(dict(sorted(next_quality_hashes.items())), indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )

    if errors:
        print("FAIL")
        for e in errors:
            print(f"- {e}")
        raise SystemExit(1)

    print("PASS")


if __name__ == "__main__":
    main()
