#!/usr/bin/env python3
"""Monitor Live Submissions & Telemetry Auditor for Kaggriculture.

Interfaces with the Kaggle CLI to:
1. Validate submission archive size (< 100 MiB) before deployment.
2. Automatically retrieve the latest submission ID from Kaggle.
3. Fetch recent match episodes for that submission.
4. Download live execution logs for our agent (identifying seat 0 or 1).
5. Audit logs for fallback triggers (@impenetrable_agent / SafeFallbackController),
   RAM limit breaches (MemoryError, numpy ArrayMemoryError), and overage time consumption.
6. Generate a clear diagnostic report with exact exception autopsies.

Usage:
    .venv/bin/python scripts/monitor_live_submissions.py
    .venv/bin/python scripts/monitor_live_submissions.py --max-episodes 5
    .venv/bin/python scripts/monitor_live_submissions.py --check-size build/submission.tar.gz
    .venv/bin/python scripts/monitor_live_submissions.py --submission-id 56647370
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

# Strict Kaggle resource boundary
MAX_SUBMISSION_BYTES: int = 100 * 1024 * 1024  # 100 MiB = 104,857,600 bytes
DEFAULT_COMPETITION: str = "kaggriculture"
SOFT_TURN_LIMIT_S: float = 1.0
TOTAL_OVERAGE_BANK_S: float = 60.0

# Regex patterns for telemetry scanning
FALLBACK_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"FALLBACK TRIGGERED", re.IGNORECASE),
    re.compile(r"\[IMPENETRABLE_AGENT\]"),
    re.compile(r"Activating SafeFallbackController", re.IGNORECASE),
    re.compile(r"\[IMPENETRABLE_AGENT_WATCHDOG\]"),
    re.compile(r"SafeFallbackController taking over", re.IGNORECASE),
    re.compile(r"Non-dict action returned", re.IGNORECASE),
    re.compile(r"_exception_shield_triggered"),
]

MEMORY_ERROR_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"MemoryError"),
    re.compile(r"numpy\.core\._exceptions\._ArrayMemoryError"),
    re.compile(r"numpy\.exceptions\._ArrayMemoryError"),
    re.compile(r"OutOfMemoryError"),
    re.compile(r"torch\.cuda\.OutOfMemoryError"),
    re.compile(r"Cannot allocate memory", re.IGNORECASE),
    re.compile(r"Killed\s*$"),
]

OVERAGE_PATTERNS: Sequence[re.Pattern[str]] = [
    re.compile(r"[Oo]verage"),
    re.compile(r"[Ww]atchdog"),
    re.compile(r"remaining overage", re.IGNORECASE),
    re.compile(r"exceeded soft turn limit", re.IGNORECASE),
]

EXCEPTION_EXTRACTOR = re.compile(
    r"(?:Traceback \(most recent call last\):[\s\S]*?(?:Exception|Error|Crash):[^\n]+|"
    r"\b[A-Za-z0-9_]+(?:Error|Exception): [^\n]+)",
    re.MULTILINE,
)


def find_kaggle_cli() -> str:
    """Locates the kaggle CLI executable in PATH or project virtualenv."""
    candidate = shutil.which("kaggle")
    if candidate:
        return candidate

    venv_bin = Path(sys.executable).parent / "kaggle"
    if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
        return str(venv_bin)

    root_venv = Path(__file__).resolve().parents[1] / ".venv" / "bin" / "kaggle"
    if root_venv.is_file() and os.access(root_venv, os.X_OK):
        return str(root_venv)

    return "kaggle"


# =============================================================================
# 1. Submission Size Validation (< 100 MiB)
# =============================================================================

def validate_submission_size(
    tar_path: Union[str, Path] = "build/submission.tar.gz",
    max_bytes: int = MAX_SUBMISSION_BYTES,
) -> Tuple[bool, float, str]:
    """Asserts that submission archive is strictly < 100 MiB.

    Returns:
        (is_valid, size_mib, message)
    Raises:
        FileNotFoundError: If tar_path does not exist.
        ValueError: If size exceeds max_bytes.
    """
    path = Path(tar_path)
    if not path.is_file():
        raise FileNotFoundError(f"Submission archive not found: {path.resolve()}")

    size_bytes = path.stat().st_size
    size_mib = size_bytes / (1024 * 1024)
    max_mib = max_bytes / (1024 * 1024)

    if size_bytes >= max_bytes:
        msg = (
            f"❌ SIZE VIOLATION: '{path.name}' is {size_mib:.2f} MiB "
            f"({size_bytes:,} bytes), exceeding Kaggle limit of {max_mib:.1f} MiB "
            f"({max_bytes:,} bytes). Silent Docker build failure will occur!"
        )
        raise ValueError(msg)

    msg = (
        f"✅ SIZE VALID: '{path.name}' is {size_mib:.2f} MiB "
        f"({size_bytes:,} bytes) [< {max_mib:.1f} MiB limit]."
    )
    return True, size_mib, msg


# =============================================================================
# 2. Submission Retrieval
# =============================================================================

def parse_submissions_csv(csv_text: str) -> List[Dict[str, str]]:
    """Parses raw CSV or text output from `kaggle competitions submissions`."""
    if not csv_text.strip():
        return []

    lines = [line.strip() for line in csv_text.splitlines() if line.strip()]
    if not lines:
        return []

    valid_lines = [
        l for l in lines
        if not l.startswith("Use \"kaggle") and not l.startswith("kaggle ")
    ]
    if not valid_lines:
        return []

    # Check if header line is comma-separated
    first_line = valid_lines[0]
    if "," in first_line:
        reader = csv.DictReader(valid_lines)
        return [r for r in reader if r.get("ref") and str(r.get("ref")).isdigit()]

    # Fallback table parser
    rows: List[Dict[str, str]] = []
    headers = [h.strip() for h in re.split(r"\s{2,}", lines[0]) if h.strip()]
    for line in lines[1:]:
        parts = [p.strip() for p in re.split(r"\s{2,}", line) if p.strip()]
        if len(parts) >= len(headers):
            rows.append({headers[i]: parts[i] for i in range(len(headers))})
    return rows


def get_latest_submission_id(
    competition: str = DEFAULT_COMPETITION,
    cli_path: Optional[str] = None,
) -> Tuple[str, Dict[str, str]]:
    """Retrieves the submission ID of the most recent successful upload.

    Returns:
        (submission_id, submission_metadata_dict)
    """
    cli = cli_path or find_kaggle_cli()
    cmd = [cli, "competitions", "submissions", competition, "-v"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Failed to list submissions for '{competition}':\n{res.stderr.strip()}"
        )

    submissions = parse_submissions_csv(res.stdout)
    if not submissions:
        raise RuntimeError(f"No submissions found for competition '{competition}'.")

    # Filter for successful completions
    complete_submissions = [
        s for s in submissions
        if "complete" in s.get("status", "").lower()
    ]

    target = complete_submissions[0] if complete_submissions else submissions[0]
    sub_id = target.get("ref") or target.get("id") or target.get("submissionId") or ""
    if not sub_id:
        raise ValueError(f"Unable to parse submission ID from row: {target}")

    return str(sub_id), target


# =============================================================================
# 3. Episode Fetching
# =============================================================================

def parse_episodes_csv(csv_text: str) -> List[Dict[str, str]]:
    """Parses raw CSV output from `kaggle competitions episodes <SUBMISSION_ID> -v`."""
    if not csv_text.strip():
        return []
    valid_lines = [
        line.strip() for line in csv_text.splitlines()
        if line.strip() and not line.strip().startswith("Use \"kaggle") and not line.strip().startswith("kaggle ")
    ]
    if not valid_lines:
        return []
    reader = csv.DictReader(io.StringIO("\n".join(valid_lines)))
    rows: List[Dict[str, str]] = []
    for row in reader:
        if row and row.get("id") and str(row.get("id")).isdigit():
            rows.append(row)
    return rows


def get_recent_episodes(
    submission_id: str,
    max_episodes: int = 15,
    cli_path: Optional[str] = None,
) -> List[Dict[str, str]]:
    """Extracts recent episode IDs where the submission played."""
    cli = cli_path or find_kaggle_cli()
    cmd = [cli, "competitions", "episodes", str(submission_id), "-v"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(
            f"Failed to list episodes for submission '{submission_id}':\n{res.stderr.strip()}"
        )

    episodes = parse_episodes_csv(res.stdout)
    # Prefer COMPLETED matches
    completed = [
        ep for ep in episodes
        if "completed" in str(ep.get("state") or "").lower()
    ]
    selected = completed[:max_episodes] if completed else episodes[:max_episodes]
    return selected


# =============================================================================
# 4. Log Scraping & Agent Seat Resolution
# =============================================================================

def fetch_agent_log(
    episode_id: Union[str, int],
    target_dir: Union[str, Path],
    cli_path: Optional[str] = None,
) -> Tuple[Optional[int], Optional[Path], Optional[str]]:
    """Downloads execution logs for our agent seat (trying 0, then 1).

    Kaggle returns 403 Forbidden when requesting an opponent's logs,
    which deterministically identifies our agent seat.

    Returns:
        (agent_index, path_to_downloaded_log, error_message)
    """
    cli = cli_path or find_kaggle_cli()
    td = Path(target_dir)
    td.mkdir(parents=True, exist_ok=True)

    # Try seat 0 first
    for seat in (0, 1):
        cmd = [cli, "competitions", "logs", str(episode_id), str(seat), "-p", str(td), "-q"]
        res = subprocess.run(cmd, capture_output=True, text=True)
        expected_file = td / f"episode-{episode_id}-agent-{seat}-logs.json"

        if res.returncode == 0 and expected_file.is_file():
            return seat, expected_file, None

        # Check if downloaded without quiet prefix or different name pattern
        matches = list(td.glob(f"episode-{episode_id}-agent-{seat}*.json"))
        if matches and matches[0].is_file():
            return seat, matches[0], None

    return None, None, f"Could not download logs for episode {episode_id} (both seats failed or 403)."


# =============================================================================
# 5. Telemetry Parsing
# =============================================================================

def parse_telemetry_log(log_path: Union[str, Path]) -> Dict[str, Any]:
    """Scans agent execution log for fallback triggers, OOMs, and timing overages."""
    path = Path(log_path)
    with open(path, "r", encoding="utf-8") as f:
        log_data = json.load(f)

    turns_analyzed = len(log_data)
    durations: List[float] = []
    stderrs: List[str] = []
    stdouts: List[str] = []

    fallback_matches: List[str] = []
    memory_errors: List[str] = []
    extracted_exceptions: List[str] = []
    overage_warnings: List[str] = []

    cumulative_overage_s: float = 0.0

    for step_idx, step_entries in enumerate(log_data):
        if not isinstance(step_entries, list) or not step_entries:
            continue
        entry = step_entries[0]
        if not isinstance(entry, dict):
            continue

        dur = float(entry.get("duration", 0.0) or 0.0)
        stderr = str(entry.get("stderr", "") or "")
        stdout = str(entry.get("stdout", "") or "")

        durations.append(dur)
        if stderr:
            stderrs.append(stderr)
        if stdout:
            stdouts.append(stdout)

        # Overage calculation (skip turn 0 container initialization)
        if step_idx > 0 and dur > SOFT_TURN_LIMIT_S:
            cumulative_overage_s += (dur - SOFT_TURN_LIMIT_S)

        # Telemetry regex scans on stderr + stdout
        combined_text = f"{stderr}\n{stdout}" if stderr else stdout

        for pat in FALLBACK_PATTERNS:
            found = pat.findall(combined_text)
            if found:
                fallback_matches.append(f"Step {step_idx}: Pattern '{pat.pattern}' matched: {found}")

        for pat in MEMORY_ERROR_PATTERNS:
            found = pat.findall(combined_text)
            if found:
                memory_errors.append(f"Step {step_idx}: Memory pattern '{pat.pattern}' matched: {found}")

        for pat in OVERAGE_PATTERNS:
            found = pat.findall(combined_text)
            if found:
                overage_warnings.append(f"Step {step_idx}: Overage pattern '{pat.pattern}' matched: {found}")

        if stderr:
            exc_found = EXCEPTION_EXTRACTOR.findall(stderr)
            if exc_found:
                for exc in exc_found:
                    clean = exc.strip()
                    if clean and clean not in extracted_exceptions:
                        extracted_exceptions.append(f"Step {step_idx}: {clean}")

    max_duration_s = max(durations[1:]) if len(durations) > 1 else (durations[0] if durations else 0.0)
    init_duration_s = durations[0] if durations else 0.0

    return {
        "turns_analyzed": turns_analyzed,
        "init_duration_s": init_duration_s,
        "max_turn_duration_ms": max_duration_s * 1000.0,
        "cumulative_overage_s": cumulative_overage_s,
        "fallback_triggered": len(fallback_matches) > 0,
        "fallback_matches": fallback_matches,
        "memory_error_detected": len(memory_errors) > 0,
        "memory_errors": memory_errors,
        "extracted_exceptions": extracted_exceptions,
        "overage_warnings": overage_warnings,
        "has_stderr": len(stderrs) > 0,
    }


# =============================================================================
# 6. Master Live Monitor & Telemetry Pipeline
# =============================================================================

def monitor_live_submissions(
    competition: str = DEFAULT_COMPETITION,
    submission_id: Optional[str] = None,
    max_episodes: int = 10,
    check_size_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    cli_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Master pipeline executing full live telemetry audit."""
    cli = cli_path or find_kaggle_cli()

    # Pre-check size validation if requested or file exists
    size_valid, size_mib, size_msg = (True, 0.0, "Size pre-check skipped.")
    target_tar = check_size_path
    if target_tar is None:
        if Path("build/submission.tar.gz").is_file():
            target_tar = "build/submission.tar.gz"
        elif Path("submission.tar.gz").is_file():
            target_tar = "submission.tar.gz"

    if target_tar:
        size_valid, size_mib, size_msg = validate_submission_size(target_tar)

    # 1. Retrieve Submission ID
    sub_meta: Dict[str, str] = {}
    if not submission_id:
        sub_id, sub_meta = get_latest_submission_id(competition, cli_path=cli)
    else:
        sub_id = str(submission_id)

    # 2. Fetch Episodes
    episodes = get_recent_episodes(sub_id, max_episodes=max_episodes, cli_path=cli)
    if not episodes:
        return {
            "submission_id": sub_id,
            "submission_meta": sub_meta,
            "size_msg": size_msg,
            "episodes_analyzed": 0,
            "fallback_episodes_count": 0,
            "avg_overage_s": 0.0,
            "results": [],
            "summary_text": f"Analyzed 0 recent episodes for submission {sub_id} (no completed matches found).",
        }

    # 3. Log Scraping & Telemetry Analysis
    results: List[Dict[str, Any]] = []
    temp_dir_ctx = tempfile.TemporaryDirectory() if output_dir is None else None
    scrape_dir = Path(output_dir) if output_dir else Path(temp_dir_ctx.name)
    scrape_dir.mkdir(parents=True, exist_ok=True)

    try:
        for ep in episodes:
            ep_id = ep.get("id") or ""
            if not ep_id:
                continue

            seat, log_file, err = fetch_agent_log(ep_id, scrape_dir, cli_path=cli)
            if log_file and log_file.is_file():
                telemetry = parse_telemetry_log(log_file)
                entry = {
                    "episode_id": ep_id,
                    "seat": seat,
                    "create_time": ep.get("createTime", ""),
                    "state": ep.get("state", ""),
                    "telemetry": telemetry,
                    "error": None,
                }
            else:
                entry = {
                    "episode_id": ep_id,
                    "seat": seat,
                    "create_time": ep.get("createTime", ""),
                    "state": ep.get("state", ""),
                    "telemetry": None,
                    "error": err,
                }
            results.append(entry)

    finally:
        if temp_dir_ctx is not None:
            temp_dir_ctx.cleanup()

    # 4. Diagnostic Aggregation
    valid_entries = [r for r in results if r.get("telemetry")]
    count = len(valid_entries)
    fallback_count = sum(1 for r in valid_entries if r["telemetry"]["fallback_triggered"])
    total_overage = sum(r["telemetry"]["cumulative_overage_s"] for r in valid_entries)
    avg_overage = (total_overage / count) if count > 0 else 0.0

    all_exceptions: List[str] = []
    all_memory_errors: List[str] = []
    for r in valid_entries:
        t = r["telemetry"]
        if t["extracted_exceptions"]:
            all_exceptions.extend([f"Ep {r['episode_id']}: {e}" for e in t["extracted_exceptions"]])
        if t["memory_errors"]:
            all_memory_errors.extend([f"Ep {r['episode_id']}: {m}" for m in t["memory_errors"]])

    summary_text = (
        f"Analyzed {count} recent episodes. Fallback triggered in {fallback_count} episodes. "
        f"Average overage consumed: {avg_overage:.4f} seconds."
    )

    return {
        "submission_id": sub_id,
        "submission_meta": sub_meta,
        "size_msg": size_msg,
        "episodes_analyzed": count,
        "fallback_episodes_count": fallback_count,
        "avg_overage_s": avg_overage,
        "all_exceptions": all_exceptions,
        "all_memory_errors": all_memory_errors,
        "results": results,
        "summary_text": summary_text,
    }


def print_diagnostic_report(report: Dict[str, Any]) -> None:
    """Formats and prints terminal diagnostic summary."""
    print("=" * 88)
    print("🌾 Kaggriculture Live Submission Telemetry & Fallback Monitor")
    print("=" * 88)
    print(f"📦 Submission ID: {report['submission_id']}")
    if report.get("submission_meta"):
        desc = report["submission_meta"].get("description", "")
        date = report["submission_meta"].get("date", "")
        score = report["submission_meta"].get("publicScore", "")
        print(f"   Date: {date} | Public Score: {score} | Description: {desc}")
    print(f"📦 Size Pre-Check: {report['size_msg']}")
    print("-" * 88)

    results = report.get("results", [])
    for r in results:
        ep_id = r["episode_id"]
        seat = r.get("seat", "?")
        t = r.get("telemetry")
        if not t:
            print(f"⚠️  Episode {ep_id} (Seat {seat}): {r.get('error')}")
            continue

        status_icon = "🚨 FALLBACK TRIGGERED" if t["fallback_triggered"] else "✅ CLEAN"
        mem_icon = "💥 MEMORY BREACH" if t["memory_error_detected"] else "OK"

        print(f"Match {ep_id} (Seat {seat}) [{r['create_time']}]:")
        print(
            f"   {status_icon} | RAM: {mem_icon} | Max Turn: {t['max_turn_duration_ms']:.2f}ms | "
            f"Overage Consumed: {t['cumulative_overage_s']:.4f}s"
        )

        if t["fallback_triggered"]:
            print(f"   ⚠️  Fallback Matches:")
            for m in t["fallback_matches"]:
                print(f"      - {m}")

        if t["memory_errors"]:
            print(f"   💥 Memory Errors:")
            for m in t["memory_errors"]:
                print(f"      - {m}")

        if t["extracted_exceptions"]:
            print(f"   🔍 Exceptions Extracted:")
            for e in t["extracted_exceptions"]:
                print(f"      - {e}")

    print("-" * 88)
    print(f"📊 SUMMARY: {report['summary_text']}")

    if report["all_memory_errors"]:
        print("\n💥 DETECTED RAM BREACHES (Breaching 6.5 GiB Limit):")
        for err in report["all_memory_errors"]:
            print(f"   • {err}")

    if report["all_exceptions"]:
        print("\n🔍 DETECTED EXCEPTIONS (Causing Fallback Triggers):")
        for err in report["all_exceptions"]:
            print(f"   • {err}")
    elif report["fallback_episodes_count"] == 0:
        print("🎉 Zero exceptions detected! SafeFallbackController remained 100% idle.")
    print("=" * 88)


def main() -> None:
    parser = argparse.ArgumentParser(description="Live Submission Telemetry Monitor for Kaggriculture.")
    parser.add_argument("--submission-id", "-s", type=str, default=None, help="Kaggle submission ID (default: latest)")
    parser.add_argument("--competition", "-c", type=str, default=DEFAULT_COMPETITION, help="Competition name")
    parser.add_argument("--max-episodes", "-n", type=int, default=10, help="Max recent episodes to analyze")
    parser.add_argument("--check-size", type=str, default=None, help="Assert tar.gz size strictly < 100 MiB")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory to save downloaded logs")
    parser.add_argument("--json", action="store_true", help="Output raw JSON summary")
    args = parser.parse_args()

    report = monitor_live_submissions(
        competition=args.competition,
        submission_id=args.submission_id,
        max_episodes=args.max_episodes,
        check_size_path=args.check_size,
        output_dir=args.output_dir,
    )

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print_diagnostic_report(report)

    # Non-zero exit code if fallback was triggered or memory was breached
    if report["fallback_episodes_count"] > 0 or len(report.get("all_memory_errors", [])) > 0:
        sys.exit(2)


if __name__ == "__main__":
    main()
