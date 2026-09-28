"""Unit tests for live Kaggle submission telemetry and fallback monitor."""

import json
import tempfile
from pathlib import Path
import pytest

from scripts.monitor_live_submissions import (
    MAX_SUBMISSION_BYTES,
    parse_episodes_csv,
    parse_submissions_csv,
    parse_telemetry_log,
    validate_submission_size,
)


def test_validate_submission_size_valid(tmp_path: Path):
    test_tar = tmp_path / "submission.tar.gz"
    # Create 1 MB file
    test_tar.write_bytes(b"0" * (1024 * 1024))
    valid, size_mib, msg = validate_submission_size(test_tar, max_bytes=MAX_SUBMISSION_BYTES)
    assert valid is True
    assert 0.99 <= size_mib <= 1.01
    assert "SIZE VALID" in msg


def test_validate_submission_size_exceeded(tmp_path: Path):
    test_tar = tmp_path / "oversized.tar.gz"
    # Pretend limit is 500 bytes and file is 1000 bytes
    test_tar.write_bytes(b"0" * 1000)
    with pytest.raises(ValueError, match="SIZE VIOLATION"):
        validate_submission_size(test_tar, max_bytes=500)


def test_validate_submission_size_missing(tmp_path: Path):
    missing_file = tmp_path / "non_existent.tar.gz"
    with pytest.raises(FileNotFoundError, match="not found"):
        validate_submission_size(missing_file)


def test_parse_submissions_csv():
    raw_csv = (
        'ref,fileName,date,description,status,publicScore,privateScore\n'
        '56647370,submission.tar.gz,2026-09-28 16:57:18,"PSRO pick: agent_final",SubmissionStatus.COMPLETE,506.7,\n'
        '56453900,submission.tar.gz,2026-09-22 05:11:58,failed_submission,SubmissionStatus.ERROR,,\n'
        'Use "kaggle competitions replay <id>" to download\n'
    )
    parsed = parse_submissions_csv(raw_csv)
    assert len(parsed) == 2
    assert parsed[0]["ref"] == "56647370"
    assert parsed[0]["status"] == "SubmissionStatus.COMPLETE"
    assert parsed[1]["ref"] == "56453900"


def test_parse_episodes_csv():
    raw_csv = (
        "id,createTime,endTime,state,type\n"
        "114817526,2026-09-28 18:09:57,2026-09-28 18:13:21,EpisodeState.COMPLETED,EpisodeType.EPISODE_TYPE_PUBLIC\n"
        "114816055,2026-09-28 18:05:55,2026-09-28 18:09:38,EpisodeState.COMPLETED,EpisodeType.EPISODE_TYPE_PUBLIC\n"
        'Use "kaggle competitions logs <id> <index>" for agent logs.\n'
    )
    parsed = parse_episodes_csv(raw_csv)
    assert len(parsed) == 2
    assert parsed[0]["id"] == "114817526"
    assert parsed[1]["id"] == "114816055"


def test_parse_telemetry_clean_log(tmp_path: Path):
    log_file = tmp_path / "clean_log.json"
    dummy_log = [
        [{"duration": 12.5, "stdout": "", "stderr": ""}],
        [{"duration": 0.001, "stdout": "", "stderr": ""}],
        [{"duration": 0.002, "stdout": "", "stderr": ""}],
    ]
    log_file.write_text(json.dumps(dummy_log))

    result = parse_telemetry_log(log_file)
    assert result["turns_analyzed"] == 3
    assert result["fallback_triggered"] is False
    assert result["memory_error_detected"] is False
    assert result["cumulative_overage_s"] == 0.0
    assert result["max_turn_duration_ms"] == 2.0


def test_parse_telemetry_fallback_and_memory_error(tmp_path: Path):
    log_file = tmp_path / "fallback_log.json"
    dummy_log = [
        [{"duration": 10.0, "stdout": "", "stderr": ""}],
        [{"duration": 0.001, "stdout": "", "stderr": ""}],
        [{
            "duration": 1.45,
            "stdout": "",
            "stderr": (
                "[IMPENETRABLE_AGENT] CRITICAL: Caught KeyError at Step 19 (Day 0, Hour 19):\n"
                "KeyError: 'target'\n"
                "Traceback (most recent call last):\n"
                "  File 'main.py', line 50, in agent\n"
                "KeyError: 'target'\n"
                "Activating SafeFallbackController for this turn.\n"
            ),
        }],
        [{
            "duration": 0.001,
            "stdout": "",
            "stderr": "numpy.core._exceptions._ArrayMemoryError: Unable to allocate 6.8 GiB",
        }],
    ]
    log_file.write_text(json.dumps(dummy_log))

    result = parse_telemetry_log(log_file)
    assert result["turns_analyzed"] == 4
    assert result["fallback_triggered"] is True
    assert result["memory_error_detected"] is True
    assert pytest.approx(result["cumulative_overage_s"], 0.001) == 0.45
    assert len(result["extracted_exceptions"]) >= 1
    assert any("KeyError: 'target'" in exc for exc in result["extracted_exceptions"])
