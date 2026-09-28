#!/usr/bin/env python3
"""
Submission Builder for Kaggriculture.
Packages agent scripts and the kaggriculture library into Kaggle-compliant submissions
(either a multi-file submission.tar.gz or a standalone single-file main.py).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Iterable, Optional


ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src" / "kaggriculture"
SUBMISSIONS_DIR = ROOT_DIR / "submissions"
BUILD_DIR = ROOT_DIR / "build"
MAX_WEIGHT_BYTES = 20 * 1024 * 1024


def _static_linux_binary(candidate: Path) -> bool:
    if not candidate.is_file() or not os.access(candidate, os.X_OK):
        return False
    try:
        description = subprocess.check_output(
            ["file", "-b", str(candidate)], text=True, stderr=subprocess.STDOUT
        ).lower()
    except (OSError, subprocess.CalledProcessError):
        return False
    return "elf" in description and (
        "statically linked" in description
        or "static-pie" in description
        or "musl" in description
    )


def find_static_rust_binary(explicit: Optional[Path] = None) -> Path:
    """Find a Linux static kagg binary, rejecting host-native executables."""
    candidates = [explicit] if explicit else []
    if os.environ.get("KAGG_STATIC_BINARY"):
        candidates.append(Path(os.environ["KAGG_STATIC_BINARY"]))
    candidates.extend(
        ROOT_DIR / p for p in (
            "kaggriculture-simulation/src-rust/target/x86_64-unknown-linux-musl/release/kagg",
            "kaggriculture-simulation/src-rust/target/release/kagg",
            "build/kagg",
        )
    )
    for candidate in candidates:
        if candidate and _static_linux_binary(candidate):
            return candidate.resolve()
    raise FileNotFoundError(
        "No static Linux Rust binary found. Build kagg for "
        "x86_64-unknown-linux-musl or pass --rust-binary PATH."
    )


def find_weight_files(agent_path: Path, weights: Iterable[Path] = ()) -> list[Path]:
    """Collect requested/nearby .pt files and enforce the 20 MiB limit."""
    candidates = list(weights)
    candidates.extend(sorted(agent_path.parent.rglob("*.pt")))
    candidates.extend(sorted((ROOT_DIR / "experiments").rglob("*.pt")))
    result, seen = [], set()
    for item in candidates:
        item = Path(item).resolve()
        if item in seen:
            continue
        seen.add(item)
        if not item.is_file():
            raise FileNotFoundError(f"Weight file does not exist: {item}")
        if item.stat().st_size >= MAX_WEIGHT_BYTES:
            raise ValueError(f"Weight file exceeds 20 MiB submission limit: {item}")
        result.append(item)
    return result


def build_multifile_tar(
    agent_path: Path,
    output_tar: Path,
    *,
    weights: Iterable[Path] = (),
    rust_binary: Optional[Path] = None,
    include_rust_binary: bool = True,
) -> Path:
    """
    Bundles the agent's main.py and the kaggriculture library into a submission.tar.gz.
    Kaggle extracts this archive with main.py at root.
    """
    print(f"\n📦 Building multi-file tar.gz submission from {agent_path}...")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    staging = BUILD_DIR / ".submission-staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    try:
        tmp_path = staging
        
        agent_dir = agent_path.parent

        # 1. Copy main.py to root of staging dir
        shutil.copy2(agent_path, tmp_path / "main.py")

        # 2. Copy sibling agent modules (e.g. submissions/alpha_modular/*.py)
        for py_file in sorted(agent_dir.glob("*.py")):
            if py_file.name == "main.py":
                continue
            shutil.copy2(py_file, tmp_path / py_file.name)

        # 3. Copy kaggriculture package into staging dir
        shutil.copytree(
            SRC_DIR,
            tmp_path / "kaggriculture",
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )
        
        # 4. Include model weights and the Linux runtime.
        for weight in find_weight_files(agent_path, weights):
            shutil.copy2(weight, tmp_path / weight.name)
        if include_rust_binary:
            binary = find_static_rust_binary(rust_binary)
            shutil.copy2(binary, tmp_path / "kagg")
            (tmp_path / "kagg").chmod(0o755)
        else:
            print("  (skipping static kagg runtime: pure-Python agent)")

        # 5. Create tar.gz archive
        output_tar.parent.mkdir(parents=True, exist_ok=True)
        with tarfile.open(output_tar, "w:gz") as tar:
            for item in tmp_path.iterdir():
                tar.add(
                    item,
                    arcname=item.name,
                    filter=lambda info: None
                    if "__pycache__" in Path(info.name).parts
                    or info.name.endswith((".pyc", ".pyo"))
                    else info,
                )
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    archive_size = output_tar.stat().st_size
    max_archive_bytes = 100 * 1024 * 1024  # 100 MiB limit
    if archive_size >= max_archive_bytes:
        raise ValueError(
            f"Submission archive {output_tar} size ({archive_size / (1024*1024):.2f} MiB) "
            f"exceeds Kaggle limit of 100 MiB ({max_archive_bytes:,} bytes)!"
        )
    print(f"✅ Multi-file archive created: {output_tar} ({archive_size / 1024:.1f} KB, strictly < 100 MiB)")
    return output_tar


def build_standalone_single_file(agent_path: Path, output_file: Path) -> Path:
    """
    Inlines essential library components into a single standalone self-contained main.py.
    If the source agent is already self-contained, copies it directly.
    """
    print(f"\n📄 Building single-file standalone submission from {agent_path}...")
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    agent_src = agent_path.read_text()
    
    # If agent is already completely self-contained, no inlining needed
    if "from kaggriculture" not in agent_src and "import kaggriculture" not in agent_src:
        output_file.write_text(agent_src)
        print(f"✅ Standalone file created (self-contained agent): {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
        return output_file

    # Read library modules in dependency order
    items_src = (SRC_DIR / "env" / "items.py").read_text()
    models_src = (SRC_DIR / "env" / "models.py").read_text()
    actions_src = (SRC_DIR / "actions" / "actions.py").read_text()
    tracking_src = (SRC_DIR / "helpers" / "tracking.py").read_text()
    market_src = (SRC_DIR / "helpers" / "market_prediction.py").read_text()
    solver_src = (SRC_DIR / "helpers" / "solver.py").read_text()
    opponent_src = (SRC_DIR / "helpers" / "opponent.py").read_text()
    market_planning_src = (SRC_DIR / "actions" / "market_planning.py").read_text()
    controller_src = (SRC_DIR / "actions" / "controller.py").read_text()

    # Clean local package imports and future imports
    def clean_imports(code: str) -> str:
        lines = []
        skipping_import = False
        for line in code.splitlines():
            if skipping_import:
                if ")" in line:
                    skipping_import = False
                continue
            if (
                line.startswith("from kaggriculture")
                or line.startswith("import kaggriculture")
                or line.startswith("from __future__")
                or "sys.path.insert" in line
            ):
                if "(" in line and ")" not in line:
                    skipping_import = True
                continue
            lines.append(line)
        return "\n".join(lines)

    combined_code = f"""# ==============================================================================
# Self-Contained Standalone Kaggriculture Agent
# Generated by build_submission.py
# ==============================================================================
from __future__ import annotations
import math
import os
import random
import sys
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

# --- items.py ---
{clean_imports(items_src)}

# --- models.py ---
{clean_imports(models_src)}

# --- actions.py ---
{clean_imports(actions_src)}

# --- helpers/tracking.py ---
{clean_imports(tracking_src)}

# --- helpers/market_prediction.py ---
{clean_imports(market_src)}

# --- helpers/solver.py ---
{clean_imports(solver_src)}

# --- helpers/opponent.py ---
{clean_imports(opponent_src)}

# --- actions/market_planning.py ---
{clean_imports(market_planning_src)}

# --- controller.py ---
{clean_imports(controller_src)}

# --- agent logic ---
{clean_imports(agent_src)}
"""
    output_file.write_text(combined_code)
    print(f"✅ Standalone file created: {output_file} ({output_file.stat().st_size / 1024:.1f} KB)")
    return output_file


def validate_submission(target_file: Path, episode_steps: int = 720) -> bool:
    """
    Validates that the built submission executes cleanly in kaggle_environments without errors.
    If target_file is a tar.gz archive, extracts to a temp directory to validate main.py as Kaggle does.
    """
    print(f"\n🔍 Validating submission against Kaggle Environments ({episode_steps} steps)...")
    try:
        import tarfile
        from kaggle_environments import make

        agent_entry = str(target_file)
        temp_dir = None
        if target_file.name.endswith(".tar.gz") or target_file.suffix == ".tar":
            validation_dir = BUILD_DIR / ".submission-validation"
            if validation_dir.exists():
                shutil.rmtree(validation_dir)
            validation_dir.mkdir(parents=True)
            temp_dir = validation_dir
            with tarfile.open(target_file, "r:*") as tar:
                tar.extractall(path=temp_dir)
            agent_entry = str(temp_dir / "main.py")

        env = make("kaggriculture", configuration={"episodeSteps": episode_steps})
        
        # Run test match against random agent
        steps = env.run([agent_entry, "random"])

        if temp_dir is not None:
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        if steps and len(steps) > 0:
            final_p1_money = steps[-1][0].observation.farms[0]["money"]
            final_p2_money = steps[-1][1].observation.farms[1]["money"]
            print(f"🎉 Validation SUCCESS! Completed {len(steps)} turns.")
            print(f"   Agent Balance:    ${final_p1_money:,.2f}")
            print(f"   Opponent Balance: ${final_p2_money:,.2f}")
            return True
        else:
            print("❌ Validation FAILED: No steps returned.")
            return False
            
    except Exception as e:
        print(f"❌ Validation FAILED with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(description="Kaggriculture Submission Builder")
    parser.add_argument(
        "--agent", "-a",
        default="two_team_grandmaster",
        help="Agent template name (e.g. two_team_grandmaster, sovereign_apex, care_mill) or path to main.py",
    )
    parser.add_argument(
        "--format", "-f",
        choices=["tar", "single"],
        default="tar",
        help="Submission packaging format: 'tar' for submission.tar.gz (recommended) or 'single' for standalone main.py",
    )
    parser.add_argument(
        "--no-validate",
        action="store_true",
        help="Skip dry-run validation",
    )
    parser.add_argument(
        "--weights", action="append", type=Path, default=[],
        help="PyTorch .pt weight to include (repeatable; each must be <20 MiB)",
    )
    parser.add_argument(
        "--rust-binary", type=Path, default=None,
        help="Static Linux kagg binary to include",
    )
    parser.add_argument(
        "--no-rust-binary", action="store_true",
        help="Skip bundling the static kagg runtime (only for agents that "
             "never shell out to kagg; cross-compiling musl from macOS is "
             "unsupported)",
    )
    parser.add_argument(
        "--message", "-m",
        default=None,
        help="Submission message for Kaggle CLI",
    )
    parser.add_argument(
        "--submit", "-s",
        action="store_true",
        help="Directly submit to Kaggle via Kaggle CLI after building and validating",
    )
    
    args = parser.parse_args()
    
    # 1. Resolve agent path
    if Path(args.agent).is_file():
        agent_path = Path(args.agent).resolve()
        agent_name = agent_path.parent.name
    elif (SUBMISSIONS_DIR / args.agent / "main.py").is_file():
        agent_path = SUBMISSIONS_DIR / args.agent / "main.py"
        agent_name = args.agent
    else:
        print(f"❌ Could not find agent at: {args.agent}")
        print(f"Available templates: {[d.name for d in SUBMISSIONS_DIR.iterdir() if d.is_dir()]}")
        sys.exit(1)
        
    print(f"🚀 Selected agent: '{agent_name}' ({agent_path})")
    
    # 2. Build submission artifact
    if args.format == "tar":
        out_path = BUILD_DIR / "submission.tar.gz"
        built_artifact = build_multifile_tar(
            agent_path, out_path, weights=args.weights, rust_binary=args.rust_binary,
            include_rust_binary=not args.no_rust_binary,
        )
    else:
        out_path = BUILD_DIR / "main.py"
        built_artifact = build_standalone_single_file(agent_path, out_path)
        
    # 3. Validate
    if not args.no_validate:
        is_valid = validate_submission(built_artifact)
        if not is_valid:
            print("\n❌ Build aborted due to validation failure. Fix errors above.")
            sys.exit(1)
            
    # 4. Display submission instructions
    msg = args.message or f"Kaggriculture {agent_name} ({args.format})"
    
    print("\n" + "=" * 70)
    print("🎯 SUBMISSION READY FOR KAGGLE")
    print("=" * 70)
    print(f"File to submit: {built_artifact}")
    print("\nRun this command to submit to Kaggle:")
    
    if args.format == "tar":
        cmd = f'kaggle competitions submit kaggriculture -f "{built_artifact}" -m "{msg}"'
    else:
        cmd = f'kaggle competitions submit kaggriculture -f "{built_artifact}" -m "{msg}"'
        
    print(f"  \033[1;32m{cmd}\033[0m\n")
    
    # 5. Direct submission if requested
    if args.submit:
        print("Executing Kaggle CLI submission...")
        res = subprocess.run(cmd, shell=True)
        if res.returncode == 0:
            print("🎉 Submission submitted successfully!")
        else:
            print(f"⚠️ Submission command exited with code {res.returncode}")


if __name__ == "__main__":
    main()
