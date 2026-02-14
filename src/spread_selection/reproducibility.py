"""Reproducibility guardrails for spread selection pipeline.

This module enforces determinism within a single dataset:
- Same data hash + same git commit + same command = identical output
- Explicit change detection when data or code changes
- No silent regeneration of outputs

Design Principles:
- Research flexibility: data CAN evolve when code changes intentionally
- Accidental drift protection: identical inputs MUST produce identical outputs
- Explicit visibility: all changes are logged and announced
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default paths
DEFAULT_FINGERPRINT_DIR = "data/spread_selection/fingerprints"


def compute_file_hash(file_path: str | Path, algorithm: str = "md5") -> str:
    """Compute hash of a file for integrity verification."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def get_git_commit_hash() -> Optional[str]:
    """Get current git commit hash if in a git repository."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()[:12]  # Short hash
    except Exception:
        pass
    return None


def get_git_dirty() -> bool:
    """Check if there are uncommitted changes."""
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return len(result.stdout.strip()) > 0
    except Exception:
        pass
    return False


@dataclass
class RunFingerprint:
    """Fingerprint capturing all inputs that affect output.

    If two runs have identical fingerprints, their outputs MUST be identical.
    """
    # Data identity
    data_hash: str  # MD5 of ats_export.csv
    data_path: str  # Path to data file

    # Code identity
    git_commit: Optional[str]  # Git commit hash
    git_dirty: bool  # True if uncommitted changes exist

    # Command identity
    command: str  # Command name (backtest, validate, predict)
    calibration_mode: str  # PRIMARY or ULTRA
    training_seasons: Optional[int]  # Training window (None = all)
    start_year: int
    end_year: int

    # Additional args that affect output
    line_type: str = "close"  # close or open
    extra_args: dict = None  # Any other relevant args

    # Metadata (not part of identity)
    timestamp: str = ""  # ISO timestamp

    def __post_init__(self):
        if self.extra_args is None:
            self.extra_args = {}
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    def identity_hash(self) -> str:
        """Compute hash of identity-relevant fields only.

        This excludes timestamp and other metadata.
        """
        identity = {
            "data_hash": self.data_hash,
            "git_commit": self.git_commit,
            "git_dirty": self.git_dirty,
            "command": self.command,
            "calibration_mode": self.calibration_mode,
            "training_seasons": self.training_seasons,
            "start_year": self.start_year,
            "end_year": self.end_year,
            "line_type": self.line_type,
            "extra_args": self.extra_args,
        }
        identity_str = json.dumps(identity, sort_keys=True)
        return hashlib.md5(identity_str.encode()).hexdigest()[:16]

    def to_dict(self) -> dict:
        d = asdict(self)
        d["identity_hash"] = self.identity_hash()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "RunFingerprint":
        # Remove computed fields
        data = {k: v for k, v in data.items() if k != "identity_hash"}
        return cls(**data)

    def print_header(self) -> None:
        """Print fingerprint as console header."""
        print("=" * 80)
        print("RUN FINGERPRINT")
        print("=" * 80)
        print(f"  Data:      {self.data_path}")
        print(f"  Hash:      {self.data_hash[:16]}...")
        print(f"  Git:       {self.git_commit or 'N/A'}" + (" (dirty)" if self.git_dirty else ""))
        print(f"  Mode:      {self.calibration_mode}")
        print(f"  Training:  {self.training_seasons or 'ALL'} seasons")
        print(f"  Years:     {self.start_year}-{self.end_year}")
        print(f"  Line:      {self.line_type}")
        print(f"  Identity:  {self.identity_hash()}")
        print(f"  Timestamp: {self.timestamp}")
        print("=" * 80)


@dataclass
class OutputRecord:
    """Record of a previous run's output for comparison."""
    fingerprint: RunFingerprint
    output_file: str
    output_hash: str  # Hash of output file
    created_at: str

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint.to_dict(),
            "output_file": self.output_file,
            "output_hash": self.output_hash,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "OutputRecord":
        return cls(
            fingerprint=RunFingerprint.from_dict(data["fingerprint"]),
            output_file=data["output_file"],
            output_hash=data["output_hash"],
            created_at=data["created_at"],
        )


class ReproducibilityGuard:
    """Enforces reproducibility rules for the spread selection pipeline."""

    def __init__(self, fingerprint_dir: str | Path = DEFAULT_FINGERPRINT_DIR):
        self.fingerprint_dir = Path(fingerprint_dir)
        self.fingerprint_dir.mkdir(parents=True, exist_ok=True)
        self.history_path = self.fingerprint_dir / "run_history.jsonl"
        self.outputs_dir = self.fingerprint_dir / "outputs"
        self.outputs_dir.mkdir(parents=True, exist_ok=True)

    def _output_record_path(self, output_file: str | Path) -> Path:
        """Get path for an output's fingerprint record."""
        # Use hash of output path as filename to avoid path issues
        path_hash = hashlib.md5(str(output_file).encode()).hexdigest()[:16]
        return self.outputs_dir / f"{path_hash}.json"

    def _load_output_record(self, output_file: str | Path) -> Optional[OutputRecord]:
        """Load the record for a specific output file."""
        record_path = self._output_record_path(output_file)
        if not record_path.exists():
            return None
        try:
            with open(record_path) as f:
                return OutputRecord.from_dict(json.load(f))
        except Exception as e:
            logger.warning(f"Could not load output record: {e}")
            return None

    def _save_output_record(self, record: OutputRecord) -> None:
        """Save the record for a specific output file."""
        record_path = self._output_record_path(record.output_file)
        with open(record_path, "w") as f:
            json.dump(record.to_dict(), f, indent=2)

    def _append_history(self, record: OutputRecord) -> None:
        """Append to run history log."""
        with open(self.history_path, "a") as f:
            f.write(json.dumps(record.to_dict()) + "\n")

    def check_for_changes(
        self,
        output_path: str | Path,
        current: RunFingerprint,
    ) -> tuple[bool, bool, Optional[OutputRecord]]:
        """Check if data or code changed since last run for a specific output.

        Returns:
            (data_changed, code_changed, last_record)
        """
        last = self._load_output_record(output_path)
        if last is None:
            return False, False, None

        data_changed = current.data_hash != last.fingerprint.data_hash
        code_changed = current.git_commit != last.fingerprint.git_commit

        return data_changed, code_changed, last

    def print_change_detection(
        self,
        output_path: str | Path,
        current: RunFingerprint,
    ) -> None:
        """Print change detection messages for a specific output."""
        data_changed, code_changed, last = self.check_for_changes(output_path, current)

        if data_changed:
            print("\n*** DATA CHANGED — recalibrating ***")
            if last:
                print(f"    Previous: {last.fingerprint.data_hash[:16]}...")
                print(f"    Current:  {current.data_hash[:16]}...")

        if code_changed:
            print("\n*** CODE CHANGED — recalibrating ***")
            if last:
                print(f"    Previous: {last.fingerprint.git_commit or 'N/A'}")
                print(f"    Current:  {current.git_commit or 'N/A'}")

        if current.git_dirty:
            print("\n*** WARNING: Uncommitted changes detected ***")
            print("    Results may not be reproducible until changes are committed.")

        if not data_changed and not code_changed and last:
            print("\n*** INPUTS UNCHANGED — expecting identical output ***")

    def check_output_exists(
        self,
        output_path: str | Path,
        current: RunFingerprint,
        rebuild_history: bool = False,
    ) -> tuple[bool, str]:
        """Check if output file exists and whether to proceed.

        Returns:
            (should_proceed, message)
        """
        path = Path(output_path)
        if not path.exists():
            return True, ""

        # Output exists - check if inputs are identical using per-output record
        last = self._load_output_record(output_path)

        if last is None:
            # No record for this output - treat as new
            return True, "No previous record for this output"

        data_changed = current.data_hash != last.fingerprint.data_hash
        code_changed = current.git_commit != last.fingerprint.git_commit

        if not data_changed and not code_changed:
            # Identical inputs - should produce identical output
            if rebuild_history:
                return True, "Rebuilding with identical inputs (--rebuild-history)"
            else:
                # Check if current identity matches last run's identity
                if current.identity_hash() == last.fingerprint.identity_hash():
                    return False, (
                        f"Output exists with identical inputs: {path}\n"
                        f"Use --rebuild-history to regenerate, or verify existing output."
                    )

        # Inputs changed - recalibration expected
        if rebuild_history:
            return True, "Rebuilding history with changed inputs"
        else:
            return True, "Inputs changed — regenerating output"

    def verify_determinism(
        self,
        output_path: str | Path,
        current: RunFingerprint,
    ) -> tuple[bool, str]:
        """Verify that output is deterministic given identical inputs.

        Call this AFTER generating output to verify it matches expectations.

        Returns:
            (is_deterministic, message)
        """
        path = Path(output_path)
        if not path.exists():
            return True, "No output to verify"

        current_output_hash = compute_file_hash(path)

        # Check if we have a previous run with identical inputs for THIS output
        last = self._load_output_record(output_path)
        if last is None:
            return True, "First run — no baseline for comparison"

        # Check if inputs are identical
        if current.identity_hash() != last.fingerprint.identity_hash():
            return True, "Inputs changed — determinism check not applicable"

        # Inputs identical — outputs MUST match
        if current_output_hash == last.output_hash:
            return True, "DETERMINISM VERIFIED: Identical inputs produced identical output"
        else:
            return False, (
                f"REPRODUCIBILITY ERROR: Identical inputs produced different output!\n"
                f"  Identity hash:  {current.identity_hash()}\n"
                f"  Previous output: {last.output_hash[:16]}...\n"
                f"  Current output:  {current_output_hash[:16]}...\n"
                f"  This indicates non-deterministic behavior in the pipeline."
            )

    def record_run(
        self,
        fingerprint: RunFingerprint,
        output_path: str | Path,
    ) -> OutputRecord:
        """Record a completed run for future comparison."""
        path = Path(output_path)
        output_hash = compute_file_hash(path) if path.exists() else ""

        record = OutputRecord(
            fingerprint=fingerprint,
            output_file=str(path),
            output_hash=output_hash,
            created_at=datetime.now().isoformat(),
        )

        self._save_output_record(record)
        self._append_history(record)

        return record


def create_fingerprint(
    data_path: str,
    command: str,
    calibration_mode: str,
    start_year: int,
    end_year: int,
    training_seasons: Optional[int] = None,
    line_type: str = "close",
    extra_args: dict = None,
) -> RunFingerprint:
    """Create a fingerprint for the current run."""
    return RunFingerprint(
        data_hash=compute_file_hash(data_path),
        data_path=str(data_path),
        git_commit=get_git_commit_hash(),
        git_dirty=get_git_dirty(),
        command=command,
        calibration_mode=calibration_mode,
        training_seasons=training_seasons,
        start_year=start_year,
        end_year=end_year,
        line_type=line_type,
        extra_args=extra_args or {},
    )
