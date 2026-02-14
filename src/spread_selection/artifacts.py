"""Calibration artifact persistence for reproducible spread selection.

This module provides:
1. CalibrationArtifact dataclass for storing fold-level calibration parameters
2. ArtifactStore for saving/loading versioned calibration artifacts
3. RunMetadata for logging each execution
4. Hash computation for data integrity verification

Design Principles:
- Artifacts are immutable once saved
- Each artifact set is identified by a unique artifact_id
- Production mode loads frozen artifacts (no refitting)
- Research mode can regenerate artifacts
"""

import hashlib
import json
import logging
import subprocess
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default artifact directory
DEFAULT_ARTIFACT_DIR = "data/spread_selection/artifacts"
DEFAULT_RUN_LOG_PATH = "data/spread_selection/run_log.jsonl"


def compute_file_hash(file_path: str | Path, algorithm: str = "md5") -> str:
    """Compute hash of a file for integrity verification.

    Args:
        file_path: Path to file
        algorithm: Hash algorithm (md5, sha256)

    Returns:
        Hex digest of file hash
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Cannot hash non-existent file: {file_path}")

    hasher = hashlib.new(algorithm)
    with open(path, "rb") as f:
        # Read in chunks for large files
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


@dataclass
class FoldCalibration:
    """Calibration parameters for a single walk-forward fold.

    Attributes:
        eval_year: Year being evaluated
        slope: Logistic regression coefficient on edge_abs
        intercept: Logistic regression intercept
        breakeven_edge: Edge where P(cover) = 0.524 (breakeven at -110)
        p_cover_at_zero: P(cover) when edge=0
        n_train_games: Number of games used for training
        training_years: List of years used for training
    """
    eval_year: int
    slope: float
    intercept: float
    breakeven_edge: float
    p_cover_at_zero: float
    n_train_games: int
    training_years: list[int]


@dataclass
class FoldPushRates:
    """Push rate parameters for a single walk-forward fold.

    Attributes:
        eval_year: Year being evaluated
        tick_rates: Mapping of tick -> push rate
        default_even: Default rate for even ticks with insufficient data
        default_overall: Overall average push rate
        training_years: Years used for estimation
    """
    eval_year: int
    tick_rates: dict[int, float]  # JSON will convert int keys to strings
    default_even: float
    default_overall: float
    training_years: list[int]


@dataclass
class CalibrationArtifact:
    """Complete calibration artifact for reproducible selection.

    Contains all parameters needed to reproduce EV calculations exactly.
    """
    # Identification
    artifact_id: str  # Unique identifier (timestamp-based)
    created_at: str  # ISO timestamp

    # Configuration
    calibration_mode: str  # "primary" (ROLLING_2) or "ultra" (INCLUDE_ALL)
    training_window_seasons: Optional[int]  # None for INCLUDE_ALL
    years_range: tuple[int, int]  # (start_year, end_year)

    # Data integrity
    ats_export_hash: str  # MD5 hash of ats_export.csv used
    ats_export_path: str  # Path to source file
    git_commit: Optional[str]  # Git commit hash if available

    # Fold-level parameters
    fold_calibrations: list[FoldCalibration]
    fold_push_rates: list[FoldPushRates]

    # Metadata
    description: str = ""
    frozen: bool = False  # If True, this is a production artifact

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "calibration_mode": self.calibration_mode,
            "training_window_seasons": self.training_window_seasons,
            "years_range": list(self.years_range),
            "ats_export_hash": self.ats_export_hash,
            "ats_export_path": self.ats_export_path,
            "git_commit": self.git_commit,
            "fold_calibrations": [asdict(fc) for fc in self.fold_calibrations],
            "fold_push_rates": [
                {
                    "eval_year": fp.eval_year,
                    "tick_rates": {str(k): v for k, v in fp.tick_rates.items()},
                    "default_even": fp.default_even,
                    "default_overall": fp.default_overall,
                    "training_years": fp.training_years,
                }
                for fp in self.fold_push_rates
            ],
            "description": self.description,
            "frozen": self.frozen,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CalibrationArtifact":
        """Create from JSON dict."""
        fold_calibrations = [
            FoldCalibration(**fc) for fc in data["fold_calibrations"]
        ]
        fold_push_rates = [
            FoldPushRates(
                eval_year=fp["eval_year"],
                tick_rates={int(k): v for k, v in fp["tick_rates"].items()},
                default_even=fp["default_even"],
                default_overall=fp["default_overall"],
                training_years=fp["training_years"],
            )
            for fp in data["fold_push_rates"]
        ]
        return cls(
            artifact_id=data["artifact_id"],
            created_at=data["created_at"],
            calibration_mode=data["calibration_mode"],
            training_window_seasons=data["training_window_seasons"],
            years_range=tuple(data["years_range"]),
            ats_export_hash=data["ats_export_hash"],
            ats_export_path=data["ats_export_path"],
            git_commit=data.get("git_commit"),
            fold_calibrations=fold_calibrations,
            fold_push_rates=fold_push_rates,
            description=data.get("description", ""),
            frozen=data.get("frozen", False),
        )

    def get_fold_calibration(self, year: int) -> Optional[FoldCalibration]:
        """Get calibration for a specific evaluation year."""
        for fc in self.fold_calibrations:
            if fc.eval_year == year:
                return fc
        return None

    def get_fold_push_rates(self, year: int) -> Optional[FoldPushRates]:
        """Get push rates for a specific evaluation year."""
        for fp in self.fold_push_rates:
            if fp.eval_year == year:
                return fp
        return None


@dataclass
class RunMetadata:
    """Metadata for a single run execution."""
    timestamp: str  # ISO timestamp
    command: str  # Command that was run
    calibration_mode: str
    artifact_id: Optional[str]  # If using frozen artifacts
    years_range: tuple[int, int]
    ats_export_hash: str
    git_commit: Optional[str]
    frozen_mode: bool
    output_files: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "command": self.command,
            "calibration_mode": self.calibration_mode,
            "artifact_id": self.artifact_id,
            "years_range": list(self.years_range),
            "ats_export_hash": self.ats_export_hash,
            "git_commit": self.git_commit,
            "frozen_mode": self.frozen_mode,
            "output_files": self.output_files,
            "notes": self.notes,
        }


class ArtifactStore:
    """Storage and retrieval of calibration artifacts."""

    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_id: str) -> Path:
        """Get path for artifact file."""
        return self.artifact_dir / f"{artifact_id}.json"

    def generate_artifact_id(self, mode: str) -> str:
        """Generate unique artifact ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"cal_{mode}_{timestamp}"

    def save(self, artifact: CalibrationArtifact) -> Path:
        """Save artifact to JSON file.

        Args:
            artifact: CalibrationArtifact to save

        Returns:
            Path to saved file
        """
        path = self._artifact_path(artifact.artifact_id)

        if path.exists():
            raise FileExistsError(
                f"Artifact already exists: {path}. "
                "Artifacts are immutable - create a new one instead."
            )

        with open(path, "w") as f:
            json.dump(artifact.to_dict(), f, indent=2)

        logger.info(f"Saved calibration artifact: {path}")
        return path

    def load(self, artifact_id: str) -> CalibrationArtifact:
        """Load artifact from JSON file.

        Args:
            artifact_id: Artifact identifier

        Returns:
            CalibrationArtifact
        """
        path = self._artifact_path(artifact_id)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        with open(path) as f:
            data = json.load(f)

        artifact = CalibrationArtifact.from_dict(data)
        logger.info(f"Loaded calibration artifact: {artifact_id}")
        return artifact

    def load_latest(self, mode: str = "primary") -> Optional[CalibrationArtifact]:
        """Load most recent artifact for a mode.

        Args:
            mode: Calibration mode ("primary" or "ultra")

        Returns:
            Most recent CalibrationArtifact or None
        """
        pattern = f"cal_{mode}_*.json"
        files = sorted(self.artifact_dir.glob(pattern), reverse=True)

        if not files:
            return None

        # Load most recent
        artifact_id = files[0].stem
        return self.load(artifact_id)

    def load_frozen(self, mode: str = "primary") -> CalibrationArtifact:
        """Load frozen production artifact.

        Args:
            mode: Calibration mode

        Returns:
            CalibrationArtifact marked as frozen

        Raises:
            FileNotFoundError: If no frozen artifact exists
            ValueError: If artifact is not marked as frozen
        """
        artifact = self.load_latest(mode)

        if artifact is None:
            raise FileNotFoundError(
                f"No artifact found for mode '{mode}'. "
                f"Run 'backtest --save-artifacts' first."
            )

        if not artifact.frozen:
            logger.warning(
                f"Artifact {artifact.artifact_id} is not marked as frozen. "
                "Consider running 'freeze-artifacts' to lock for production."
            )

        return artifact

    def list_artifacts(self, mode: Optional[str] = None) -> list[str]:
        """List all artifact IDs.

        Args:
            mode: Filter by mode (optional)

        Returns:
            List of artifact IDs
        """
        if mode:
            pattern = f"cal_{mode}_*.json"
        else:
            pattern = "cal_*.json"

        files = sorted(self.artifact_dir.glob(pattern), reverse=True)
        return [f.stem for f in files]

    def freeze(self, artifact_id: str) -> None:
        """Mark an artifact as frozen for production use.

        Args:
            artifact_id: Artifact to freeze
        """
        path = self._artifact_path(artifact_id)

        with open(path) as f:
            data = json.load(f)

        data["frozen"] = True

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Froze artifact: {artifact_id}")


class RunLog:
    """Append-only log of run executions."""

    def __init__(self, log_path: str | Path = DEFAULT_RUN_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, metadata: RunMetadata) -> None:
        """Append run metadata to log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metadata.to_dict()) + "\n")
        logger.debug(f"Logged run: {metadata.timestamp}")

    def read_all(self) -> list[RunMetadata]:
        """Read all log entries."""
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append(RunMetadata(
                        timestamp=data["timestamp"],
                        command=data["command"],
                        calibration_mode=data["calibration_mode"],
                        artifact_id=data.get("artifact_id"),
                        years_range=tuple(data["years_range"]),
                        ats_export_hash=data["ats_export_hash"],
                        git_commit=data.get("git_commit"),
                        frozen_mode=data["frozen_mode"],
                        output_files=data.get("output_files", []),
                        notes=data.get("notes", ""),
                    ))
        return entries

    def read_recent(self, n: int = 10) -> list[RunMetadata]:
        """Read most recent n log entries."""
        all_entries = self.read_all()
        return all_entries[-n:]


def _to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {_to_python_type(k): _to_python_type(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_python_type(v) for v in val]
    return val


def create_artifact_from_walk_forward(
    wf_result,  # WalkForwardResult from calibration.py
    mode: str,
    ats_export_path: str,
    description: str = "",
) -> CalibrationArtifact:
    """Create CalibrationArtifact from walk-forward validation result.

    Args:
        wf_result: WalkForwardResult from walk_forward_validate()
        mode: Calibration mode ("primary" or "ultra")
        ats_export_path: Path to ats_export.csv used
        description: Optional description

    Returns:
        CalibrationArtifact ready to save
    """
    from .calibration import CALIBRATION_MODES

    # Get config for mode
    config = CALIBRATION_MODES.get(mode, CALIBRATION_MODES["primary"])

    # Extract fold calibrations (convert numpy types to Python types)
    fold_calibrations = []
    for fold in wf_result.fold_summaries:
        fold_calibrations.append(FoldCalibration(
            eval_year=_to_python_type(fold["eval_year"]),
            slope=_to_python_type(fold["slope"]),
            intercept=_to_python_type(fold["intercept"]),
            breakeven_edge=_to_python_type(fold["breakeven_edge"]),
            p_cover_at_zero=_to_python_type(fold["p_cover_at_zero"]),
            n_train_games=_to_python_type(fold["n_train"]),
            training_years=_to_python_type(fold["training_years_used"]),
        ))

    # Extract fold push rates (convert numpy types to Python types)
    fold_push_rates = []
    if hasattr(wf_result, "push_rate_summaries") and wf_result.push_rate_summaries:
        for prs in wf_result.push_rate_summaries:
            fold_push_rates.append(FoldPushRates(
                eval_year=_to_python_type(prs["eval_year"]),
                tick_rates=_to_python_type(prs.get("key_tick_rates", {})),
                default_even=_to_python_type(prs["default_even"]),
                default_overall=_to_python_type(prs["default_overall"]),
                training_years=_to_python_type(prs["years_trained"]),
            ))

    # Determine years range
    years = [fc.eval_year for fc in fold_calibrations]
    years_range = (min(years), max(years)) if years else (0, 0)

    # Compute data hash
    ats_hash = compute_file_hash(ats_export_path)

    # Generate artifact ID
    store = ArtifactStore()
    artifact_id = store.generate_artifact_id(mode)

    return CalibrationArtifact(
        artifact_id=artifact_id,
        created_at=datetime.now().isoformat(),
        calibration_mode=mode,
        training_window_seasons=config["training_window_seasons"],
        years_range=years_range,
        ats_export_hash=ats_hash,
        ats_export_path=str(ats_export_path),
        git_commit=get_git_commit_hash(),
        fold_calibrations=fold_calibrations,
        fold_push_rates=fold_push_rates,
        description=description,
        frozen=False,
    )


def verify_data_integrity(
    artifact: CalibrationArtifact,
    ats_export_path: str,
) -> tuple[bool, str]:
    """Verify that current data matches artifact's data hash.

    Args:
        artifact: CalibrationArtifact to verify against
        ats_export_path: Path to current ats_export.csv

    Returns:
        (is_valid, message) tuple
    """
    try:
        current_hash = compute_file_hash(ats_export_path)
    except FileNotFoundError:
        return False, f"Data file not found: {ats_export_path}"

    if current_hash != artifact.ats_export_hash:
        return False, (
            f"Data hash mismatch!\n"
            f"  Artifact hash: {artifact.ats_export_hash}\n"
            f"  Current hash:  {current_hash}\n"
            f"  Data may have changed since artifact was created."
        )

    return True, "Data integrity verified"
