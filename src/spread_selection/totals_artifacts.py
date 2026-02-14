"""Calibration artifact persistence for reproducible totals selection.

This module provides:
1. TotalsCalibrationArtifact dataclass for storing calibration parameters
2. TotalsArtifactStore for saving/loading versioned calibration artifacts
3. TotalsRunLog for logging each execution
4. Hash computation for data integrity verification

Design Principles:
- Artifacts are immutable once saved
- Each artifact is identified by a unique artifact_id
- Production mode loads frozen artifacts (no refitting)
- Research mode can regenerate artifacts

Mirrors the spread selection artifact system for consistency.
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

# Default paths
DEFAULT_ARTIFACT_DIR = "data/totals_selection/artifacts"
DEFAULT_RUN_LOG_PATH = "data/totals_selection/run_log.jsonl"
DEFAULT_FINGERPRINT_DIR = "data/totals_selection/fingerprints"


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


def _to_python_type(val):
    """Convert numpy types to Python native types for JSON serialization."""
    import numpy as np
    if isinstance(val, (np.integer, np.int64, np.int32)):
        return int(val)
    if isinstance(val, (np.floating, np.float64, np.float32)):
        return float(val)
    if isinstance(val, np.bool_):
        return bool(val)
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, dict):
        return {_to_python_type(k): _to_python_type(v) for k, v in val.items()}
    if isinstance(val, list):
        return [_to_python_type(v) for v in val]
    return val


def _numpy_json_encoder(obj):
    """JSON encoder for numpy types. Use as default= parameter to json.dump/dumps."""
    import numpy as np
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


@dataclass
class TotalsCalibrationArtifact:
    """Complete calibration artifact for reproducible totals selection.

    Contains all parameters needed to reproduce EV calculations exactly.
    """
    # Identification
    artifact_id: str  # Unique identifier (timestamp-based)
    created_at: str  # ISO timestamp

    # Configuration
    calibration_mode: str  # "model_only" or "weather_adjusted"
    years_range: tuple[int, int]  # (start_year, end_year)

    # Core calibration parameters
    sigma_base: float  # Base sigma value
    sigma_mode: str  # "fixed", "week_bucket", or "reliability_scaled"
    week_bucket_multipliers: dict[str, float]  # Week range -> multiplier

    # Reliability scaling parameters
    reliability_k: float
    reliability_sigma_min: float
    reliability_sigma_max: float
    reliability_max_games: int

    # EV thresholds
    ev_min: float
    ev_min_phase1: float

    # Kelly settings
    kelly_fraction: float
    max_bet_fraction: float

    # Data integrity
    backtest_data_hash: Optional[str]  # Hash of backtest CSV if available
    backtest_data_path: Optional[str]  # Path to backtest data
    git_commit: Optional[str]  # Git commit hash if available

    # Calibration statistics
    n_games_calibrated: int
    coverage_68: Optional[float] = None  # Empirical 68% coverage
    coverage_95: Optional[float] = None  # Empirical 95% coverage
    mae_calibration: Optional[float] = None  # MAE during calibration

    # Metadata
    description: str = ""
    frozen: bool = False  # If True, this is a production artifact
    has_weather_data: bool = False

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dict."""
        return {
            "artifact_id": self.artifact_id,
            "created_at": self.created_at,
            "calibration_mode": self.calibration_mode,
            "years_range": list(self.years_range),
            "sigma_base": _to_python_type(self.sigma_base),
            "sigma_mode": self.sigma_mode,
            "week_bucket_multipliers": _to_python_type(self.week_bucket_multipliers),
            "reliability_k": _to_python_type(self.reliability_k),
            "reliability_sigma_min": _to_python_type(self.reliability_sigma_min),
            "reliability_sigma_max": _to_python_type(self.reliability_sigma_max),
            "reliability_max_games": _to_python_type(self.reliability_max_games),
            "ev_min": _to_python_type(self.ev_min),
            "ev_min_phase1": _to_python_type(self.ev_min_phase1),
            "kelly_fraction": _to_python_type(self.kelly_fraction),
            "max_bet_fraction": _to_python_type(self.max_bet_fraction),
            "backtest_data_hash": self.backtest_data_hash,
            "backtest_data_path": self.backtest_data_path,
            "git_commit": self.git_commit,
            "n_games_calibrated": _to_python_type(self.n_games_calibrated),
            "coverage_68": _to_python_type(self.coverage_68) if self.coverage_68 else None,
            "coverage_95": _to_python_type(self.coverage_95) if self.coverage_95 else None,
            "mae_calibration": _to_python_type(self.mae_calibration) if self.mae_calibration else None,
            "description": self.description,
            "frozen": self.frozen,
            "has_weather_data": self.has_weather_data,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TotalsCalibrationArtifact":
        """Create from JSON dict."""
        return cls(
            artifact_id=data["artifact_id"],
            created_at=data["created_at"],
            calibration_mode=data["calibration_mode"],
            years_range=tuple(data["years_range"]),
            sigma_base=data["sigma_base"],
            sigma_mode=data["sigma_mode"],
            week_bucket_multipliers=data["week_bucket_multipliers"],
            reliability_k=data["reliability_k"],
            reliability_sigma_min=data["reliability_sigma_min"],
            reliability_sigma_max=data["reliability_sigma_max"],
            reliability_max_games=data["reliability_max_games"],
            ev_min=data["ev_min"],
            ev_min_phase1=data["ev_min_phase1"],
            kelly_fraction=data["kelly_fraction"],
            max_bet_fraction=data["max_bet_fraction"],
            backtest_data_hash=data.get("backtest_data_hash"),
            backtest_data_path=data.get("backtest_data_path"),
            git_commit=data.get("git_commit"),
            n_games_calibrated=data["n_games_calibrated"],
            coverage_68=data.get("coverage_68"),
            coverage_95=data.get("coverage_95"),
            mae_calibration=data.get("mae_calibration"),
            description=data.get("description", ""),
            frozen=data.get("frozen", False),
            has_weather_data=data.get("has_weather_data", False),
        )

    @classmethod
    def from_calibration_config(
        cls,
        config,  # TotalsCalibrationConfig
        artifact_id: str,
        years_range: tuple[int, int],
        backtest_data_path: Optional[str] = None,
        description: str = "",
        coverage_68: Optional[float] = None,
        coverage_95: Optional[float] = None,
        mae_calibration: Optional[float] = None,
    ) -> "TotalsCalibrationArtifact":
        """Create artifact from a TotalsCalibrationConfig.

        Args:
            config: TotalsCalibrationConfig from calibration
            artifact_id: Unique identifier
            years_range: (start_year, end_year)
            backtest_data_path: Path to backtest CSV
            description: Optional description
            coverage_68: Empirical 68% coverage
            coverage_95: Empirical 95% coverage
            mae_calibration: MAE from calibration

        Returns:
            TotalsCalibrationArtifact ready to save
        """
        # Compute data hash if path provided
        backtest_hash = None
        if backtest_data_path and Path(backtest_data_path).exists():
            backtest_hash = compute_file_hash(backtest_data_path)

        return cls(
            artifact_id=artifact_id,
            created_at=datetime.now().isoformat(),
            calibration_mode=config.calibration_mode,
            years_range=years_range,
            sigma_base=config.sigma_base,
            sigma_mode=config.sigma_mode,
            week_bucket_multipliers=config.week_bucket_multipliers,
            reliability_k=config.reliability_k,
            reliability_sigma_min=config.reliability_sigma_min,
            reliability_sigma_max=config.reliability_sigma_max,
            reliability_max_games=config.reliability_max_games,
            ev_min=config.ev_min,
            ev_min_phase1=config.ev_min_phase1,
            kelly_fraction=config.kelly_fraction,
            max_bet_fraction=config.max_bet_fraction,
            backtest_data_hash=backtest_hash,
            backtest_data_path=str(backtest_data_path) if backtest_data_path else None,
            git_commit=get_git_commit_hash(),
            n_games_calibrated=config.n_games_calibrated,
            coverage_68=coverage_68,
            coverage_95=coverage_95,
            mae_calibration=mae_calibration,
            description=description,
            frozen=False,
            has_weather_data=config.has_weather_data,
        )


@dataclass
class TotalsRunMetadata:
    """Metadata for a single totals run execution."""
    timestamp: str  # ISO timestamp
    command: str  # Command that was run
    calibration_mode: str  # "model_only" or "weather_adjusted"
    artifact_id: Optional[str]  # If using frozen artifacts
    years_range: tuple[int, int]
    backtest_data_hash: Optional[str]
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
            "backtest_data_hash": self.backtest_data_hash,
            "git_commit": self.git_commit,
            "frozen_mode": self.frozen_mode,
            "output_files": self.output_files,
            "notes": self.notes,
        }


class TotalsArtifactStore:
    """Storage and retrieval of totals calibration artifacts."""

    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.artifact_dir.mkdir(parents=True, exist_ok=True)

    def _artifact_path(self, artifact_id: str) -> Path:
        """Get path for artifact file."""
        return self.artifact_dir / f"{artifact_id}.json"

    def generate_artifact_id(self, mode: str = "model_only") -> str:
        """Generate unique artifact ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"totals_{mode}_{timestamp}"

    def save(self, artifact: TotalsCalibrationArtifact) -> Path:
        """Save artifact to JSON file.

        Args:
            artifact: TotalsCalibrationArtifact to save

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
            json.dump(artifact.to_dict(), f, indent=2, default=_numpy_json_encoder)

        logger.info(f"Saved totals calibration artifact: {path}")
        return path

    def load(self, artifact_id: str) -> TotalsCalibrationArtifact:
        """Load artifact from JSON file.

        Args:
            artifact_id: Artifact identifier

        Returns:
            TotalsCalibrationArtifact
        """
        path = self._artifact_path(artifact_id)

        if not path.exists():
            raise FileNotFoundError(f"Artifact not found: {path}")

        with open(path) as f:
            data = json.load(f)

        artifact = TotalsCalibrationArtifact.from_dict(data)
        logger.info(f"Loaded totals calibration artifact: {artifact_id}")
        return artifact

    def load_latest(self, mode: str = "model_only") -> Optional[TotalsCalibrationArtifact]:
        """Load most recent artifact for a mode.

        Args:
            mode: Calibration mode ("model_only" or "weather_adjusted")

        Returns:
            Most recent TotalsCalibrationArtifact or None
        """
        pattern = f"totals_{mode}_*.json"
        files = sorted(self.artifact_dir.glob(pattern), reverse=True)

        if not files:
            return None

        # Load most recent
        artifact_id = files[0].stem
        return self.load(artifact_id)

    def load_frozen(self, mode: str = "model_only") -> TotalsCalibrationArtifact:
        """Load frozen production artifact.

        Args:
            mode: Calibration mode

        Returns:
            TotalsCalibrationArtifact marked as frozen

        Raises:
            FileNotFoundError: If no frozen artifact exists
            ValueError: If artifact is not marked as frozen
        """
        artifact = self.load_latest(mode)

        if artifact is None:
            raise FileNotFoundError(
                f"No artifact found for mode '{mode}'. "
                f"Run calibration first."
            )

        if not artifact.frozen:
            logger.warning(
                f"Artifact {artifact.artifact_id} is not marked as frozen. "
                "Consider freezing it for production use."
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
            pattern = f"totals_{mode}_*.json"
        else:
            pattern = "totals_*.json"

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

        logger.info(f"Froze totals artifact: {artifact_id}")


class TotalsRunLog:
    """Append-only log of totals run executions."""

    def __init__(self, log_path: str | Path = DEFAULT_RUN_LOG_PATH):
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, metadata: TotalsRunMetadata) -> None:
        """Append run metadata to log."""
        with open(self.log_path, "a") as f:
            f.write(json.dumps(metadata.to_dict(), default=_numpy_json_encoder) + "\n")
        logger.debug(f"Logged totals run: {metadata.timestamp}")

    def read_all(self) -> list[TotalsRunMetadata]:
        """Read all log entries."""
        if not self.log_path.exists():
            return []

        entries = []
        with open(self.log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    entries.append(TotalsRunMetadata(
                        timestamp=data["timestamp"],
                        command=data["command"],
                        calibration_mode=data["calibration_mode"],
                        artifact_id=data.get("artifact_id"),
                        years_range=tuple(data["years_range"]),
                        backtest_data_hash=data.get("backtest_data_hash"),
                        git_commit=data.get("git_commit"),
                        frozen_mode=data["frozen_mode"],
                        output_files=data.get("output_files", []),
                        notes=data.get("notes", ""),
                    ))
        return entries

    def read_recent(self, n: int = 10) -> list[TotalsRunMetadata]:
        """Read most recent n log entries."""
        all_entries = self.read_all()
        return all_entries[-n:]


def verify_data_integrity(
    artifact: TotalsCalibrationArtifact,
    backtest_data_path: str,
) -> tuple[bool, str]:
    """Verify that current data matches artifact's data hash.

    Args:
        artifact: TotalsCalibrationArtifact to verify against
        backtest_data_path: Path to current backtest data

    Returns:
        (is_valid, message) tuple
    """
    if not artifact.backtest_data_hash:
        return True, "No data hash in artifact - skipping verification"

    try:
        current_hash = compute_file_hash(backtest_data_path)
    except FileNotFoundError:
        return False, f"Data file not found: {backtest_data_path}"

    if current_hash != artifact.backtest_data_hash:
        return False, (
            f"Data hash mismatch!\n"
            f"  Artifact hash: {artifact.backtest_data_hash}\n"
            f"  Current hash:  {current_hash}\n"
            f"  Data may have changed since artifact was created."
        )

    return True, "Data integrity verified"


def create_artifact_from_calibration_report(
    report,  # CalibrationReport from totals_calibration.py
    mode: str = "model_only",
    years_range: tuple[int, int] = (2023, 2025),
    backtest_data_path: Optional[str] = None,
    description: str = "",
) -> TotalsCalibrationArtifact:
    """Create TotalsCalibrationArtifact from a CalibrationReport.

    Args:
        report: CalibrationReport from calibrate_sigma()
        mode: Calibration mode
        years_range: (start_year, end_year)
        backtest_data_path: Path to backtest data file
        description: Optional description

    Returns:
        TotalsCalibrationArtifact ready to save
    """
    store = TotalsArtifactStore()
    artifact_id = store.generate_artifact_id(mode)

    # Extract coverage results for the best sigma
    coverage_68 = None
    coverage_95 = None
    best_sigma_str = str(report.best_sigma)
    if best_sigma_str in report.coverage_results:
        for cov in report.coverage_results[best_sigma_str]:
            if abs(cov.target_coverage - 0.68) < 0.01:
                coverage_68 = cov.empirical_coverage
            elif abs(cov.target_coverage - 0.95) < 0.01:
                coverage_95 = cov.empirical_coverage

    return TotalsCalibrationArtifact.from_calibration_config(
        config=report.recommended_config,
        artifact_id=artifact_id,
        years_range=years_range,
        backtest_data_path=backtest_data_path,
        description=description,
        coverage_68=coverage_68,
        coverage_95=coverage_95,
        mae_calibration=None,  # MAE not in CalibrationReport by default
    )
