#!/usr/bin/env python3
"""
CHECK 2: Output Schema Stability Validator

Validates that spread selection outputs conform to the contract schema
defined in data/outputs/spread_selection_output_schema.md

Usage:
    python scripts/validate_spread_selection_outputs.py data/spread_selection/outputs/
    python scripts/validate_spread_selection_outputs.py data/spread_selection/outputs/bets_primary_2026_week1.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd


class ValidationError:
    def __init__(self, file: str, field: str, message: str):
        self.file = file
        self.field = field
        self.message = message

    def __str__(self):
        return f"[{self.file}] {self.field}: {self.message}"


class SchemaValidator:
    """Validates spread selection output files against the contract schema."""

    VALID_MODES = {"primary", "ultra"}
    VALID_TIERS = {"HIGH", "MED", "BET", "PASS"}
    VALID_SIDES = {"HOME", "AWAY"}

    # Required fields in metadata
    REQUIRED_METADATA_FIELDS = {
        "mode", "label", "training_window_seasons", "training_years",
        "min_ev_threshold", "calibration", "prediction_year", "prediction_week",
        "generated_at"
    }

    # Required fields in calibration
    REQUIRED_CALIBRATION_FIELDS = {
        "intercept", "slope", "implied_breakeven_edge", "p_cover_at_zero",
        "implied_5pt_pcover", "n_train_games"
    }

    # Required fields in game/bet records
    REQUIRED_GAME_FIELDS = {
        "game_id", "home_team", "away_team", "jp_spread", "vegas_spread",
        "edge_pts", "edge_abs", "jp_favored_side", "p_cover_no_push", "ev", "tier"
    }

    def __init__(self):
        self.errors: list[ValidationError] = []
        self.warnings: list[str] = []
        self.files_validated = 0
        self.files_passed = 0

    def add_error(self, file: str, field: str, message: str):
        self.errors.append(ValidationError(file, field, message))

    def add_warning(self, message: str):
        self.warnings.append(message)

    def validate_json(self, filepath: Path) -> bool:
        """Validate a JSON output file."""
        filename = filepath.name
        self.files_validated += 1

        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.add_error(filename, "file", f"Invalid JSON: {e}")
            return False

        # Determine file type (bets or slate)
        is_bets = filename.startswith("bets_")
        is_slate = filename.startswith("slate_")

        if not is_bets and not is_slate:
            self.add_warning(f"Unknown file type: {filename}")

        # Validate metadata
        if "metadata" not in data:
            self.add_error(filename, "metadata", "Missing metadata object")
            return False

        self._validate_metadata(filename, data["metadata"])

        # Validate games/bets array
        records_key = "bets" if is_bets else "games"
        if records_key not in data:
            self.add_error(filename, records_key, f"Missing {records_key} array")
            return False

        records = data[records_key]
        if not isinstance(records, list):
            self.add_error(filename, records_key, f"{records_key} must be an array")
            return False

        for i, record in enumerate(records):
            self._validate_game_record(filename, f"{records_key}[{i}]", record)

        if not self.errors:
            self.files_passed += 1
            return True
        return False

    def validate_csv(self, filepath: Path) -> bool:
        """Validate a CSV output file."""
        filename = filepath.name
        self.files_validated += 1

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            self.add_error(filename, "file", f"Failed to read CSV: {e}")
            return False

        # Check required columns
        missing_cols = self.REQUIRED_GAME_FIELDS - set(df.columns)
        if missing_cols:
            self.add_error(filename, "columns", f"Missing required columns: {missing_cols}")

        # Validate column types and values
        for i, row in df.iterrows():
            self._validate_game_record_values(filename, f"row[{i}]", row.to_dict())

        if not self.errors:
            self.files_passed += 1
            return True
        return False

    def _validate_metadata(self, filename: str, metadata: dict):
        """Validate metadata object."""
        # Check required fields
        missing = self.REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing:
            self.add_error(filename, "metadata", f"Missing fields: {missing}")

        # Validate mode
        mode = metadata.get("mode")
        if mode and mode not in self.VALID_MODES:
            self.add_error(filename, "metadata.mode", f"Invalid mode: {mode}. Must be one of {self.VALID_MODES}")

        # Validate training_window_seasons
        tws = metadata.get("training_window_seasons")
        if tws is not None and not isinstance(tws, int):
            self.add_error(filename, "metadata.training_window_seasons",
                          f"Must be int or null, got {type(tws).__name__}")

        # Validate training_years
        ty = metadata.get("training_years")
        if ty is not None:
            if not isinstance(ty, list):
                self.add_error(filename, "metadata.training_years", "Must be an array")
            elif len(ty) == 0:
                self.add_error(filename, "metadata.training_years", "Must not be empty")

        # Validate min_ev_threshold
        mev = metadata.get("min_ev_threshold")
        if mev is not None and not isinstance(mev, (int, float)):
            self.add_error(filename, "metadata.min_ev_threshold", "Must be numeric")

        # Validate calibration
        calibration = metadata.get("calibration")
        if calibration:
            self._validate_calibration(filename, calibration)

    def _validate_calibration(self, filename: str, calibration: dict):
        """Validate calibration object."""
        missing = self.REQUIRED_CALIBRATION_FIELDS - set(calibration.keys())
        if missing:
            self.add_error(filename, "metadata.calibration", f"Missing fields: {missing}")

        # Type checks
        for field in ["intercept", "slope", "implied_breakeven_edge", "p_cover_at_zero", "implied_5pt_pcover"]:
            value = calibration.get(field)
            if value is not None and not isinstance(value, (int, float)):
                self.add_error(filename, f"metadata.calibration.{field}", "Must be numeric")

        n_train = calibration.get("n_train_games")
        if n_train is not None and not isinstance(n_train, int):
            self.add_error(filename, "metadata.calibration.n_train_games", "Must be int")

    def _validate_game_record(self, filename: str, prefix: str, record: dict):
        """Validate a game/bet record."""
        # Check required fields
        missing = self.REQUIRED_GAME_FIELDS - set(record.keys())
        if missing:
            self.add_error(filename, prefix, f"Missing fields: {missing}")

        self._validate_game_record_values(filename, prefix, record)

    def _validate_game_record_values(self, filename: str, prefix: str, record: dict):
        """Validate values in a game/bet record."""
        # p_cover_no_push: float in [0.01, 0.99]
        p_cover = record.get("p_cover_no_push")
        if p_cover is not None:
            if not isinstance(p_cover, (int, float)):
                self.add_error(filename, f"{prefix}.p_cover_no_push", "Must be numeric")
            elif not (0.01 <= p_cover <= 0.99):
                self.add_error(filename, f"{prefix}.p_cover_no_push",
                              f"Must be in [0.01, 0.99], got {p_cover}")

        # ev: float (can be negative)
        ev = record.get("ev")
        if ev is not None and not isinstance(ev, (int, float)):
            self.add_error(filename, f"{prefix}.ev", "Must be numeric")

        # edge_abs: float >= 0
        edge_abs = record.get("edge_abs")
        if edge_abs is not None:
            if not isinstance(edge_abs, (int, float)):
                self.add_error(filename, f"{prefix}.edge_abs", "Must be numeric")
            elif edge_abs < 0:
                self.add_error(filename, f"{prefix}.edge_abs", f"Must be >= 0, got {edge_abs}")

        # jp_favored_side: "HOME" or "AWAY"
        side = record.get("jp_favored_side")
        if side is not None and side not in self.VALID_SIDES:
            self.add_error(filename, f"{prefix}.jp_favored_side",
                          f"Must be one of {self.VALID_SIDES}, got {side}")

        # tier: valid tier value
        tier = record.get("tier")
        if tier is not None and tier not in self.VALID_TIERS:
            self.add_error(filename, f"{prefix}.tier",
                          f"Must be one of {self.VALID_TIERS}, got {tier}")

        # Consistency checks
        jp_spread = record.get("jp_spread")
        vegas_spread = record.get("vegas_spread")
        edge_pts = record.get("edge_pts")

        if all(v is not None for v in [jp_spread, vegas_spread, edge_pts]):
            expected_edge = jp_spread - vegas_spread
            if abs(edge_pts - expected_edge) > 0.01:
                self.add_error(filename, f"{prefix}.edge_pts",
                              f"Inconsistent: expected {expected_edge:.2f}, got {edge_pts:.2f}")

        # Side consistency
        if edge_pts is not None and side is not None:
            expected_side = "HOME" if edge_pts < 0 else "AWAY"
            # Allow edge_pts == 0 to be either (rare edge case)
            if edge_pts != 0 and side != expected_side:
                self.add_error(filename, f"{prefix}.jp_favored_side",
                              f"Inconsistent: edge_pts={edge_pts:.2f} should mean {expected_side}, got {side}")

    def validate_path(self, path: Path) -> bool:
        """Validate a file or directory."""
        if path.is_dir():
            # Validate all JSON and CSV files in directory
            json_files = list(path.glob("*.json"))
            csv_files = list(path.glob("*.csv"))

            for f in json_files:
                self.validate_json(f)
            for f in csv_files:
                self.validate_csv(f)
        elif path.suffix == ".json":
            self.validate_json(path)
        elif path.suffix == ".csv":
            self.validate_csv(path)
        else:
            print(f"Unknown file type: {path}")
            return False

        return len(self.errors) == 0

    def print_report(self) -> str:
        """Generate and print validation report."""
        lines = []
        lines.append("=" * 70)
        lines.append("SPREAD SELECTION OUTPUT SCHEMA VALIDATION REPORT")
        lines.append("=" * 70)
        lines.append(f"Files validated: {self.files_validated}")
        lines.append(f"Files passed: {self.files_passed}")
        lines.append(f"Total errors: {len(self.errors)}")
        lines.append(f"Total warnings: {len(self.warnings)}")
        lines.append("")

        if self.errors:
            lines.append("ERRORS:")
            lines.append("-" * 70)
            for err in self.errors:
                lines.append(f"  ✗ {err}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 70)
            for warn in self.warnings:
                lines.append(f"  ⚠ {warn}")
            lines.append("")

        if not self.errors:
            lines.append("✓ ALL FILES PASSED VALIDATION")
        else:
            lines.append("✗ VALIDATION FAILED")

        lines.append("=" * 70)

        report = "\n".join(lines)
        print(report)
        return report


def main():
    parser = argparse.ArgumentParser(
        description="Validate spread selection output files against the contract schema"
    )
    parser.add_argument(
        "path",
        type=str,
        help="File or directory to validate"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for validation report (markdown)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()
    path = Path(args.path)

    if not path.exists():
        print(f"Error: Path does not exist: {path}")
        sys.exit(1)

    validator = SchemaValidator()
    passed = validator.validate_path(path)
    report = validator.print_report()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(f"# Spread Selection Output Schema Validation\n\n")
            f.write(f"**Path validated:** `{path}`\n\n")
            f.write(f"**Result:** {'✓ PASSED' if passed else '✗ FAILED'}\n\n")
            f.write("```\n")
            f.write(report)
            f.write("\n```\n")
        print(f"\nReport written to: {args.output}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
