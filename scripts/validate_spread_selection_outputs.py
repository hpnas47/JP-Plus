#!/usr/bin/env python3
"""
CHECK 2: Output Schema Stability Validator

Validates that spread selection outputs conform to the contract schema
defined in data/outputs/spread_selection_output_schema.md

Supports both:
- List A (PRIMARY/ULTRA): EV-based calibrated selection outputs
- List B (PHASE1_EDGE_CONFIRM): Edge-based with SP+ confirmation outputs

Usage:
    python scripts/validate_spread_selection_outputs.py data/spread_selection/outputs/
    python scripts/validate_spread_selection_outputs.py data/spread_selection/outputs/bets_primary_2026_week1.json
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import pandas as pd


class ValidationError:
    def __init__(self, file: str, field: str, message: str):
        self.file = file
        self.field = field
        self.message = message

    def __str__(self):
        return f"[{self.file}] {self.field}: {self.message}"


class SchemaValidator:
    """Validates spread selection output files against the contract schema.

    Supports three output types:
    - List A (ENGINE_EV PRIMARY/ULTRA): EV-based selection with calibration metadata
    - List B (PHASE1_EDGE): Edge-based selection for Phase 1
    - Week Summary: Summary of bets generated for a week
    """

    # Valid modes for List A
    VALID_LISTA_MODES = {"primary", "ultra"}

    # Valid list families
    VALID_LIST_FAMILIES = {"ENGINE_EV", "PHASE1_EDGE"}

    # Valid list names
    VALID_LIST_NAMES = {"PRIMARY", "ULTRA", "EDGE_BASELINE", "EDGE_HYBRID_VETO_2"}

    # Valid selection bases
    VALID_SELECTION_BASES = {"EV", "EDGE"}

    # Valid tiers for List A
    VALID_TIERS = {"HIGH", "MED", "BET", "PASS"}

    # Valid bet sides
    VALID_SIDES = {"HOME", "AWAY", "NO_BET"}

    # Valid SP+ gate categories (deprecated but kept for backward compat)
    VALID_SP_GATE_CATEGORIES = {"confirm", "neutral", "oppose", "no_bet_sp", "missing"}

    # Valid Phase1 Edge results
    VALID_EDGE_RESULTS = {"selected", "vetoed", "rejected_edge", "not_phase1"}

    # =========================================================================
    # List A (ENGINE_EV PRIMARY/ULTRA) Schema
    # =========================================================================

    # Required fields in List A metadata
    LISTA_REQUIRED_METADATA_FIELDS = {
        "mode", "label", "training_window_seasons", "training_years",
        "min_ev_threshold", "calibration", "prediction_year", "prediction_week",
        "generated_at"
    }

    # Required fields in calibration
    LISTA_REQUIRED_CALIBRATION_FIELDS = {
        "intercept", "slope", "implied_breakeven_edge", "p_cover_at_zero",
        "implied_5pt_pcover", "n_train_games"
    }

    # Required fields in List A game/bet records
    LISTA_REQUIRED_GAME_FIELDS = {
        "game_id", "home_team", "away_team", "jp_spread", "vegas_spread",
        "edge_pts", "edge_abs", "jp_favored_side", "p_cover_no_push", "ev", "tier"
    }

    # New required metadata fields (V2)
    LISTA_V2_METADATA_FIELDS = {
        "list_family", "list_name", "selection_basis", "is_official_engine",
        "execution_default", "line_type", "rationale"
    }

    # =========================================================================
    # List B (PHASE1_EDGE) Schema
    # =========================================================================

    # Required fields in List B metadata
    LISTB_REQUIRED_METADATA_FIELDS = {
        "list_family", "list_name", "selection_basis", "is_official_engine",
        "execution_default", "line_type", "jp_edge_min",
        "year", "week", "n_candidates", "n_selected", "generated_at"
    }

    # Required fields in List B game/bet records
    LISTB_REQUIRED_GAME_FIELDS = {
        "game_id", "season", "week", "home_team", "away_team",
        "jp_spread", "vegas_spread", "edge_pts", "edge_abs",
        "bet_side", "bet_team",
        "list_family", "list_name", "selection_basis",
        "is_official_engine", "execution_default", "line_type", "rationale",
        "veto_applied", "veto_reason"
    }

    # =========================================================================
    # Week Summary Schema
    # =========================================================================

    WEEK_SUMMARY_REQUIRED_FIELDS = {
        "year", "week", "generated_at",
        "engine_ev_primary_count", "phase1_edge_baseline_count",
        "config", "files"
    }

    def __init__(self):
        self.errors: list[ValidationError] = []
        self.warnings: list[str] = []
        self.files_validated = 0
        self.files_passed = 0
        self.lista_files = []
        self.listb_files = []
        self.overlap_files = []

    def add_error(self, file: str, field: str, message: str):
        self.errors.append(ValidationError(file, field, message))

    def add_warning(self, message: str):
        self.warnings.append(message)

    def _detect_output_type(self, filepath: Path, data: dict) -> str:
        """Detect whether file is List A, List B, or Week Summary.

        Returns:
            "lista" for EV-based PRIMARY/ULTRA
            "listb" for PHASE1_EDGE
            "week_summary" for week summary files
            "unknown" if cannot determine
        """
        filename = filepath.name

        # Check filename pattern
        if filename.startswith("week_summary_"):
            return "week_summary"

        if "phase1_edge" in filename:
            return "listb"

        if filename.startswith("bets_primary") or filename.startswith("bets_ultra"):
            return "lista"

        if filename.startswith("slate_primary") or filename.startswith("slate_ultra"):
            return "lista"

        # Check metadata
        metadata = data.get("metadata", {})

        # New schema: check list_family
        list_family = metadata.get("list_family")
        if list_family == "ENGINE_EV":
            return "lista"
        if list_family == "PHASE1_EDGE":
            return "listb"

        # Legacy schema: check strategy
        if metadata.get("strategy") == "PHASE1_EDGE_CONFIRM":
            return "listb"

        if metadata.get("mode") in self.VALID_LISTA_MODES:
            return "lista"

        # Check for week summary fields
        if "engine_ev_primary_count" in data or "phase1_edge_baseline_count" in data:
            return "week_summary"

        # Check records
        records = data.get("bets", data.get("games", []))
        if records and len(records) > 0:
            first_record = records[0]
            # New schema: check list_family in record
            if first_record.get("list_family") == "ENGINE_EV":
                return "lista"
            if first_record.get("list_family") == "PHASE1_EDGE":
                return "listb"
            # Legacy schema
            if "strategy_name" in first_record and first_record.get("strategy_name") == "PHASE1_EDGE_CONFIRM":
                return "listb"
            if "p_cover_no_push" in first_record and "ev" in first_record:
                return "lista"
            if "bet_side" in first_record and "sp_gate_category" in first_record:
                return "listb"

        return "unknown"

    def validate_json(self, filepath: Path) -> bool:
        """Validate a JSON output file."""
        filename = filepath.name
        self.files_validated += 1
        errors_before = len(self.errors)

        try:
            with open(filepath) as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            self.add_error(filename, "file", f"Invalid JSON: {e}")
            return False

        # Detect output type
        output_type = self._detect_output_type(filepath, data)

        if output_type == "lista":
            self.lista_files.append(filename)
            self._validate_lista_json(filename, data)
        elif output_type == "listb":
            self.listb_files.append(filename)
            self._validate_listb_json(filename, data)
        elif output_type == "week_summary":
            self._validate_week_summary_json(filename, data)
        else:
            self.add_warning(f"Could not determine output type for: {filename}")
            return False

        if len(self.errors) == errors_before:
            self.files_passed += 1
            return True
        return False

    def _validate_lista_json(self, filename: str, data: dict):
        """Validate a List A (PRIMARY/ULTRA) JSON file."""
        is_bets = filename.startswith("bets_")
        is_slate = filename.startswith("slate_")

        # Validate metadata
        if "metadata" not in data:
            self.add_error(filename, "metadata", "Missing metadata object")
            return

        self._validate_lista_metadata(filename, data["metadata"])

        # Validate games/bets array
        records_key = "bets" if is_bets else "games"
        if records_key not in data:
            self.add_error(filename, records_key, f"Missing {records_key} array")
            return

        records = data[records_key]
        if not isinstance(records, list):
            self.add_error(filename, records_key, f"{records_key} must be an array")
            return

        for i, record in enumerate(records):
            self._validate_lista_game_record(filename, f"{records_key}[{i}]", record)

    def _validate_lista_metadata(self, filename: str, metadata: dict):
        """Validate List A metadata object."""
        # Check required fields
        missing = self.LISTA_REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing:
            self.add_error(filename, "metadata", f"Missing fields: {missing}")

        # Validate mode
        mode = metadata.get("mode")
        if mode and mode not in self.VALID_LISTA_MODES:
            self.add_error(filename, "metadata.mode",
                          f"Invalid mode: {mode}. Must be one of {self.VALID_LISTA_MODES}")

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
        missing = self.LISTA_REQUIRED_CALIBRATION_FIELDS - set(calibration.keys())
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

    def _validate_lista_game_record(self, filename: str, prefix: str, record: dict):
        """Validate a List A game/bet record."""
        # Check required fields
        missing = self.LISTA_REQUIRED_GAME_FIELDS - set(record.keys())
        if missing:
            self.add_error(filename, prefix, f"Missing fields: {missing}")

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

        # EV consistency cross-check: ev should match p_cover at -110 odds
        # Assumes -110 placeholder odds; update when real odds are integrated.
        if (p_cover is not None and ev is not None
                and isinstance(p_cover, (int, float))
                and isinstance(ev, (int, float))):
            expected_ev = p_cover * (100 / 110) - (1 - p_cover)
            if abs(ev - expected_ev) > 0.01:
                self.add_error(
                    filename, f"{prefix}.ev",
                    f"EV inconsistent with p_cover at assumed -110 odds: "
                    f"expected {expected_ev:.4f}, got {ev:.4f}"
                )

        # edge_abs: float >= 0
        edge_abs = record.get("edge_abs")
        if edge_abs is not None:
            if not isinstance(edge_abs, (int, float)):
                self.add_error(filename, f"{prefix}.edge_abs", "Must be numeric")
            elif edge_abs < 0:
                self.add_error(filename, f"{prefix}.edge_abs", f"Must be >= 0, got {edge_abs}")

        # jp_favored_side: "HOME" or "AWAY"
        side = record.get("jp_favored_side")
        if side is not None and side not in {"HOME", "AWAY"}:
            self.add_error(filename, f"{prefix}.jp_favored_side",
                          f"Must be HOME or AWAY, got {side}")

        # tier: valid tier value
        tier = record.get("tier")
        if tier is not None and tier not in self.VALID_TIERS:
            self.add_error(filename, f"{prefix}.tier",
                          f"Must be one of {self.VALID_TIERS}, got {tier}")

        # Consistency checks
        self._validate_edge_consistency(filename, prefix, record, side_field="jp_favored_side")

    def _validate_listb_json(self, filename: str, data: dict):
        """Validate a List B (PHASE1_EDGE) JSON file."""
        # Validate metadata
        if "metadata" not in data:
            self.add_error(filename, "metadata", "Missing metadata object")
            return

        self._validate_listb_metadata(filename, data["metadata"])

        # Validate bets array
        if "bets" not in data:
            self.add_error(filename, "bets", "Missing bets array")
            return

        records = data["bets"]
        if not isinstance(records, list):
            self.add_error(filename, "bets", "bets must be an array")
            return

        for i, record in enumerate(records):
            self._validate_listb_game_record(filename, f"bets[{i}]", record)

    def _validate_listb_metadata(self, filename: str, metadata: dict):
        """Validate List B metadata object."""
        # Check required fields (new schema)
        missing = self.LISTB_REQUIRED_METADATA_FIELDS - set(metadata.keys())
        if missing:
            # Allow legacy schema
            if "strategy" not in metadata:
                self.add_error(filename, "metadata", f"Missing fields: {missing}")

        # Validate list_family (new schema)
        list_family = metadata.get("list_family")
        if list_family is not None and list_family != "PHASE1_EDGE":
            self.add_error(filename, "metadata.list_family",
                          f"Must be 'PHASE1_EDGE', got {list_family}")

        # Validate selection_basis
        basis = metadata.get("selection_basis")
        if basis is not None and basis != "EDGE":
            self.add_error(filename, "metadata.selection_basis",
                          f"Must be 'EDGE', got {basis}")

        # Validate is_official_engine (must be False)
        is_official = metadata.get("is_official_engine")
        if is_official is True:
            self.add_error(filename, "metadata.is_official_engine",
                          "List B must have is_official_engine=False")

        # Validate execution_default (must be False)
        exec_default = metadata.get("execution_default")
        if exec_default is True:
            self.add_error(filename, "metadata.execution_default",
                          "List B must have execution_default=False")

        # Validate thresholds
        for field in ["jp_edge_min"]:
            value = metadata.get(field)
            if value is not None and not isinstance(value, (int, float)):
                self.add_error(filename, f"metadata.{field}", "Must be numeric")

        # Validate counts
        for field in ["n_candidates", "n_selected"]:
            value = metadata.get(field)
            if value is not None and not isinstance(value, int):
                self.add_error(filename, f"metadata.{field}", "Must be int")

    def _validate_listb_game_record(self, filename: str, prefix: str, record: dict):
        """Validate a List B game/bet record."""
        # Check required fields (new schema)
        missing = self.LISTB_REQUIRED_GAME_FIELDS - set(record.keys())
        if missing:
            # Allow legacy schema with sp_gate fields
            if "sp_gate_category" not in record:
                self.add_error(filename, prefix, f"Missing fields: {missing}")

        # bet_side: "HOME" or "AWAY" (not NO_BET for selected bets)
        side = record.get("bet_side")
        if side is not None and side not in {"HOME", "AWAY"}:
            self.add_error(filename, f"{prefix}.bet_side",
                          f"Must be HOME or AWAY, got {side}")

        # list_family (new schema)
        list_family = record.get("list_family")
        if list_family is not None and list_family not in self.VALID_LIST_FAMILIES:
            self.add_error(filename, f"{prefix}.list_family",
                          f"Must be one of {self.VALID_LIST_FAMILIES}, got {list_family}")

        # selection_basis
        basis = record.get("selection_basis")
        if basis is not None and basis not in self.VALID_SELECTION_BASES:
            self.add_error(filename, f"{prefix}.selection_basis",
                          f"Must be one of {self.VALID_SELECTION_BASES}, got {basis}")

        # is_official_engine (must be False for List B)
        is_official = record.get("is_official_engine")
        if is_official is True:
            self.add_error(filename, f"{prefix}.is_official_engine",
                          "List B records must have is_official_engine=False")

        # veto_applied: must be bool
        veto_applied = record.get("veto_applied")
        if veto_applied is not None and not isinstance(veto_applied, bool):
            self.add_error(filename, f"{prefix}.veto_applied",
                          f"Must be boolean, got {type(veto_applied).__name__}")

        # Legacy schema: sp_gate_category (if present)
        sp_cat = record.get("sp_gate_category")
        if sp_cat is not None and sp_cat not in self.VALID_SP_GATE_CATEGORIES:
            self.add_error(filename, f"{prefix}.sp_gate_category",
                          f"Must be one of {self.VALID_SP_GATE_CATEGORIES}, got {sp_cat}")

        # Legacy schema: sp_gate_passed (if present)
        # Only enforce sp_gate_passed=True for non-vetoed bets (vetoed/rejected records legitimately have False)
        sp_passed = record.get("sp_gate_passed")
        # Reuse veto_applied from line 466 (no need to re-read from record)
        is_vetoed = veto_applied is True or str(veto_applied).lower() == "true"
        if sp_passed is not None and not is_vetoed:
            # Use equality check (not identity) to handle CSV-parsed strings/ints
            sp_passed_bool = sp_passed is True or str(sp_passed).lower() == "true"
            if not sp_passed_bool:
                self.add_error(filename, f"{prefix}.sp_gate_passed",
                              f"Selected (non-vetoed) bets must have sp_gate_passed=True, got {sp_passed}")

        # edge_abs: float >= 0
        edge_abs = record.get("edge_abs")
        if edge_abs is not None:
            if not isinstance(edge_abs, (int, float)):
                self.add_error(filename, f"{prefix}.edge_abs", "Must be numeric")
            elif edge_abs < 0:
                self.add_error(filename, f"{prefix}.edge_abs", f"Must be >= 0, got {edge_abs}")

        # Consistency checks
        self._validate_edge_consistency(filename, prefix, record, side_field="bet_side")

    def _validate_edge_consistency(self, filename: str, prefix: str, record: dict, side_field: str):
        """Validate edge_pts consistency with jp_spread, vegas_spread, and side."""
        jp_spread = record.get("jp_spread")
        vegas_spread = record.get("vegas_spread")
        edge_pts = record.get("edge_pts")

        if all(v is not None for v in [jp_spread, vegas_spread, edge_pts]):
            expected_edge = jp_spread - vegas_spread
            if abs(edge_pts - expected_edge) > 0.01:
                self.add_error(filename, f"{prefix}.edge_pts",
                              f"Inconsistent: expected {expected_edge:.2f}, got {edge_pts:.2f}")

        # Side consistency
        side = record.get(side_field)
        if edge_pts is not None and side is not None:
            expected_side = "HOME" if edge_pts < 0 else "AWAY"
            # Allow edge_pts == 0 to be either (rare edge case)
            if edge_pts != 0 and side != expected_side:
                self.add_error(filename, f"{prefix}.{side_field}",
                              f"Inconsistent: edge_pts={edge_pts:.2f} should mean {expected_side}, got {side}")

    def _validate_week_summary_json(self, filename: str, data: dict):
        """Validate a week summary JSON file."""
        # Check required fields
        missing = self.WEEK_SUMMARY_REQUIRED_FIELDS - set(data.keys())
        if missing:
            self.add_error(filename, "root", f"Missing fields: {missing}")

        # Validate year and week
        year = data.get("year")
        if year is not None and not isinstance(year, int):
            self.add_error(filename, "year", "Must be int")

        week = data.get("week")
        if week is not None and not isinstance(week, int):
            self.add_error(filename, "week", "Must be int")

        # Validate counts
        for field in ["engine_ev_primary_count", "engine_ev_ultra_count",
                      "phase1_edge_baseline_count", "phase1_edge_vetoed_count"]:
            value = data.get(field)
            if value is not None and not isinstance(value, int):
                self.add_error(filename, field, f"Must be int, got {type(value).__name__}")

        # Validate config
        config = data.get("config")
        if config is not None:
            if not isinstance(config, dict):
                self.add_error(filename, "config", "Must be object")

        # Validate files
        files = data.get("files")
        if files is not None:
            if not isinstance(files, dict):
                self.add_error(filename, "files", "Must be object")

    # =========================================================================
    # Overlap Report Schema
    # =========================================================================

    OVERLAP_REQUIRED_FIELDS = {
        "game_id", "in_engine_primary", "engine_side", "engine_ev",
        "in_phase1_edge", "phase1_side", "phase1_edge_abs",
        "side_agrees", "conflict", "recommended_resolution"
    }

    # Legacy overlap fields (for backward compat)
    OVERLAP_LEGACY_FIELDS = {
        "game_id", "in_primary", "primary_side", "primary_ev",
        "in_phase1_edge_confirm", "phase1_side", "phase1_edge_abs",
        "side_agrees", "conflict"
    }

    def validate_csv(self, filepath: Path) -> bool:
        """Validate a CSV output file."""
        filename = filepath.name
        self.files_validated += 1
        errors_before = len(self.errors)

        try:
            df = pd.read_csv(filepath)
        except Exception as e:
            self.add_error(filename, "file", f"Failed to read CSV: {e}")
            return False

        # Detect output type
        if "overlap_" in filename:
            # Overlap report - different schema
            self._validate_overlap_csv(filename, df)
        elif "phase1_edge" in filename:
            self.listb_files.append(filename)
            self._validate_listb_csv(filename, df)
        else:
            self.lista_files.append(filename)
            self._validate_lista_csv(filename, df)

        if len(self.errors) == errors_before:
            self.files_passed += 1
            return True
        return False

    def _validate_lista_csv(self, filename: str, df: pd.DataFrame):
        """Validate a List A CSV file."""
        # Check required columns
        missing_cols = self.LISTA_REQUIRED_GAME_FIELDS - set(df.columns)
        if missing_cols:
            self.add_error(filename, "columns", f"Missing required columns: {missing_cols}")

        # Validate values
        for i, row in df.iterrows():
            record = row.to_dict()
            self._validate_lista_game_record(filename, f"row[{i}]", record)

    def _validate_listb_csv(self, filename: str, df: pd.DataFrame):
        """Validate a List B CSV file."""
        # Check required columns
        missing_cols = self.LISTB_REQUIRED_GAME_FIELDS - set(df.columns)
        if missing_cols:
            self.add_error(filename, "columns", f"Missing required columns: {missing_cols}")

        # Validate values
        for i, row in df.iterrows():
            record = row.to_dict()
            self._validate_listb_game_record(filename, f"row[{i}]", record)

    def _validate_overlap_csv(self, filename: str, df: pd.DataFrame):
        """Validate an overlap report CSV file."""
        self.overlap_files.append(filename)

        # Check required columns - accept either new or legacy schema
        new_schema_missing = self.OVERLAP_REQUIRED_FIELDS - set(df.columns)
        legacy_schema_missing = self.OVERLAP_LEGACY_FIELDS - set(df.columns)

        if new_schema_missing and legacy_schema_missing:
            # Neither schema is satisfied
            self.add_error(filename, "columns",
                          f"Missing required columns. New schema missing: {new_schema_missing}. "
                          f"Legacy schema missing: {legacy_schema_missing}")
            return

        # Determine which schema is in use
        using_new_schema = len(new_schema_missing) == 0

        # Validate values
        for i, row in df.iterrows():
            record = row.to_dict()

            # Boolean fields (adapt to schema)
            if using_new_schema:
                bool_fields = ["in_engine_primary", "in_phase1_edge"]
            else:
                bool_fields = ["in_primary", "in_phase1_edge_confirm"]

            for field in bool_fields:
                value = record.get(field)
                if value is not None and not isinstance(value, bool):
                    # CSV reads bools as True/False strings
                    if str(value).lower() not in ("true", "false"):
                        self.add_error(filename, f"row[{i}].{field}", f"Must be boolean, got {value}")

            # side_agrees and conflict should be bool or null
            for field in ["side_agrees", "conflict"]:
                value = record.get(field)
                if pd.notna(value) and str(value).lower() not in ("true", "false"):
                    self.add_error(filename, f"row[{i}].{field}", f"Must be boolean or null, got {value}")

            # Side fields should be HOME/AWAY or null (adapt to schema)
            if using_new_schema:
                side_fields = ["engine_side", "phase1_side"]
            else:
                side_fields = ["primary_side", "phase1_side"]

            for field in side_fields:
                value = record.get(field)
                if pd.notna(value) and value not in {"HOME", "AWAY"}:
                    self.add_error(filename, f"row[{i}].{field}",
                                  f"Must be HOME, AWAY, or null, got {value}")

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

        lines.append(f"List A files (ENGINE_EV PRIMARY/ULTRA): {len(self.lista_files)}")
        for f in self.lista_files:
            lines.append(f"  - {f}")
        lines.append("")

        lines.append(f"List B files (PHASE1_EDGE): {len(self.listb_files)}")
        for f in self.listb_files:
            lines.append(f"  - {f}")
        lines.append("")

        lines.append(f"Overlap report files: {len(self.overlap_files)}")
        for f in self.overlap_files:
            lines.append(f"  - {f}")
        lines.append("")

        if self.errors:
            lines.append("ERRORS:")
            lines.append("-" * 70)
            for err in self.errors:
                lines.append(f"  X {err}")
            lines.append("")

        if self.warnings:
            lines.append("WARNINGS:")
            lines.append("-" * 70)
            for warn in self.warnings:
                lines.append(f"  ! {warn}")
            lines.append("")

        if not self.errors:
            lines.append("V ALL FILES PASSED VALIDATION")
        else:
            lines.append("X VALIDATION FAILED")

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
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Result:** {'V PASSED' if passed else 'X FAILED'}\n\n")
            f.write("```\n")
            f.write(report)
            f.write("\n```\n")
        print(f"\nReport written to: {args.output}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
