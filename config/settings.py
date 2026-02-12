"""Application settings and configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class Settings:
    """Application configuration settings."""

    # API Configuration
    cfbd_api_key: str = field(default_factory=lambda: os.getenv("CFBD_API_KEY", ""))

    # Data Configuration
    historical_years: tuple = (2021, 2022, 2023, 2024, 2025)
    current_year: int = 2025

    # Model Hyperparameters
    # Alpha=10 is appropriate for FBS-only data (~130 teams, ~800 games/year)
    # Higher alpha (150) was for all divisions (700+ teams)
    ridge_alpha: float = field(
        default_factory=lambda: float(os.getenv("RIDGE_ALPHA", "10"))
    )

    # Garbage Time Thresholds (points differential by quarter)
    garbage_time_q1: int = 28
    garbage_time_q2: int = 24
    garbage_time_q3: int = 21
    garbage_time_q4: int = 16

    # Luck Regression Factor
    luck_regression_factor: float = 0.5

    # Home Field Advantage (baseline, will be calculated from data)
    base_hfa: float = 2.8

    # Situational Adjustments (points)
    bye_week_advantage: float = 1.5
    short_week_penalty: float = -2.5  # Penalty when on short week vs normal/rested opponent
    letdown_penalty: float = -2.0
    letdown_away_multiplier: float = 1.25  # Letdown worse on the road (sleepy noon kickoff)
    lookahead_penalty: float = -1.5
    sandwich_extra_penalty: float = -1.0  # Extra penalty when BOTH letdown AND lookahead
    rivalry_underdog_boost: float = 0.0  # Ablation tested 2026-02-12: removing improved Core 5+ Edge
    consecutive_road_penalty: float = -1.5  # Penalty for 2nd consecutive road game

    # Travel Adjustment
    timezone_adjustment: float = 0.5  # points per timezone crossed

    # Special Teams Shrinkage (Empirical Bayes toward league average)
    # REJECTED (2026-02-10): All k values tested degraded ATS despite slight MAE improvement.
    # k=6-8 maintained 5+ Edge but degraded 3+ Edge and All games. Per "Edge > Accuracy" rule, disabled.
    # Infrastructure preserved for future experimentation via CLI --st-k-fg/--st-k-punt/--st-k-ko.
    st_shrink_enabled: bool = False  # Master toggle for ST shrinkage (DISABLED)
    # Opportunity-based k-parameters (shrink = n / (n + k))
    # Higher k = more shrinkage; lower k = less shrinkage
    st_k_fg: float = 6.0  # FG attempts needed to trust rating fully
    st_k_punt: float = 6.0  # Punts needed to trust rating fully
    st_k_ko: float = 6.0  # Kickoff events needed to trust rating fully

    # Special Teams Spread Impact Cap (Approach B: margin-level capping)
    # Caps the ST differential's effect on the spread, not the rating itself.
    # This preserves ST rating magnitude while limiting influence on final spread.
    # APPROVED (2026-02-10): Cap 2.5 improved 5+ Edge +0.3% with minimal spread compression.
    # Sweep tested: 1.5, 2.0, 2.5, 3.0 — cap 2.5 best for both Core and Full Season.
    st_spread_cap: Optional[float] = 2.5

    # FCS Strength Estimator (Dynamic penalties based on prior FBS-vs-FCS game margins)
    # Replaces static ELITE_FCS_TEAMS frozenset with walk-forward-safe, data-driven estimates.
    # Uses Bayesian shrinkage toward baseline when data is sparse.
    # Calibrated to match static baseline: elite FCS (+18), avg FCS (+32), weak FCS (capped +45).
    fcs_static: bool = False  # If True, use static elite list instead of dynamic estimator
    fcs_k: float = 8.0  # Shrinkage k (games for 50% trust in data)
    fcs_baseline_margin: float = -28.0  # Prior margin for unknown FCS (FCS - FBS, negative = FCS loses)
    fcs_min_penalty: float = 10.0  # Floor for elite FCS teams
    fcs_max_penalty: float = 45.0  # Ceiling for weak FCS teams
    fcs_slope: float = 0.8  # Penalty increase per point of avg loss to FBS
    fcs_intercept: float = 10.0  # Base penalty (elite FCS with 0 avg loss)

    # Vegas Comparison
    value_threshold: float = field(
        default_factory=lambda: float(os.getenv("VALUE_THRESHOLD", "3.0"))
    )
    vegas_provider: str = "consensus"

    # Email Notification Settings
    smtp_host: str = field(default_factory=lambda: os.getenv("SMTP_HOST", ""))
    smtp_port: int = field(
        default_factory=lambda: int(os.getenv("SMTP_PORT", "587"))
    )
    smtp_user: str = field(default_factory=lambda: os.getenv("SMTP_USER", ""))
    smtp_password: str = field(default_factory=lambda: os.getenv("SMTP_PASSWORD", ""))
    notification_email: str = field(
        default_factory=lambda: os.getenv("NOTIFICATION_EMAIL", "")
    )

    # Retry Configuration
    max_retries: int = 3
    retry_base_delay: float = 3600.0  # 1 hour in seconds
    data_check_interval: float = 300.0  # 5 minutes

    # Paths
    @property
    def project_root(self) -> Path:
        return Path(__file__).parent.parent

    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"

    @property
    def historical_dir(self) -> Path:
        return self.data_dir / "historical"

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    def validate(self) -> list[str]:
        """Validate required settings. Returns list of errors."""
        errors = []
        if not self.cfbd_api_key:
            errors.append("CFBD_API_KEY is required. Set it in .env file.")
        return errors


# Singleton instance
_settings: Optional[Settings] = None


def get_settings() -> Settings:
    """Get the settings singleton."""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
