"""Sunday pipeline orchestrator — runs odds capture + all models."""

from __future__ import annotations

import logging
from collections import OrderedDict

from bot.config import PIPELINE_STEP_TIMEOUT, PROJECT_ROOT
from bot.state_manager import pipeline_completed, record_error, record_pipeline_run
from bot.task_runner import PYTHON, run_command_async

logger = logging.getLogger(__name__)


async def run_pipeline(year: int, week: int, force: bool = False) -> dict:
    """Run the full Sunday pipeline. Returns dict of step -> result info.

    Steps:
        1. Capture opening odds
        2. Run spread predictions (run_weekly.py)
        3. Run spread EV selection (run_spread_weekly.py)
        4. Run totals predictions (run_weekly_totals.py)
        5. Run moneyline selection (run_moneyline_weekly.py)
    """
    if not force and pipeline_completed(year, week):
        return {"pipeline": {"status": "skipped", "duration": 0, "error": "Already ran this week"}}

    results: dict[str, dict] = OrderedDict()

    steps = [
        (
            "Capture Opening Odds",
            [PYTHON, "scripts/weekly_odds_capture.py", "--opening", "--year", str(year), "--week", str(week)],
        ),
        (
            "Spread Predictions",
            [PYTHON, "scripts/run_weekly.py", "--year", str(year), "--week", str(week), "--no-wait", "--export-slate"],
        ),
        (
            "Spread EV Selection",
            [PYTHON, "scripts/run_spread_weekly.py", "--year", str(year), "--week", str(week)],
        ),
        (
            "Totals Predictions",
            [PYTHON, "scripts/run_weekly_totals.py", "--year", str(year), "--week", str(week)],
        ),
        (
            "Moneyline Selection",
            [PYTHON, "scripts/run_moneyline_weekly.py", "--year", str(year), "--week", str(week)],
        ),
        (
            "Refresh Ratings",
            [PYTHON, "scripts/show_ratings.py", str(year), "all", "--refresh"],
        ),
    ]

    for step_name, cmd in steps:
        logger.info(f"Pipeline step: {step_name}")
        try:
            result = await run_command_async(
                cmd,
                cwd=str(PROJECT_ROOT),
                timeout=PIPELINE_STEP_TIMEOUT,
            )
            if result.returncode == 0:
                results[step_name] = {"status": "success", "duration": result.duration}
            else:
                error_msg = result.stderr[:300] or result.stdout[:300]
                results[step_name] = {"status": "error", "duration": result.duration, "error": error_msg}
                record_error(f"{step_name}: {error_msg}")
        except Exception as e:
            results[step_name] = {"status": "error", "duration": 0, "error": str(e)[:300]}
            record_error(f"{step_name}: {e}")

    record_pipeline_run(year, week, results)
    return results
