#!/usr/bin/env python3
"""Profile the backtest pipeline to identify performance bottlenecks."""

import cProfile
import pstats
import io
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import the backtest script's main function
from backtest import main

if __name__ == "__main__":
    # Create profiler
    profiler = cProfile.Profile()

    # Profile the backtest with limited scope for faster profiling
    # Use 2024 only, start-week 4 to isolate in-season performance
    sys.argv = ["backtest.py", "--years", "2024", "--start-week", "4"]

    print("Starting profiling run (2024, weeks 4+)...")
    profiler.enable()

    try:
        main()
    except SystemExit:
        pass  # Ignore sys.exit() from main

    profiler.disable()

    # Print results
    print("\n" + "="*80)
    print("PROFILING RESULTS - Top 50 functions by cumulative time")
    print("="*80 + "\n")

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    stats.print_stats(50)

    print(s.getvalue())

    # Save full profile for detailed analysis
    profile_path = project_root / "backtest_profile.prof"
    stats.dump_stats(str(profile_path))
    print(f"\nFull profile saved to: {profile_path}")
    print("To analyze: python -m pstats backtest_profile.prof")
