"""Script execution wrapper for bot commands."""

import asyncio
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent

# Use the exact same Python that's running the bot for all subprocesses
PYTHON = sys.executable  # e.g. /Library/Frameworks/Python.framework/Versions/3.12/bin/python3

# Build an environment dict for subprocesses
_ENV = os.environ.copy()
_ENV["PYTHONPATH"] = str(PROJECT_ROOT) + ":" + _ENV.get("PYTHONPATH", "")
_ENV["HOME"] = os.path.expanduser("~")


@dataclass
class CommandResult:
    returncode: int
    stdout: str
    stderr: str
    duration: float


def run_display_script(script_name: str, args: list[str] | None = None) -> CommandResult:
    """Run a fast display script synchronously. Returns captured stdout."""
    cmd = [PYTHON, f"scripts/{script_name}"] + (args or [])
    start = time.monotonic()
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
        env=_ENV,
    )
    duration = time.monotonic() - start
    return CommandResult(
        returncode=result.returncode,
        stdout=result.stdout,
        stderr=result.stderr,
        duration=duration,
    )


async def run_command_async(
    cmd: list[str],
    cwd: str | None = None,
    timeout: int = 300,
) -> CommandResult:
    """Run a command asynchronously. For long-running pipeline steps."""
    start = time.monotonic()
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=cwd or str(PROJECT_ROOT),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=_ENV,
    )
    try:
        stdout_bytes, stderr_bytes = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.communicate()
        duration = time.monotonic() - start
        return CommandResult(
            returncode=-1,
            stdout="",
            stderr=f"Command timed out after {timeout}s",
            duration=duration,
        )
    duration = time.monotonic() - start
    return CommandResult(
        returncode=proc.returncode or 0,
        stdout=stdout_bytes.decode(errors="replace"),
        stderr=stderr_bytes.decode(errors="replace"),
        duration=duration,
    )
