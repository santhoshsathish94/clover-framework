"""Controlled experiment environment for Clover.

Start deliberately small. Every environment action returns observable output.
"""
import subprocess, sys

def run_python(source: str, timeout: int = 10):
    result = subprocess.run(
        [sys.executable, "-c", source],
        capture_output=True, text=True, timeout=timeout
    )
    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }
