import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run_help(script_name: str) -> str:
    cmd = [sys.executable, str(ROOT / script_name), "--help"]
    p = subprocess.run(cmd, capture_output=True, text=True, check=True)
    return p.stdout


def test_run_browser_agent_help_has_compat_flags():
    out = _run_help("run_browser_agent.py")
    assert "--query" in out
    assert "--perception" in out
    assert "--env-backend" in out
    assert "--legacy-fallback" in out


def test_run_agent_dual_mode_help_kept():
    out = _run_help("run_agent_dual_mode.py")
    assert "--query" in out
    assert "--verbal-only" in out
    assert "--visual-only" in out
