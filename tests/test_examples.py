"""Every script in examples/ runs offline and exits 0."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.py"))

REQUIRES = {
    "02_langchain_fake_llm.py": "langchain_core",
    "03_llamaindex_mock.py": "llama_index.core",
    "04_openai_wrapper_mock.py": "openai",
    "05_anthropic_wrapper_mock.py": "anthropic",
}


@pytest.mark.parametrize("script", EXAMPLES, ids=[p.name for p in EXAMPLES])
def test_example_runs(script: Path) -> None:
    dep = REQUIRES.get(script.name)
    if dep:
        try:
            __import__(dep)
        except Exception as exc:  # upstream import failures on old interpreters
            pytest.skip(f"{dep} not importable: {exc.__class__.__name__}")
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, cwd=str(ROOT), timeout=120)
    assert proc.returncode == 0, f"{script.name} failed:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "verify" in proc.stdout.lower()
