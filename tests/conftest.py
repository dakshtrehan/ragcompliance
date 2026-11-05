"""Shared pytest configuration."""

from __future__ import annotations

from pathlib import Path

import pytest


def _llamaindex_importable() -> bool:
    """``pytest.importorskip`` only catches ``ImportError``. Some upstream
    dependencies of llama-index-core fail with other exceptions on older
    interpreters (for example ``X | None`` evaluated at runtime on 3.9), so
    the LlamaIndex test modules are skipped at collection time instead."""
    try:
        import llama_index.core  # noqa: F401
        from llama_index.core.callbacks.schema import CBEventType  # noqa: F401
    except Exception:  # any failure means "not available here"
        return False
    return True


def pytest_ignore_collect(collection_path: Path, config: pytest.Config) -> bool | None:
    if collection_path.name.startswith("test_llamaindex") and not _llamaindex_importable():
        return True
    return None
