"""Selftest CLI, control mapping, package surface and documentation hygiene."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import ragcompliance
from ragcompliance import selftest
from ragcompliance.controls import CONTROL_STATUS_LABELS, CONTROLS, controls_json, controls_markdown

ROOT = Path(__file__).resolve().parents[1]


class TestSelftest:
    def test_dev_mode_passes(self, monkeypatch, capsys):
        monkeypatch.delenv("RAGCOMPLIANCE_SINK", raising=False)
        assert selftest.main(["--dev-mode"]) == 0
        out = capsys.readouterr().out
        assert "All blocking checks passed" in out and "chain.roundtrip" in out

    def test_json_and_sink_detection(self, monkeypatch, capsys):
        monkeypatch.delenv("RAGCOMPLIANCE_DEV_MODE", raising=False)
        monkeypatch.delenv("RAGCOMPLIANCE_SUPABASE_URL", raising=False)
        monkeypatch.delenv("RAGCOMPLIANCE_SINK", raising=False)
        assert selftest.main(["--json"]) == 1
        results = json.loads(capsys.readouterr().out)
        by_name = {r["name"]: r for r in results}
        assert by_name["sink.configured"]["ok"] is False
        monkeypatch.setenv("RAGCOMPLIANCE_SINK", "memory://")
        assert selftest.main([]) == 0
        monkeypatch.delenv("RAGCOMPLIANCE_SINK")
        monkeypatch.setenv("RAGCOMPLIANCE_SUPABASE_URL", "https://x")
        monkeypatch.setenv("RAGCOMPLIANCE_SUPABASE_KEY", "k")
        assert selftest._check_sink_configured(False).ok

    def test_chain_roundtrip_with_hmac(self, monkeypatch):
        monkeypatch.setenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", "secret")
        res = selftest._check_chain_roundtrip()
        assert res.ok and res.data["head_hash"]

    def test_broken_extra_is_reported_not_fatal(self, monkeypatch):
        import importlib

        real = importlib.import_module

        def fake(name, *a, **k):
            if name == "openai":
                raise TypeError("upstream package broken on this interpreter")
            return real(name, *a, **k)

        monkeypatch.setattr(selftest.importlib, "import_module", fake)
        results = {r.name: r for r in selftest._check_optional_extras()}
        assert results["extras.openai"].ok is False
        assert "failed to import (TypeError)" in results["extras.openai"].detail

    def test_config_check_handles_errors(self, monkeypatch):
        import ragcompliance.config as cfg_mod

        monkeypatch.setattr(cfg_mod.RAGComplianceConfig, "from_env", classmethod(lambda cls: (_ for _ in ()).throw(RuntimeError("bad"))))
        res = selftest._check_config()
        assert not res.ok and "raised" in res.detail

    def test_tty_formatting_branch(self, monkeypatch):
        monkeypatch.setattr("sys.stdout.isatty", lambda: True)
        text = selftest._format_human([selftest.CheckResult("x", False, selftest.SEVERITY_RED, "d"),
                                       selftest.CheckResult("y", False, selftest.SEVERITY_YELLOW, "d")])
        assert "\033[31m" in text and "blocking check(s) failed" in text

    def test_roundtrip_detects_broken_verifier(self, monkeypatch):
        import ragcompliance.selftest as st

        monkeypatch.setattr(st, "_check_chain_roundtrip", lambda: st.CheckResult("chain.roundtrip", False, "red", "x"))
        assert any(not r.ok for r in st.run_selftest(dev_mode=True))


class TestControls:
    def test_every_control_has_fields_and_known_status(self):
        for c in CONTROLS:
            assert c["id"] and c["title"] and c["feature"] and c["evidence"]
            assert c["status"] in CONTROL_STATUS_LABELS
        ids = [c["id"] for c in CONTROLS]
        assert len(ids) == len(set(ids))

    def test_markdown_and_json(self):
        md = controls_markdown()
        assert md.startswith("| Framework |") and "CC8.1" in md and "—" not in md
        js = controls_json()
        assert js[0]["status_label"] in CONTROL_STATUS_LABELS.values()

    def test_docs_control_mapping_covers_every_control(self):
        doc = (ROOT / "docs" / "control-mapping.md").read_text()
        for c in CONTROLS:
            assert c["id"] in doc, c["id"]
        assert "not a certification" in doc.lower() or "not itself a" in doc.lower()


class TestDocsHygiene:
    DOC_FILES = ["README.md", "CHANGELOG.md", "AUDIT.md", "DECISIONS.md", "RELEASE.md", "SECURITY.md", "CONTRIBUTING.md"]

    def test_no_em_dashes_in_markdown(self):
        offenders = []
        for path in [ROOT / f for f in self.DOC_FILES] + list((ROOT / "docs").glob("*.md")):
            if path.exists() and re.search("[—–]", path.read_text(encoding="utf-8")):
                offenders.append(str(path))
        assert offenders == []

    def test_no_em_dashes_in_sources(self):
        offenders = [
            str(p) for p in (ROOT / "ragcompliance").rglob("*.py")
            if re.search("[—–]", p.read_text(encoding="utf-8"))
        ]
        assert offenders == []

    def test_mkdocs_nav_files_exist(self):
        import yaml  # provided by mkdocs; skip if unavailable

        cfg = yaml.safe_load((ROOT / "mkdocs.yml").read_text())
        def walk(nav):
            for item in nav:
                for _, v in item.items():
                    if isinstance(v, list):
                        yield from walk(v)
                    else:
                        yield v
        for rel in walk(cfg["nav"]):
            assert (ROOT / "docs" / rel).exists(), rel

    def test_version_consistency(self):
        assert ragcompliance.__version__ == "1.0.0rc1"
        changelog = (ROOT / "CHANGELOG.md").read_text()
        assert "## [1.0.0rc1]" in changelog
        readme = (ROOT / "README.md").read_text()
        assert "1.0.0rc1" in readme or "pip install" in readme


class TestPackageSurface:
    def test_lazy_exports(self):
        assert ragcompliance.RAGComplianceHandler.__name__ == "RAGComplianceHandler"
        assert ragcompliance.BillingManager.__name__ == "BillingManager"
        assert isinstance(ragcompliance.PLANS, dict)
        with pytest.raises(AttributeError):
            ragcompliance.does_not_exist  # noqa: B018
        for name in ragcompliance.__all__:
            assert getattr(ragcompliance, name) is not None

    def test_core_import_does_not_need_langchain(self):
        import subprocess
        import sys

        code = (
            "import sys; sys.modules['langchain_core']=None; sys.modules['stripe']=None; "
            "import ragcompliance; from ragcompliance import Auditor; print('ok')"
        )
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=str(ROOT))
        assert out.returncode == 0 and out.stdout.strip() == "ok", out.stderr
