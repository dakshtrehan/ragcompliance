"""Evidence pack exporter, verifier CLI and the unified ragcompliance CLI."""

from __future__ import annotations

import json
import zipfile
from datetime import datetime, timedelta, timezone

import pytest

from ragcompliance import Auditor, AuditStorage, RAGComplianceConfig, RetrievedChunk
from ragcompliance.cli import main as cli_main
from ragcompliance.evidence import (
    EVIDENCE_FORMAT,
    export_evidence_pack,
    read_evidence_pack,
    verify_evidence_pack,
)
from ragcompliance.verify import main as verify_main


def _populated(tmp_path, **cfg):
    config = RAGComplianceConfig(
        sink_url=f"sqlite:///{tmp_path}/audit.db", async_writes=False, workspace_id="acme",
        redact_pii=True, retention_days=90, **cfg,
    )
    storage = AuditStorage(config)
    auditor = Auditor(config, storage=storage, session_id="s", actor={"id": "svc"}, billing=None, alerter=None)
    for i in range(4):
        with auditor.trace(f"question {i} from bob@example.com") as t:
            t.add_chunk(RetrievedChunk(f"ctx {i}", "doc.pdf", f"c{i}", 0.8))
            t.set_answer(f"answer {i}", model_name="m")
    return config, storage, auditor


START = datetime.now(timezone.utc) - timedelta(days=1)
END = datetime.now(timezone.utc) + timedelta(days=1)


class TestEvidencePack:
    def test_pack_contents_and_manifest(self, tmp_path):
        config, storage, auditor = _populated(tmp_path)
        out = tmp_path / "packs" / "acme.zip"
        pack = export_evidence_pack(storage, workspace_id="acme", start=START, end=END, out_path=out,
                                    config=config, redactor=auditor.redactor, seed=1)
        assert out.exists() and pack.pack_hash and pack.verification.ok
        with zipfile.ZipFile(out) as zf:
            names = set(zf.namelist())
            assert names == {"manifest.json", "records.jsonl", "verification.json", "report.md", "controls.json",
                             "controls.md", "retention.json", "redaction.json", "README.txt"}
            manifest = json.loads(zf.read("manifest.json"))
            records = [json.loads(line) for line in zf.read("records.jsonl").decode().splitlines()]
            report = zf.read("report.md").decode()
            redaction = json.loads(zf.read("redaction.json"))
            retention = json.loads(zf.read("retention.json"))
        assert manifest["format"] == EVIDENCE_FORMAT and manifest["record_count"] == 4
        assert manifest["chain_ok"] is True and manifest["sink"]["type"] == "sqlite"
        assert manifest["retention"] == {"days": 90, "enabled": True} and manifest["dropped_count"] == 0
        assert set(manifest["files"]) == names - {"manifest.json"}
        assert all("bob@example.com" not in r["query"] for r in records)
        assert "# RAG Audit Evidence: acme" in report and "—" not in report
        assert redaction["patterns"][0] == "anthropic_key" and retention["policy"]["days"] == 90
        assert manifest["redaction"]["patterns"] == redaction["patterns"]

    def test_verify_pack_detects_tampering(self, tmp_path):
        config, storage, _ = _populated(tmp_path)
        out = tmp_path / "p.zip"
        export_evidence_pack(storage, workspace_id="acme", start=START, end=END, out_path=out, config=config)
        problems, ver = verify_evidence_pack(out)
        assert problems == [] and ver.ok
        manifest, files = read_evidence_pack(out)
        files["records.jsonl"] = files["records.jsonl"].replace(b"answer 1", b"answer X")
        tampered = tmp_path / "t.zip"
        with zipfile.ZipFile(tampered, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
            for name, data in files.items():
                if name != "README.txt":
                    zf.writestr(name, data)
            zf.writestr("extra.txt", b"surprise")
        problems, ver = verify_evidence_pack(tampered)
        assert "hash mismatch: records.jsonl" in problems
        assert "missing file: README.txt" in problems
        assert "unexpected file not in manifest: extra.txt" in problems
        assert not ver.ok and {p.code for p in ver.problems} == {"content_signature_mismatch", "record_hash_mismatch"}
        manifest["pack_hash"] = "0" * 64
        manifest["format"] = "other"
        with zipfile.ZipFile(tampered, "w") as zf:
            zf.writestr("manifest.json", json.dumps(manifest))
        problems, _ = verify_evidence_pack(tampered)
        assert "pack_hash does not match the file table" in problems
        assert "manifest.format is not a ragcompliance evidence pack" in problems

    def test_records_supplied_directly_and_config_fallbacks(self, tmp_path):
        config, storage, _ = _populated(tmp_path)
        recs = list(storage.iter_records(workspace_id="acme"))
        out = tmp_path / "direct.zip"
        pack = export_evidence_pack(None, workspace_id="acme", start=START, end=END, out_path=out,
                                    config=RAGComplianceConfig(redact_pii=True, redaction_patterns=["email"]),
                                    records=recs)
        assert pack.manifest["record_count"] == 4 and pack.manifest["sink"] is None
        assert pack.manifest["redaction"]["patterns"] == ["email"]
        # a bare sink works too (describe() without the storage wrapper shape)
        pack2 = export_evidence_pack(storage.sink, workspace_id="acme", start=START, end=END, out_path=tmp_path / "s.zip")
        assert pack2.manifest["sink"]["type"] == "sqlite"

    def test_invalid_period_and_storage(self, tmp_path):
        with pytest.raises(ValueError):
            export_evidence_pack(None, workspace_id="a", start=END, end=START, out_path=tmp_path / "x.zip", records=[])
        with pytest.raises(TypeError):
            export_evidence_pack(object(), workspace_id="a", start=START, end=END, out_path=tmp_path / "x.zip")

    def test_pack_includes_retention_checkpoints(self, tmp_path):
        from ragcompliance.retention import RetentionPolicy

        config, storage, auditor = _populated(tmp_path)
        storage.purge_expired(RetentionPolicy(days=0), workspace_id="acme", now=datetime.now(timezone.utc) + timedelta(seconds=5))
        with auditor.trace("after") as t:
            t.set_answer("x")
        pack = export_evidence_pack(storage, workspace_id="acme", start=START, end=END, out_path=tmp_path / "r.zip", config=config)
        assert pack.manifest["retention_checkpoint_count"] == 1 and pack.manifest["invocation_count"] == 1
        with zipfile.ZipFile(pack.path) as zf:
            retention = json.loads(zf.read("retention.json"))
        assert retention["checkpoints"][0]["purged"] == 4


class TestVerifyCLI:
    def test_sink_ok_and_json(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", raising=False)
        _populated(tmp_path)
        url = f"sqlite:///{tmp_path}/audit.db"
        assert verify_main(["--sink", url, "--workspace", "acme"]) == 0
        out = capsys.readouterr().out
        assert "RESULT: INTACT" in out and "acme/default" in out
        assert verify_main(["--sink", url, "--json"]) == 0
        payload = json.loads(capsys.readouterr().out)
        assert payload["ok"] is True and payload["records_checked"] == 4

    def test_env_sink_and_hmac_env(self, tmp_path, capsys, monkeypatch):
        monkeypatch.setenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", "k")
        _populated(tmp_path)  # written WITH key from config? no: config has no key -> hmac mismatch
        url = f"sqlite:///{tmp_path}/audit.db"
        monkeypatch.setenv("RAGCOMPLIANCE_SINK", url)
        assert verify_main([]) == 1
        assert "hmac_mismatch" in capsys.readouterr().out
        monkeypatch.setenv("OTHER_KEY", "")
        assert verify_main(["--hmac-key-env", "OTHER_KEY"]) == 0

    def test_file_mode_and_filters(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", raising=False)
        config, storage, _ = _populated(tmp_path)
        path = tmp_path / "records.jsonl"
        from ragcompliance.chain import canonical_json

        recs = list(storage.iter_records())
        path.write_text("\n".join(canonical_json(r) for r in recs) + "\n\n")
        assert verify_main(["--file", str(path), "--workspace", "acme", "--chain-id", "default"]) == 0
        assert verify_main(["--file", str(path), "--workspace", "nobody"]) == 0  # nothing to check
        recs[2]["llm_answer"] = "tampered"
        path.write_text("\n".join(canonical_json(r) for r in recs))
        assert verify_main(["--file", str(path)]) == 1
        assert "TAMPER EVIDENCE" in capsys.readouterr().out
        assert verify_main(["--file", str(tmp_path / "missing.jsonl")]) == 2
        bad = tmp_path / "bad.jsonl"
        bad.write_text("{not json")
        assert verify_main(["--file", str(bad)]) == 2

    def test_require_chain_and_trust_anchor(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", raising=False)
        from ragcompliance.chain import canonical_json

        config, storage, _ = _populated(tmp_path)
        recs = list(storage.iter_records())
        legacy = dict(recs[0])
        for k in ("record_hash", "sequence", "prev_hash", "record_hmac"):
            legacy[k] = "" if k != "sequence" else None
        path = tmp_path / "legacy.jsonl"
        path.write_text(canonical_json(legacy) + "\n")
        assert verify_main(["--file", str(path)]) == 0
        assert verify_main(["--file", str(path), "--require-chain"]) == 1
        tail = tmp_path / "tail.jsonl"
        tail.write_text("\n".join(canonical_json(r) for r in recs[2:]))
        assert verify_main(["--file", str(tail)]) == 1
        assert verify_main(["--file", str(tail), "--trust-anchor", recs[1]["record_hash"]]) == 0

    def test_evidence_mode(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", raising=False)
        config, storage, _ = _populated(tmp_path)
        out = tmp_path / "p.zip"
        export_evidence_pack(storage, workspace_id="acme", start=START, end=END, out_path=out, config=config)
        assert verify_main(["--evidence", str(out), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["pack_problems"] == []
        assert verify_main(["--evidence", str(tmp_path / "nope.zip")]) == 2

    def test_no_sink_configured(self, monkeypatch, capsys):
        for var in ("RAGCOMPLIANCE_SINK", "RAGCOMPLIANCE_SUPABASE_URL", "RAGCOMPLIANCE_SUPABASE_KEY", "RAGCOMPLIANCE_DEV_MODE"):
            monkeypatch.delenv(var, raising=False)
        assert verify_main([]) == 2
        assert "no sink" in capsys.readouterr().err


class TestUnifiedCLI:
    def test_dispatch_and_help(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", raising=False)
        for var in ("RAGCOMPLIANCE_SINK", "RAGCOMPLIANCE_SUPABASE_URL", "RAGCOMPLIANCE_SUPABASE_KEY", "RAGCOMPLIANCE_DEV_MODE"):
            monkeypatch.delenv(var, raising=False)
        _populated(tmp_path)
        url = f"sqlite:///{tmp_path}/audit.db"
        assert cli_main([]) == 2
        assert cli_main(["verify", "--sink", url]) == 0
        capsys.readouterr()
        assert cli_main(["show", "--sink", url, "--json"]) == 0
        shown = json.loads(capsys.readouterr().out)
        assert shown[0]["records"] == 4 and shown[0]["last"] == 4
        assert cli_main(["show", "--sink", url]) == 0
        assert "acme/default: 4 records" in capsys.readouterr().out
        assert cli_main(["soc2", "--sink", url, "--workspace", "acme", "--start", "2020-01-01", "--end", "2030-01-01"]) == 0
        assert "Records in scope: **4**" in capsys.readouterr().out
        assert cli_main(["selftest", "--dev-mode", "--json"]) == 0
        with pytest.raises(SystemExit):
            cli_main(["--version"])

    def test_evidence_and_purge(self, tmp_path, capsys, monkeypatch):
        monkeypatch.delenv("RAGCOMPLIANCE_CHAIN_HMAC_KEY", raising=False)
        _populated(tmp_path)
        url = f"sqlite:///{tmp_path}/audit.db"
        out = tmp_path / "e.zip"
        assert cli_main(["evidence", "--sink", url, "--workspace", "acme", "--start", "2020-01-01", "--end", "2030-01-01",
                         "--out", str(out), "--json"]) == 0
        assert json.loads(capsys.readouterr().out)["records"] == 4
        assert cli_main(["evidence", "--sink", url, "--workspace", "acme", "--start", "2030-01-01", "--end", "2020-01-01",
                         "--out", str(out)]) == 2
        assert cli_main(["purge", "--sink", url, "--workspace", "acme"]) == 2  # no days
        assert cli_main(["purge", "--sink", url, "--workspace", "acme", "--days", "0", "--dry-run"]) == 0
        assert "dry run" in capsys.readouterr().out
        assert cli_main(["purge", "--sink", url, "--workspace", "acme", "--days", "3650"]) == 0
        assert "purged 0" in capsys.readouterr().out
        assert cli_main(["purge", "--sink", url, "--workspace", "acme", "--days", "0", "--json"]) == 0
        purged = json.loads(capsys.readouterr().out)
        assert purged["purged"] == 4
        assert cli_main(["show", "--sink", url]) == 0

    def test_purge_not_supported_and_no_sink(self, tmp_path, capsys, monkeypatch):
        for var in ("RAGCOMPLIANCE_SINK", "RAGCOMPLIANCE_SUPABASE_URL", "RAGCOMPLIANCE_SUPABASE_KEY", "RAGCOMPLIANCE_DEV_MODE"):
            monkeypatch.delenv(var, raising=False)
        assert cli_main(["show"]) == 2
        assert "no sink" in capsys.readouterr().err
        monkeypatch.setenv("RAGCOMPLIANCE_RETENTION_DAYS", "1")
        # memory sink with no records: nothing to do
        assert cli_main(["purge", "--sink", "memory://", "--workspace", "w"]) == 0
        assert cli_main(["show", "--sink", "memory://"]) == 0
        assert "no records" in capsys.readouterr().out

    def test_verify_entry_point(self, monkeypatch, capsys):
        from ragcompliance.cli import verify_entry

        monkeypatch.setattr("sys.argv", ["ragcompliance-verify", "--file", "/nonexistent.jsonl"])
        assert verify_entry() == 2
