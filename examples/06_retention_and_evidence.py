"""Retention purge with a chain checkpoint, evidence pack export and
independent verification. Offline.

    python examples/06_retention_and_evidence.py
"""

from __future__ import annotations

import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ragcompliance import Auditor, RAGComplianceConfig, RetentionPolicy, RetrievedChunk
from ragcompliance.evidence import export_evidence_pack, verify_evidence_pack


def main() -> int:
    tmp = Path(tempfile.mkdtemp())
    config = RAGComplianceConfig(sink_url=f"sqlite:///{tmp}/audit.db", workspace_id="demo",
                                 async_writes=False, retention_days=30)
    auditor = Auditor(config, session_id="demo", billing=None, alerter=None)
    for i in range(5):
        with auditor.trace(f"question {i}") as t:
            t.add_chunk(RetrievedChunk(f"context {i}", "doc.pdf", f"c{i}", 0.9))
            t.set_answer(f"answer {i}", model_name="fake-model")

    # Pretend 30 days passed: everything is now older than the retention cutoff.
    later = datetime.now(timezone.utc) + timedelta(days=31)
    purge = auditor.storage.purge_expired(RetentionPolicy(days=30), workspace_id="demo", now=later)
    print(f"purged {purge.purged} records; checkpoint at sequence {purge.checkpoint['sequence']} "
          f"anchoring hash {purge.last_purged_hash[:12]}")

    with auditor.trace("question after purge") as t:
        t.set_answer("still chained", model_name="fake-model")

    result = auditor.storage.verify(workspace_id="demo")
    chain = result.chains[0]
    print(f"verify: {'intact' if result.ok else 'BROKEN'}; seq {chain.first_sequence}..{chain.last_sequence}; "
          f"anchored by checkpoint: {chain.anchored_by_checkpoint}")

    out = tmp / "demo-evidence.zip"
    pack = export_evidence_pack(auditor.storage, workspace_id="demo",
                                start=datetime.now(timezone.utc) - timedelta(days=1),
                                end=datetime.now(timezone.utc) + timedelta(days=1),
                                out_path=out, config=config)
    print(f"evidence pack: {out} ({pack.manifest['record_count']} records, pack_hash {pack.pack_hash[:12]})")

    problems, verification = verify_evidence_pack(out)
    print("pack verification:", "ok" if not problems and verification.ok else ("problems", problems))
    auditor.close()
    return 0 if result.ok and not problems and verification.ok else 1


if __name__ == "__main__":
    sys.exit(main())
