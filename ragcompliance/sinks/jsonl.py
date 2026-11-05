"""Append-only JSON Lines file sink.

One canonical JSON object per line. Writers serialise through a
``threading.RLock`` inside the process and, on POSIX, an ``fcntl.flock``
on a sibling ``.lock`` file across processes, so several processes can
append to the same file without forking the chain.

Options (URL query string or constructor arguments):

``fsync``
    Call ``os.fsync`` after every write. Default off; turn on when the
    file is the system of record and the host may lose power.

Purging rewrites the file atomically (temp file + ``os.replace``) and
then appends the retention checkpoint.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from ..chain import canonical_json
from .base import AuditSink, select_purge_victims

try:  # pragma: no cover - platform dependent
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]


class JSONLSink(AuditSink):
    name = "jsonl"
    supports_purge = True

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        chain_id: str = "default",
        hmac_key: bytes | str | None = None,
        fsync: bool = False,
    ):
        super().__init__(chain_id=chain_id, hmac_key=hmac_key)
        self.path = Path(path)
        self.fsync = bool(fsync)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock_path = self.path.with_name(self.path.name + ".lock")
        self._tails: dict[tuple[str, str], tuple[int, str]] = {}
        self._scanned_size = -1
        self._scanned_ino: int | None = None

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update({"path": str(self.path), "fsync": self.fsync})
        return d

    # ------------------------------------------------------------------ #
    # Locking                                                            #
    # ------------------------------------------------------------------ #

    @contextmanager
    def _locked(self, workspace_id: str, chain_id: str) -> Iterator[None]:
        with self._lock:
            fh = open(self._lock_path, "a+")  # noqa: SIM115 - closed in finally
            try:
                if fcntl is not None:
                    fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
                self._refresh_tails()
                yield
            finally:
                if fcntl is not None:
                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                    except OSError:  # pragma: no cover - best effort
                        pass
                fh.close()

    def _refresh_tails(self) -> None:
        """Bring the in-memory tail cache up to date with the file, which
        another process may have appended to or rewritten."""
        try:
            st = self.path.stat()
        except FileNotFoundError:
            self.path.touch(exist_ok=True)
            st = self.path.stat()
        if st.st_ino != self._scanned_ino or st.st_size < self._scanned_size:
            self._tails = {}
            self._scanned_size = 0
            self._scanned_ino = st.st_ino
        if st.st_size == self._scanned_size:
            return
        with open(self.path, "rb") as fh:
            fh.seek(max(0, self._scanned_size))
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                self._note_tail(rec)
            self._scanned_size = fh.tell()

    def _note_tail(self, rec: dict[str, Any]) -> None:
        key = (str(rec.get("workspace_id", "")), str(rec.get("chain_id", "")))
        seq = rec.get("sequence")
        if seq is None:
            return
        current = self._tails.get(key)
        if current is None or int(seq) >= current[0]:
            self._tails[key] = (int(seq), str(rec.get("record_hash", "")))

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def _tail_locked(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        return self._tails.get((workspace_id, chain_id))

    def _write_locked(self, sealed: dict[str, Any]) -> None:
        line = canonical_json(sealed) + "\n"
        with open(self.path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()
            if self.fsync:
                os.fsync(fh.fileno())
        self._note_tail(sealed)
        self._scanned_size = self.path.stat().st_size

    def _iter_all(
        self, *, workspace_id: str | None, chain_id: str | None
    ) -> Iterator[dict[str, Any]]:
        records = self._read_all()
        records.sort(
            key=lambda r: (
                str(r.get("workspace_id", "")),
                str(r.get("chain_id", "")),
                int(r.get("sequence") or 0),
            )
        )
        for rec in records:
            if workspace_id is not None and rec.get("workspace_id") != workspace_id:
                continue
            if chain_id is not None and rec.get("chain_id") != chain_id:
                continue
            yield rec

    def _read_all(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        if not self.path.exists():
            return out
        with open(self.path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return out

    def _purge_locked(
        self, cutoff: datetime, workspace_id: str, chain_id: str
    ) -> tuple[int, int | None, str]:
        records = self._read_all()
        chain = [
            r
            for r in records
            if r.get("workspace_id") == workspace_id and r.get("chain_id") == chain_id
        ]
        chain.sort(key=lambda r: int(r.get("sequence") or 0))
        purged, _kept = select_purge_victims(chain, cutoff)
        if not purged:
            return 0, None, ""
        purged_ids = {r.get("id") for r in purged}
        survivors = [r for r in records if r.get("id") not in purged_ids]

        fd, tmp_name = tempfile.mkstemp(
            prefix=self.path.name + ".", suffix=".tmp", dir=str(self.path.parent)
        )
        with os.fdopen(fd, "w", encoding="utf-8") as tmp:
            for rec in survivors:
                tmp.write(canonical_json(rec) + "\n")
            tmp.flush()
            os.fsync(tmp.fileno())
        os.replace(tmp_name, self.path)

        self._tails = {}
        self._scanned_size = -1
        self._scanned_ino = None
        self._refresh_tails()
        last = purged[-1]
        return len(purged), int(last["sequence"]), str(last["record_hash"])
