"""S3 compatible object store sink (optional extra ``s3``, uses ``boto3``).

Layout::

    {prefix}/{workspace_id}/{chain_id}/{sequence:012d}-{record_id}.json
    {prefix}/{workspace_id}/{chain_id}/_head.json   -> {"sequence", "record_hash"}

One object per record keeps every record independently addressable and
lets S3 Object Lock make each one WORM. The head pointer makes the tail
lookup one GET; when it is missing (first write, or a deleted pointer)
the sink falls back to listing the chain prefix.

Object Lock: pass ``object_lock_mode`` (``GOVERNANCE`` or ``COMPLIANCE``)
and ``object_lock_days`` and every record object is written with a
retain-until date. The bucket must have Object Lock enabled. With Object
Lock the purge path raises ``PurgeNotSupported`` because deletes are
refused by the store, which is the point.

Concurrency: writers inside one process are serialised. Several
processes appending to the same chain are not (there is no lock in S3),
so give each writer its own ``chain_id``; see DECISIONS.md D02.

Works with MinIO, Ceph RGW, Cloudflare R2 and any other S3 compatible
endpoint via ``endpoint_url``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import datetime, timedelta, timezone
from typing import Any

from ..chain import canonical_json
from ..retention import PurgeNotSupported
from .base import AuditSink, select_purge_victims


class S3Sink(AuditSink):
    name = "s3"
    supports_purge = True

    def __init__(
        self,
        bucket: str,
        prefix: str = "ragcompliance",
        *,
        chain_id: str = "default",
        hmac_key: bytes | str | None = None,
        client: Any | None = None,
        endpoint_url: str | None = None,
        region_name: str | None = None,
        object_lock_mode: str | None = None,
        object_lock_days: int | None = None,
    ):
        super().__init__(chain_id=chain_id, hmac_key=hmac_key)
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.object_lock_mode = (object_lock_mode or "").upper() or None
        self.object_lock_days = int(object_lock_days) if object_lock_days else None
        if client is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover - depends on extras
                raise ImportError(
                    "boto3 is not installed. Install with: pip install 'ragcompliance[s3]'"
                ) from exc
            kwargs: dict[str, Any] = {}
            if endpoint_url:
                kwargs["endpoint_url"] = endpoint_url
            if region_name:
                kwargs["region_name"] = region_name
            client = boto3.client("s3", **kwargs)
        self._client = client
        self._tails: dict[tuple[str, str], tuple[int, str]] = {}

    def describe(self) -> dict[str, Any]:
        d = super().describe()
        d.update(
            {
                "bucket": self.bucket,
                "prefix": self.prefix,
                "object_lock_mode": self.object_lock_mode,
                "object_lock_days": self.object_lock_days,
            }
        )
        return d

    # ------------------------------------------------------------------ #
    # Keys                                                               #
    # ------------------------------------------------------------------ #

    def _chain_prefix(self, workspace_id: str, chain_id: str) -> str:
        base = f"{self.prefix}/" if self.prefix else ""
        return f"{base}{workspace_id}/{chain_id}/"

    def _record_key(self, sealed: dict[str, Any]) -> str:
        return (
            self._chain_prefix(str(sealed.get("workspace_id", "")), str(sealed["chain_id"]))
            + f"{int(sealed['sequence']):012d}-{sealed['id']}.json"
        )

    def _head_key(self, workspace_id: str, chain_id: str) -> str:
        return self._chain_prefix(workspace_id, chain_id) + "_head.json"

    # ------------------------------------------------------------------ #
    # Hooks                                                              #
    # ------------------------------------------------------------------ #

    def _tail_locked(self, workspace_id: str, chain_id: str) -> tuple[int, str] | None:
        key = (workspace_id, chain_id)
        if key in self._tails:
            return self._tails[key]
        head = self._get_json(self._head_key(workspace_id, chain_id))
        if head is not None and head.get("sequence") is not None:
            cached: tuple[int, str] = (int(head["sequence"]), str(head.get("record_hash", "")))
            self._tails[key] = cached
            return cached
        # Fall back to listing. Keys are zero padded so the last key in
        # lexical order is the newest record.
        last_key = None
        for obj_key in self._list_keys(self._chain_prefix(workspace_id, chain_id)):
            if not obj_key.endswith("_head.json"):
                last_key = obj_key
        if last_key is None:
            return None
        rec = self._get_json(last_key)
        if rec is None:
            return None
        found: tuple[int, str] = (int(rec["sequence"]), str(rec["record_hash"]))
        self._tails[key] = found
        return found

    def _write_locked(self, sealed: dict[str, Any]) -> None:
        body = canonical_json(sealed).encode("utf-8")
        put_kwargs: dict[str, Any] = {
            "Bucket": self.bucket,
            "Key": self._record_key(sealed),
            "Body": body,
            "ContentType": "application/json",
        }
        if self.object_lock_mode:
            put_kwargs["ObjectLockMode"] = self.object_lock_mode
            put_kwargs["ObjectLockRetainUntilDate"] = datetime.now(timezone.utc) + timedelta(
                days=self.object_lock_days or 1
            )
        self._client.put_object(**put_kwargs)
        workspace_id = str(sealed.get("workspace_id", ""))
        chain_id = str(sealed["chain_id"])
        seq = int(sealed["sequence"])
        record_hash = str(sealed["record_hash"])
        self._client.put_object(
            Bucket=self.bucket,
            Key=self._head_key(workspace_id, chain_id),
            Body=json.dumps({"sequence": seq, "record_hash": record_hash}).encode("utf-8"),
            ContentType="application/json",
        )
        self._tails[(workspace_id, chain_id)] = (seq, record_hash)

    def _iter_all(
        self, *, workspace_id: str | None, chain_id: str | None
    ) -> Iterator[dict[str, Any]]:
        base = f"{self.prefix}/" if self.prefix else ""
        if workspace_id is not None:
            base = f"{base}{workspace_id}/"
            if chain_id is not None:
                base = f"{base}{chain_id}/"
        for key in self._list_keys(base):
            if key.endswith("_head.json"):
                continue
            rec = self._get_json(key)
            if rec is None:
                continue
            if chain_id is not None and rec.get("chain_id") != chain_id:
                continue
            if workspace_id is not None and rec.get("workspace_id") != workspace_id:
                continue
            yield rec

    def _purge_locked(
        self, cutoff: datetime, workspace_id: str, chain_id: str
    ) -> tuple[int, int | None, str]:
        if self.object_lock_mode:
            raise PurgeNotSupported(
                "this S3 sink writes with Object Lock; records cannot be deleted "
                "before their retain-until date"
            )
        chain = list(self._iter_all(workspace_id=workspace_id, chain_id=chain_id))
        chain.sort(key=lambda r: int(r.get("sequence") or 0))
        purged, _kept = select_purge_victims(chain, cutoff)
        if not purged:
            return 0, None, ""
        for rec in purged:
            self._client.delete_object(Bucket=self.bucket, Key=self._record_key(rec))
        last = purged[-1]
        return len(purged), int(last["sequence"]), str(last["record_hash"])

    # ------------------------------------------------------------------ #
    # S3 helpers                                                         #
    # ------------------------------------------------------------------ #

    def _get_json(self, key: str) -> dict[str, Any] | None:
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=key)
        except Exception as exc:
            code = getattr(getattr(exc, "response", None), "get", lambda *_: None)("Error")
            if code and code.get("Code") in ("NoSuchKey", "404", "NotFound"):
                return None
            if exc.__class__.__name__ in ("NoSuchKey", "ClientError"):
                return None
            raise
        body = resp["Body"].read()
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

    def _list_keys(self, prefix: str) -> Iterator[str]:
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix):
            for obj in page.get("Contents", []) or []:
                yield obj["Key"]
