"""S3Sink tests against moto's in-memory S3."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")

from ragcompliance.chain import content_signature, verify_records  # noqa: E402
from ragcompliance.models import AuditRecord  # noqa: E402
from ragcompliance.retention import PurgeNotSupported  # noqa: E402
from ragcompliance.sinks import make_sink  # noqa: E402
from ragcompliance.sinks.s3 import S3Sink  # noqa: E402


def _rec(i: int, ts: datetime | None = None, ws: str = "ws") -> dict:
    r = AuditRecord(session_id="s", workspace_id=ws, query=f"q{i}", llm_answer=f"a{i}")
    r.chain_signature = content_signature(r.query, r.retrieved_chunks, r.llm_answer)
    if ts:
        r.timestamp = ts
    return r.to_dict()


@pytest.fixture
def s3_client():
    with moto.mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="audit")
        yield client


class TestS3Sink:
    def test_append_layout_and_head(self, s3_client):
        sink = S3Sink("audit", "rc", client=s3_client)
        s1 = sink.append(_rec(1))
        s2 = sink.append(_rec(2))
        keys = sorted(o["Key"] for o in s3_client.list_objects_v2(Bucket="audit")["Contents"])
        assert keys[0] == f"rc/ws/default/000000000001-{s1['id']}.json"
        assert keys[1] == f"rc/ws/default/000000000002-{s2['id']}.json"
        assert keys[2] == "rc/ws/default/_head.json"
        head = json.loads(s3_client.get_object(Bucket="audit", Key="rc/ws/default/_head.json")["Body"].read())
        assert head == {"sequence": 2, "record_hash": s2["record_hash"]}
        assert verify_records(sink.iter_records()).ok

    def test_new_instance_reads_head(self, s3_client):
        a = S3Sink("audit", "rc", client=s3_client)
        a.append(_rec(1))
        b = S3Sink("audit", "rc", client=s3_client)
        assert b.append(_rec(2))["sequence"] == 2

    def test_missing_head_falls_back_to_listing(self, s3_client):
        a = S3Sink("audit", "rc", client=s3_client)
        a.append(_rec(1))
        a.append(_rec(2))
        s3_client.delete_object(Bucket="audit", Key="rc/ws/default/_head.json")
        b = S3Sink("audit", "rc", client=s3_client)
        assert b.append(_rec(3))["sequence"] == 3
        assert verify_records(b.iter_records()).ok

    def test_empty_chain_tail_is_none(self, s3_client):
        sink = S3Sink("audit", "rc", client=s3_client)
        assert sink.tail("ws") is None

    def test_iter_filters_and_get(self, s3_client):
        sink = S3Sink("audit", "", client=s3_client)
        sink.append(_rec(1, ws="a"))
        sink.append(_rec(2, ws="b"))
        assert [r["workspace_id"] for r in sink.iter_records()] == ["a", "b"]
        assert [r["workspace_id"] for r in sink.iter_records(workspace_id="b")] == ["b"]
        assert list(sink.iter_records(workspace_id="a", chain_id="other")) == []
        sealed = next(sink.iter_records(workspace_id="a"))
        assert sink.get(sealed["id"])["query"] == "q1"

    def test_purge_and_checkpoint(self, s3_client):
        sink = S3Sink("audit", "rc", client=s3_client)
        old = datetime.now(timezone.utc) - timedelta(days=40)
        sink.append(_rec(1, old))
        sink.append(_rec(2, old))
        sink.append(_rec(3))
        res = sink.purge_before(datetime.now(timezone.utc) - timedelta(days=30), workspace_id="ws")
        assert res.purged == 2
        left = list(sink.iter_records())
        assert [r["sequence"] for r in left] == [3, 4]
        assert verify_records(left).ok
        assert sink.purge_before(datetime.now(timezone.utc) - timedelta(days=30), workspace_id="ws").purged == 0

    def test_object_lock_refuses_purge_and_sets_retention(self, s3_client):
        s3_client.create_bucket(Bucket="worm", ObjectLockEnabledForBucket=True)
        sink = S3Sink("worm", "rc", client=s3_client, object_lock_mode="governance", object_lock_days=30)
        sealed = sink.append(_rec(1))
        key = f"rc/ws/default/000000000001-{sealed['id']}.json"
        head = s3_client.head_object(Bucket="worm", Key=key)
        assert head.get("ObjectLockMode") == "GOVERNANCE"
        with pytest.raises(PurgeNotSupported):
            sink.purge_before(datetime.now(timezone.utc), workspace_id="ws")
        assert sink.describe()["object_lock_mode"] == "GOVERNANCE"

    def test_corrupt_object_skipped(self, s3_client):
        sink = S3Sink("audit", "rc", client=s3_client)
        sink.append(_rec(1))
        s3_client.put_object(Bucket="audit", Key="rc/ws/default/000000000002-bad.json", Body=b"nope")
        assert len(list(sink.iter_records())) == 1

    def test_factory_options(self, s3_client):
        sink = make_sink(
            "s3://audit/prefix/deep?region_name=us-east-1&object_lock_mode=COMPLIANCE&object_lock_days=7&chain_id=w2",
            client=s3_client,
        )
        assert isinstance(sink, S3Sink)
        assert sink.bucket == "audit" and sink.prefix == "prefix/deep"
        assert sink.object_lock_mode == "COMPLIANCE" and sink.object_lock_days == 7
        assert sink.chain_id == "w2"

    def test_factory_builds_boto_client(self):
        with moto.mock_aws():
            sink = make_sink("s3://b/p?endpoint_url=http://localhost:9000&region_name=us-east-1")
            assert isinstance(sink, S3Sink)

    def test_unexpected_client_error_propagates(self, s3_client):
        class Boom(Exception):
            pass

        class BadClient:
            def get_object(self, **kw):
                raise Boom("network")

        sink = S3Sink("audit", "rc", client=BadClient())
        with pytest.raises(Boom):
            sink._get_json("k")
