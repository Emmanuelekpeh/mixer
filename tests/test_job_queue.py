import uuid

import fakeredis
import pytest

from tournament_webapp.backend import task_queue as tq


def _patch_redis(monkeypatch):
    fake_r = fakeredis.FakeRedis(decode_responses=True)
    monkeypatch.setattr(tq, "_redis_client", fake_r)
    return fake_r


def test_enqueue_and_status(monkeypatch):
    r = _patch_redis(monkeypatch)

    job_id = tq.enqueue_mix_job(
        audio_path="/tmp/input.wav",
        model_id="model123",
        output_path="/tmp/output.wav",
        tournament_id="tourn1",
        user_id="user1",
    )

    # Queue should contain one entry
    assert r.llen(tq.MIX_JOB_QUEUE) == 1

    status = tq.get_job_status(job_id)
    assert status["status"] == tq.MixJobStatus.PENDING

    # Simulate worker start
    tq.update_job_status(job_id, tq.MixJobStatus.STARTED)
    status = tq.get_job_status(job_id)
    assert status["status"] == tq.MixJobStatus.STARTED

    # Simulate success
    tq.update_job_status(job_id, tq.MixJobStatus.SUCCESS, {"output_path": "/tmp/output.wav"})
    status = tq.get_job_status(job_id)
    assert status["status"] == tq.MixJobStatus.SUCCESS
    assert status["output_path"] == "/tmp/output.wav"


def test_cancel_job(monkeypatch):
    r = _patch_redis(monkeypatch)

    job_id = tq.enqueue_mix_job(
        audio_path="/tmp/in2.wav",
        model_id="m2",
        output_path="/tmp/out2.wav",
        tournament_id="t2",
        user_id="u2",
    )

    cancelled = tq.cancel_job(job_id)
    assert cancelled is True
    # Queue should be empty now
    assert r.llen(tq.MIX_JOB_QUEUE) == 0
    status = tq.get_job_status(job_id)
    assert status["status"] == tq.MixJobStatus.CANCELLED 