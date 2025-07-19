import os
import json
import uuid
from typing import Dict, Optional, Any

import redis

# ---------------------------------------------------------------------------
# Redis connection handling
# ---------------------------------------------------------------------------

REDIS_URL: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Lazy connection (so that importing module does not fail if Redis is absent)
_redis_client: Optional[redis.Redis] = None


def get_redis() -> redis.Redis:
    """Return a cached Redis client instance."""
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.from_url(REDIS_URL, decode_responses=True)  # type: ignore
    return _redis_client

# ---------------------------------------------------------------------------
# Job schema helpers
# ---------------------------------------------------------------------------

MIX_JOB_QUEUE = "mix_jobs"  # list key
MIX_JOB_STATUS_KEY_PREFIX = "mix_job_status:"  # per-job hash prefix


class MixJobStatus:
    PENDING = "pending"
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


# Job payload field names
JOB_AUDIO_PATH = "audio_path"
JOB_MODEL_ID = "model_id"
JOB_OUTPUT_PATH = "output_path"
JOB_TOURNAMENT_ID = "tournament_id"
JOB_USER_ID = "user_id"
JOB_JOB_ID = "job_id"


# ---------------------------------------------------------------------------
# Producer helpers (API gateway)
# ---------------------------------------------------------------------------

def enqueue_mix_job(
    audio_path: str,
    model_id: str,
    output_path: str,
    tournament_id: str,
    user_id: str,
    extra: Optional[Dict[str, Any]] = None,
) -> str:
    """Push a mix job into Redis queue and return its job ID."""
    job_id: str = str(uuid.uuid4())
    payload: Dict[str, Any] = {
        JOB_JOB_ID: job_id,
        JOB_AUDIO_PATH: audio_path,
        JOB_MODEL_ID: model_id,
        JOB_OUTPUT_PATH: output_path,
        JOB_TOURNAMENT_ID: tournament_id,
        JOB_USER_ID: user_id,
    }
    if extra:
        payload.update(extra)

    r = get_redis()
    r.rpush(MIX_JOB_QUEUE, json.dumps(payload))

    # Create initial status hash
    status_key = f"{MIX_JOB_STATUS_KEY_PREFIX}{job_id}"
    r.hset(status_key, mapping={"status": MixJobStatus.PENDING})
    return job_id


# ---------------------------------------------------------------------------
# Consumer helpers (worker service)
# ---------------------------------------------------------------------------

def fetch_next_mix_job(block: bool = True, timeout: int = 5) -> Optional[Dict[str, Any]]:
    """Pop next job from queue, blocking or non-blocking."""
    r = get_redis()
    if block:
        result = r.blpop(MIX_JOB_QUEUE, timeout=timeout)
        if result is None:
            return None
        _, payload = result
    else:
        payload = r.lpop(MIX_JOB_QUEUE)
        if payload is None:
            return None
    try:
        return json.loads(payload)
    except json.JSONDecodeError:
        return None


def update_job_status(job_id: str, status: str, extra: Optional[Dict[str, Any]] = None) -> None:
    r = get_redis()
    status_key = f"{MIX_JOB_STATUS_KEY_PREFIX}{job_id}"
    mapping = {"status": status}
    if extra:
        mapping.update(extra)
    r.hset(status_key, mapping=mapping)


# ---------------------------------------------------------------------------
# Cancellation helpers
# ---------------------------------------------------------------------------


def cancel_job(job_id: str) -> bool:
    """Attempt to cancel a pending job. Returns True if cancelled."""
    r = get_redis()
    # Remove from queue if still pending
    q_items = r.lrange(MIX_JOB_QUEUE, 0, -1)
    removed = False
    for item in q_items:
        try:
            payload = json.loads(item)
            if payload.get(JOB_JOB_ID) == job_id:
                r.lrem(MIX_JOB_QUEUE, 1, item)
                removed = True
                break
        except json.JSONDecodeError:
            continue

    update_job_status(job_id, MixJobStatus.CANCELLED if removed else MixJobStatus.FAILED, {"reason": "cancelled"})
    return removed


def get_job_status(job_id: str) -> Dict[str, Any]:
    r = get_redis()
    status_key = f"{MIX_JOB_STATUS_KEY_PREFIX}{job_id}"
    status_mapping = r.hgetall(status_key)
    return status_mapping or {"status": "unknown"} 