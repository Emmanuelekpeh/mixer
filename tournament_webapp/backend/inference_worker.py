#!/usr/bin/env python3
"""
🎧 Inference Worker Service
==========================

This standalone script connects to Redis, consumes mix-jobs enqueued by the API
(`task_queue.enqueue_mix_job`), executes the audio-mixing operation via the
existing `ai_mixer_integration_fixed` helper, and updates job status.

It is designed to run as a Railway **worker** service.

Usage:
    python inference_worker.py

Environment variables:
    REDIS_URL        – Redis connection string (default redis://localhost:6379/0)
    WORKER_POLL_TIME – Seconds to block-wait on Redis BLPOP (default 5)
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, Any, Optional

from task_queue import (
    fetch_next_mix_job,
    update_job_status,
    MixJobStatus,
    JOB_AUDIO_PATH,
    JOB_MODEL_ID,
    JOB_OUTPUT_PATH,
    JOB_JOB_ID,
    get_job_status,
)

# Ensure backend path resolution when executed from root directory
CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent
import sys
sys.path.append(str(ROOT_DIR))

from ai_mixer_integration_fixed import get_tournament_ai_mixer  # noqa: E402

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [worker] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

POLL_TIME = int(os.getenv("WORKER_POLL_TIME", "5"))


def process_job(job: Dict[str, Any]) -> None:
    job_id = job[JOB_JOB_ID]
    audio_path = job[JOB_AUDIO_PATH]
    model_id = job[JOB_MODEL_ID]
    output_path = job[JOB_OUTPUT_PATH]

    update_job_status(job_id, MixJobStatus.STARTED)
    # Check if job was cancelled while waiting
    status_obj = get_job_status(job_id)
    if status_obj.get("status") == MixJobStatus.CANCELLED:
        logger.info("⏩ Job %s was cancelled, skipping", job_id)
        return
    ai_mixer = get_tournament_ai_mixer()
    try:
        ok: bool = ai_mixer.process_audio_with_model(audio_path, model_id, output_path)
        if ok:
            update_job_status(job_id, MixJobStatus.SUCCESS, {"output_path": output_path})
            logger.info(f"✅ Job {job_id} success – output {output_path}")
        else:
            update_job_status(job_id, MixJobStatus.FAILED, {"reason": "mixer returned False"})
            logger.warning(f"❌ Job {job_id} failed – mixer returned False")
    except Exception as exc:
        update_job_status(job_id, MixJobStatus.FAILED, {"reason": str(exc)})
        logger.exception(f"❌ Job {job_id} failed with exception")


def main() -> None:
    logger.info("🎧 Inference worker started – waiting for jobs...")
    while True:
        job: Optional[Dict[str, Any]] = fetch_next_mix_job(block=True, timeout=POLL_TIME)
        if job is None:
            # Timeout – loop again
            continue
        logger.info(f"➡️  Picked job {job[JOB_JOB_ID]} for model {job[JOB_MODEL_ID]}")
        process_job(job)


if __name__ == "__main__":
    main() 