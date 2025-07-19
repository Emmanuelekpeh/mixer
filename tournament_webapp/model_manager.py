#!/usr/bin/env python3
"""Model Management Micro-service
Runs separately from API; periodically discovers & validates AI models and
records metadata into the database. Publishes events on Redis channel
`model_events` whenever new models are registered.
"""
import os
import asyncio
import logging
from datetime import timedelta
from typing import Optional, Dict, Any

from fastapi import FastAPI
from pydantic import BaseModel

from redis import Redis

from tournament_webapp.backend.model_integration_system import quick_integrate_all
from storage import get_storage  # ensure storage root exists

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [model-manager] %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_CHANNEL = "model_events"
SCAN_INTERVAL_SECONDS = int(os.getenv("MODEL_SCAN_INTERVAL", "300"))  # 5 minutes by default

redis_client: Optional[Redis] = None

app = FastAPI(title="Model Management Service", version="0.1.0")


class HealthResponse(BaseModel):
    status: str
    models_discovered: int
    last_scan: Optional[str]


_state: Dict[str, Any] = {
    "last_scan": None,
    "last_count": 0,
}


@app.on_event("startup")
async def setup_service():
    global redis_client
    redis_client = Redis.from_url(REDIS_URL, decode_responses=True)
    logger.info("Connected to Redis @ %s", REDIS_URL)
    asyncio.create_task(scan_loop())


async def scan_loop():
    """Background loop that discovers & integrates models."""
    global _state
    while True:
        try:
            logger.info("🔍 Scanning for new models…")
            result = quick_integrate_all(auto_register=True, auto_integrate=False)
            count = len(result.get("discovered", []))
            _state["last_scan"] = result.get("timestamp")
            _state["last_count"] = count
            if count:
                logger.info("✅ Discovered %s models", count)
                # Publish event
                if redis_client:
                    redis_client.publish(REDIS_CHANNEL, {
                        "type": "model_added",
                        "count": count,
                    })
        except Exception as exc:
            logger.exception("Model scan failed: %s", exc)
        await asyncio.sleep(SCAN_INTERVAL_SECONDS)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(status="ok", models_discovered=_state["last_count"], last_scan=_state["last_scan"]) 