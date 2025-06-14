#!/usr/bin/env python3
"""
🚀 Deployment Verification Test
============================

Verifies that a deployed instance of the AI Mixer API is working correctly
by running a comprehensive series of API tests.
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import asyncio

import httpx
import requests
from tenacity import retry, stop_after_attempt, wait_exponential

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("verify_deployment")

# Default API endpoint
DEFAULT_API_URL = "https://ai-mixer-tournament.onrender.com"

# Test sample file
SAMPLE_AUDIO_PATH = Path("data/test/sample_verification.wav")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Verify AI Mixer API deployment")
    parser.add_argument(
        "--api-url",
        type=str,
        default=os.environ.get("API_URL", DEFAULT_API_URL),
        help=f"API URL to test (default: {DEFAULT_API_URL})"
    )
    parser.add_argument(
        "--sample-file",
        type=str,
        default=str(SAMPLE_AUDIO_PATH),
        help=f"Sample audio file for testing (default: {SAMPLE_AUDIO_PATH})"
    )
    parser.add_argument(
        "--no-cleanup",
        action="store_true",
        help="Do not clean up created tournaments after testing"
    )
    
    return parser.parse_args()


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=1, max=10))
def check_api_health(api_url: str) -> bool:
    """
    Check if the API is healthy
    
    Args:
        api_url: API URL to check
        
    Returns:
        True if the API is healthy, False otherwise
    """
    try:
        logger.info(f"Checking API health at {api_url}/api/health")
        response = requests.get(f"{api_url}/api/health", timeout=10)
        
        if response.status_code == 200:
            logger.info("API health check passed")
            return True
        else:
            logger.error(f"API health check failed: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"API health check failed: {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def create_test_tournament(api_url: str, sample_file_path: str) -> Tuple[bool, str]:
    """
    Create a test tournament
    
    Args:
        api_url: API URL
        sample_file_path: Path to sample audio file
        
    Returns:
        Tuple of (success, tournament_id)
    """
    try:
        logger.info(f"Creating test tournament with file {sample_file_path}")
        
        # Prepare the form data
        form_data = {
            "user_id": "test_verification_user",
            "username": "Deployment Verifier",
            "max_rounds": 2,
            "audio_features": json.dumps({
                "verification": True,
                "timestamp": time.time()
            })
        }
        
        # Prepare the file
        with open(sample_file_path, "rb") as f:
            files = {"audio_file": (os.path.basename(sample_file_path), f, "audio/wav")}
            
            # Send the request
            response = requests.post(
                f"{api_url}/api/tournaments/create",
                data=form_data,
                files=files,
                timeout=30
            )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success", False) and "tournament" in result:
                tournament_id = result["tournament"]["id"]
                logger.info(f"Created test tournament with ID {tournament_id}")
                return True, tournament_id
            else:
                logger.error(f"Failed to create tournament: {result}")
                return False, ""
        else:
            logger.error(f"Failed to create tournament: {response.status_code} - {response.text}")
            return False, ""
    
    except Exception as e:
        logger.error(f"Failed to create tournament: {str(e)}")
        raise


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
def execute_test_battle(api_url: str, tournament_id: str) -> Tuple[bool, str]:
    """
    Execute a test battle in the tournament
    
    Args:
        api_url: API URL
        tournament_id: Tournament ID
        
    Returns:
        Tuple of (success, task_id)
    """
    try:
        logger.info(f"Executing battle for tournament {tournament_id}")
        
        # Send the request
        response = requests.post(
            f"{api_url}/api/tournaments/{tournament_id}/battle",
            timeout=30
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            if result.get("success", False) and "task_id" in result:
                task_id = result["task_id"]
                logger.info(f"Started battle with task ID {task_id}")
                return True, task_id
            else:
                logger.error(f"Failed to execute battle: {result}")
                return False, ""
        else:
            logger.error(f"Failed to execute battle: {response.status_code} - {response.text}")
            return False, ""
    
    except Exception as e:
        logger.error(f"Failed to execute battle: {str(e)}")
        raise


async def wait_for_task_completion(api_url: str, task_id: str, timeout: int = 120) -> bool:
    """
    Wait for a task to complete
    
    Args:
        api_url: API URL
        task_id: Task ID
        timeout: Maximum wait time in seconds
        
    Returns:
        True if the task completed successfully, False otherwise
    """
    logger.info(f"Waiting for task {task_id} to complete (timeout: {timeout}s)")
    
    start_time = time.time()
    success = False
    
    async with httpx.AsyncClient() as client:
        while time.time() - start_time < timeout:
            try:
                response = await client.get(f"{api_url}/api/tasks/{task_id}", timeout=10)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    status = result.get("status", "")
                    progress = result.get("progress", 0)
                    message = result.get("message", "")
                    
                    logger.info(f"Task status: {status} - Progress: {progress:.0%} - {message}")
                    
                    if status == "completed":
                        logger.info(f"Task {task_id} completed successfully")
                        success = True
                        break
                    elif status == "failed":
                        logger.error(f"Task {task_id} failed: {message}")
                        break
                    
                    # Wait before checking again
                    await asyncio.sleep(5)
                else:
                    logger.error(f"Failed to get task status: {response.status_code} - {response.text}")
                    await asyncio.sleep(5)
            
            except Exception as e:
                logger.error(f"Error checking task status: {str(e)}")
                await asyncio.sleep(5)
    
    elapsed_time = time.time() - start_time
    logger.info(f"Task monitoring finished after {elapsed_time:.1f} seconds")
    
    return success


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=5))
def check_tournament_status(api_url: str, tournament_id: str) -> bool:
    """
    Check the status of a tournament
    
    Args:
        api_url: API URL
        tournament_id: Tournament ID
        
    Returns:
        True if the tournament is in a valid state, False otherwise
    """
    try:
        logger.info(f"Checking status of tournament {tournament_id}")
        
        # Send the request
        response = requests.get(
            f"{api_url}/api/tournaments/{tournament_id}",
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            result = response.json()
            
            if "tournament" in result:
                tournament = result["tournament"]
                status = tournament.get("status", "")
                current_round = tournament.get("current_round", 0)
                
                logger.info(f"Tournament status: {status} - Round: {current_round}")
                
                # Tournament should be in progress or completed with at least one round
                if status in ["in_progress", "completed"] and current_round >= 1:
                    return True
                else:
                    logger.error(f"Tournament is in unexpected state: {status}, round {current_round}")
                    return False
            else:
                logger.error(f"Invalid tournament response: {result}")
                return False
        else:
            logger.error(f"Failed to get tournament: {response.status_code} - {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Failed to check tournament: {str(e)}")
        raise


def cleanup_tournament(api_url: str, tournament_id: str) -> bool:
    """
    Clean up a test tournament
    
    Args:
        api_url: API URL
        tournament_id: Tournament ID
        
    Returns:
        True if cleanup was successful, False otherwise
    """
    try:
        logger.info(f"Cleaning up tournament {tournament_id}")
        
        # Send the request (this is a fictional endpoint that you may want to implement)
        response = requests.delete(
            f"{api_url}/api/admin/tournaments/{tournament_id}",
            headers={"X-Admin-Key": "deployment_verification"},
            timeout=10
        )
        
        # Check response
        if response.status_code == 200:
            logger.info(f"Tournament {tournament_id} cleaned up successfully")
            return True
        else:
            logger.warning(f"Failed to clean up tournament: {response.status_code} - {response.text}")
            # Don't fail the test if cleanup fails
            return True
    
    except Exception as e:
        logger.warning(f"Failed to clean up tournament: {str(e)}")
        # Don't fail the test if cleanup fails
        return True


async def run_verification(api_url: str, sample_file_path: str, cleanup: bool = True) -> bool:
    """
    Run the full verification test
    
    Args:
        api_url: API URL
        sample_file_path: Path to sample audio file
        cleanup: Whether to clean up after the test
        
    Returns:
        True if all tests passed, False otherwise
    """
    # Check if the sample file exists
    if not os.path.exists(sample_file_path):
        logger.error(f"Sample file {sample_file_path} does not exist")
        return False
    
    # Step 1: Check API health
    if not check_api_health(api_url):
        logger.error("API health check failed")
        return False
    
    # Step 2: Create a test tournament
    success, tournament_id = create_test_tournament(api_url, sample_file_path)
    if not success or not tournament_id:
        logger.error("Failed to create test tournament")
        return False
    
    # Step 3: Execute a battle
    success, task_id = execute_test_battle(api_url, tournament_id)
    if not success or not task_id:
        logger.error("Failed to execute battle")
        return False
    
    # Step 4: Wait for the battle to complete
    if not await wait_for_task_completion(api_url, task_id):
        logger.error("Task did not complete successfully")
        return False
    
    # Step 5: Check tournament status
    if not check_tournament_status(api_url, tournament_id):
        logger.error("Tournament is not in the expected state")
        return False
    
    # Step 6: Clean up (optional)
    if cleanup:
        cleanup_tournament(api_url, tournament_id)
    
    logger.info("🎉 All verification tests passed successfully!")
    return True


async def main():
    """Main entry point"""
    args = parse_args()
    
    logger.info(f"🚀 Starting AI Mixer deployment verification for {args.api_url}")
    
    success = await run_verification(
        api_url=args.api_url,
        sample_file_path=args.sample_file,
        cleanup=not args.no_cleanup
    )
    
    if success:
        logger.info("✅ Deployment verification completed successfully")
        return 0
    else:
        logger.error("❌ Deployment verification failed")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
