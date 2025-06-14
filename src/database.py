#!/usr/bin/env python3
"""
🗄️ Database Access Layer
======================

Provides centralized database access for the application.
"""

import logging
from typing import Any, Dict, List, Optional, Union, Tuple
import asyncio
import sqlalchemy
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.future import select
from sqlalchemy.sql import text

from src.environment_config import db_config, is_development

# Configure logging
logger = logging.getLogger(__name__)

# Get database configuration
DB_CONFIG = db_config()
DATABASE_URL = DB_CONFIG["url"]

# Check if we need to convert SQLite URL to async format
if DATABASE_URL.startswith("sqlite:"):
    if not DATABASE_URL.startswith("sqlite+aiosqlite:"):
        DATABASE_URL = DATABASE_URL.replace("sqlite:", "sqlite+aiosqlite:")

# Create async engine
engine = create_async_engine(
    DATABASE_URL,
    echo=DB_CONFIG["echo"],
    pool_size=DB_CONFIG["pool_size"],
    max_overflow=DB_CONFIG["max_overflow"],
    pool_pre_ping=True,
    pool_recycle=3600,  # Recycle connections after 1 hour
)

# Create session factory
AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False,
)

# Create declarative base
Base = declarative_base()

# Database dependency
async def get_db():
    """
    Database session dependency for FastAPI
    
    Yields:
        AsyncSession: Database session
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception as e:
            await session.rollback()
            logger.error(f"Database session error: {str(e)}", exc_info=True)
            raise
        finally:
            await session.close()

# Create all tables
async def create_tables():
    """Create all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created")

# Drop all tables (for testing only)
async def drop_tables():
    """Drop all database tables"""
    if not is_development():
        logger.error("Attempted to drop tables in non-development environment")
        return
    
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.info("Database tables dropped")

# Execute raw SQL query
async def execute_query(query: str, params: Dict[str, Any] = None) -> Tuple[List[Dict[str, Any]], int]:
    """
    Execute a raw SQL query
    
    Args:
        query: SQL query string
        params: Query parameters
        
    Returns:
        Tuple of (results, row_count)
    """
    params = params or {}
    
    async with AsyncSessionLocal() as session:
        try:
            result = await session.execute(text(query), params)
            
            if query.strip().lower().startswith("select"):
                # For SELECT queries, return results and count
                rows = [dict(row) for row in result.mappings()]
                return rows, len(rows)
            else:
                # For non-SELECT queries, return empty results and row count
                return [], result.rowcount
        except Exception as e:
            logger.error(f"Query execution error: {str(e)}", exc_info=True)
            raise
        finally:
            await session.close()

# Database health check
async def check_database_health() -> Dict[str, Any]:
    """
    Check database health
    
    Returns:
        Dict with health status information
    """
    start_time = asyncio.get_event_loop().time()
    status = "healthy"
    error = None
    
    try:
        # Execute a simple query to check database connection
        async with AsyncSessionLocal() as session:
            await session.execute(text("SELECT 1"))
    except Exception as e:
        status = "unhealthy"
        error = str(e)
        logger.error(f"Database health check failed: {error}", exc_info=True)
    
    duration = asyncio.get_event_loop().time() - start_time
    
    return {
        "status": status,
        "response_time": duration,
        "error": error,
        "database_url": DATABASE_URL.split("@")[-1] if "@" in DATABASE_URL else "sqlite"
    }

# Initialize database (create tables)
async def initialize_database():
    """Initialize the database"""
    try:
        await create_tables()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {str(e)}", exc_info=True)
        raise

# If run directly, initialize the database
if __name__ == "__main__":
    import asyncio
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    
    # Run database initialization
    asyncio.run(initialize_database())
