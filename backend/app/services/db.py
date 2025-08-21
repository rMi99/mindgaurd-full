import os
import logging
from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional

logger = logging.getLogger(__name__)

# MongoDB configuration
MONGO_DETAILS = os.getenv("MONGO_DETAILS", "mongodb://localhost:27017")
DATABASE_NAME = os.getenv("DATABASE_NAME", "mindguard_db")

# Global database instance
client: Optional[AsyncIOMotorClient] = None
database = None

async def connect_to_mongo():
    """Create database connection"""
    global client, database
    try:
        client = AsyncIOMotorClient(MONGO_DETAILS)
        database = client[DATABASE_NAME]
        
        # Test the connection
        await client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        
    except Exception as e:
        logger.error(f"Error connecting to MongoDB: {e}")
        raise

async def close_mongo_connection():
    """Close database connection"""
    global client
    if client:
        client.close()
        logger.info("MongoDB connection closed")

async def get_db():
    """Get database instance"""
    if database is None:
        await connect_to_mongo()
    return database