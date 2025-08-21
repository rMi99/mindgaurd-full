import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()

# MongoDB connection with defaults
MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("DB_NAME", "mindguard_db")

client = AsyncIOMotorClient(MONGODB_URI)
db = client[DB_NAME]

def get_database():
    """Get database instance for dependency injection"""
    return db

async def get_database_async():
    """Async version of get_database for FastAPI dependencies"""
    return db
