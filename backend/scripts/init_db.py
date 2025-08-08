import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv

load_dotenv()
client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DB_NAME")]

async def init():
    # Create collections and indexes
    await db.users.create_index("user_id", unique=True)
    await db.predictions.create_index([("user_id", 1), ("timestamp", -1)])
    print("✅ Database initialized with indexes.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(init())
