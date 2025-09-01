import os
from motor.motor_asyncio import AsyncIOMotorClient
from dotenv import load_dotenv
import pymongo.errors

load_dotenv()
client = AsyncIOMotorClient(os.getenv("MONGODB_URI"))
db = client[os.getenv("DB_NAME")]

async def init():
    # Create collections and indexes with error handling
    try:
        await db.users.create_index("user_id", unique=True)
        print("✅ Created user_id index")
    except pymongo.errors.DuplicateKeyError as e:
        print("⚠️  user_id index already exists, skipping...")
    except Exception as e:
        print(f"⚠️  Warning creating user_id index: {e}")
    
    try:
        await db.predictions.create_index([("user_id", 1), ("timestamp", -1)])
        print("✅ Created predictions index")
    except pymongo.errors.DuplicateKeyError as e:
        print("⚠️  predictions index already exists, skipping...")
    except Exception as e:
        print(f"⚠️  Warning creating predictions index: {e}")
    
    print("✅ Database initialization completed.")

if __name__ == "__main__":
    import asyncio
    asyncio.run(init())
