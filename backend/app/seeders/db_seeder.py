#!/usr/bin/env python3
"""
Database seeder for MindGuard application.
Handles MongoDB connection and data insertion for exercises, games, and initial data.
"""

import sys
import os
import logging
from typing import List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorClient
import asyncio

# Add the backend directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from app.models.health_recommendations import Exercise, Game
from app.models.facial_metrics import ComprehensiveFacialAnalysis
from app.services.db import get_db
from seed_initial_data import DataSeeder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatabaseSeeder:
    """Handles database operations for seeding."""
    
    def __init__(self):
        self.db = None
        self.data_seeder = DataSeeder()
        
    async def connect_db(self):
        """Connect to MongoDB database."""
        try:
            self.db = await get_db()
            logger.info("✅ Connected to MongoDB")
        except Exception as e:
            logger.error(f"❌ Failed to connect to MongoDB: {e}")
            raise
    
    async def clear_collections(self):
        """Clear existing collections (optional - for fresh start)."""
        try:
            if self.db is not None:
                collections = ['exercises', 'games', 'facial_analyses', 'user_sessions']
                for collection_name in collections:
                    await self.db[collection_name].delete_many({})
                logger.info("🧹 Cleared existing collections")
        except Exception as e:
            logger.warning(f"⚠️  Warning: Could not clear collections: {e}")
    
    async def seed_exercises_to_db(self, exercises: List[Exercise]):
        """Insert exercises into MongoDB."""
        try:
            if self.db is None:
                await self.connect_db()
            
            collection = self.db['exercises']
            
            # Convert Exercise objects to dictionaries
            exercise_dicts = []
            for exercise in exercises:
                # Handle both Pydantic v1 and v2
                if hasattr(exercise, 'model_dump'):
                    exercise_dict = exercise.model_dump()
                else:
                    exercise_dict = exercise.dict()
                exercise_dict['_id'] = exercise.id  # Use custom ID
                exercise_dicts.append(exercise_dict)
            
            # Insert exercises
            if exercise_dicts:
                result = await collection.insert_many(exercise_dicts)
                logger.info(f"✅ Inserted {len(result.inserted_ids)} exercises into database")
            
        except Exception as e:
            logger.error(f"❌ Error seeding exercises to database: {e}")
            raise
    
    async def seed_games_to_db(self, games: List[Game]):
        """Insert games into MongoDB."""
        try:
            if self.db is None:
                await self.connect_db()
            
            collection = self.db['games']
            
            # Convert Game objects to dictionaries
            game_dicts = []
            for game in games:
                # Handle both Pydantic v1 and v2
                if hasattr(game, 'model_dump'):
                    game_dict = game.model_dump()
                else:
                    game_dict = game.dict()
                game_dict['_id'] = game.id  # Use custom ID
                game_dicts.append(game_dict)
            
            # Insert games
            if game_dicts:
                result = await collection.insert_many(game_dicts)
                logger.info(f"✅ Inserted {len(result.inserted_ids)} games into database")
            
        except Exception as e:
            logger.error(f"❌ Error seeding games to database: {e}")
            raise
    
    async def seed_sample_analysis_to_db(self, analysis: ComprehensiveFacialAnalysis):
        """Insert sample facial analysis into MongoDB."""
        try:
            if self.db is None:
                await self.connect_db()
            
            collection = self.db['facial_analyses']
            
            # Convert analysis to dictionary
            # Handle both Pydantic v1 and v2
            if hasattr(analysis, 'model_dump'):
                analysis_dict = analysis.model_dump()
            else:
                analysis_dict = analysis.dict()
            
            # Insert sample analysis
            result = await collection.insert_one(analysis_dict)
            logger.info(f"✅ Inserted sample facial analysis with ID: {result.inserted_id}")
            
        except Exception as e:
            logger.error(f"❌ Error seeding facial analysis to database: {e}")
            raise
    
    async def check_data_exists(self) -> Dict[str, int]:
        """Check if data already exists in collections."""
        try:
            if self.db is None:
                await self.connect_db()
            
            counts = {}
            collections = ['exercises', 'games', 'facial_analyses']
            
            for collection_name in collections:
                count = await self.db[collection_name].count_documents({})
                counts[collection_name] = count
            
            return counts
            
        except Exception as e:
            logger.error(f"❌ Error checking existing data: {e}")
            return {}
    
    async def seed_all_to_database(self, force_refresh: bool = False):
        """Seed all data to MongoDB database."""
        logger.info("🌱 Starting database seeding process...")
        
        try:
            await self.connect_db()
            
            # Check existing data
            existing_counts = await self.check_data_exists()
            logger.info(f"📊 Current database state: {existing_counts}")
            
            if force_refresh:
                logger.info("🔄 Force refresh enabled - clearing existing data")
                await self.clear_collections()
            
            # Generate new data
            logger.info("🎲 Generating fresh data...")
            data = self.data_seeder.seed_all()
            
            # Seed to database
            await self.seed_exercises_to_db(data['exercises'])
            await self.seed_games_to_db(data['games'])
            await self.seed_sample_analysis_to_db(data['sample_analysis'])
            
            # Verify insertion
            final_counts = await self.check_data_exists()
            logger.info(f"✅ Final database state: {final_counts}")
            
            logger.info("🎉 Database seeding completed successfully!")
            return final_counts
            
        except Exception as e:
            logger.error(f"❌ Database seeding failed: {e}")
            raise

async def main():
    """Main database seeder function."""
    seeder = DatabaseSeeder()
    
    # Check command line arguments
    force_refresh = len(sys.argv) > 1 and sys.argv[1] == "--force"
    
    try:
        await seeder.seed_all_to_database(force_refresh=force_refresh)
    except Exception as e:
        logger.error(f"❌ Seeding failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
