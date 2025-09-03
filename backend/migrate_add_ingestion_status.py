#!/usr/bin/env python3
"""
Migration script to add ingestion_status column to filesystem_items table
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from core.config import settings
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def migrate_add_ingestion_status():
    """Add ingestion_status column to filesystem_items table"""
    
    engine = create_engine(settings.DATABASE_URL)
    
    try:
        with engine.begin() as conn:
            logger.info("Creating ingestion_status enum type...")
            
            # First, create the enum type if it doesn't exist
            conn.execute(text("""
                DO $$ BEGIN
                    CREATE TYPE ingestion_status_enum AS ENUM ('pending', 'processing', 'completed', 'failed');
                EXCEPTION
                    WHEN duplicate_object THEN null;
                END $$;
            """))
            
            logger.info("Adding ingestion_status column to filesystem_items table...")
            
            # Add the column with default value
            conn.execute(text("""
                ALTER TABLE filesystem_items 
                ADD COLUMN IF NOT EXISTS ingestion_status ingestion_status_enum DEFAULT 'pending';
            """))
            
            # Set default value for existing records based on file type
            logger.info("Setting default ingestion_status for existing records...")
            conn.execute(text("""
                UPDATE filesystem_items 
                SET ingestion_status = CASE 
                    WHEN type = 'folder' THEN NULL
                    ELSE 'completed'::ingestion_status_enum
                END
                WHERE ingestion_status IS NULL OR ingestion_status = 'pending'::ingestion_status_enum;
            """))
            
            logger.info("Migration completed successfully!")
            
    except Exception as e:
        logger.error(f"Migration failed: {str(e)}")
        raise

if __name__ == "__main__":
    migrate_add_ingestion_status()
