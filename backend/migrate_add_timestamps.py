#!/usr/bin/env python3
"""
Migration script to add created_at and updated_at columns to filesystem_items table.
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from sqlalchemy import create_engine, text
from core.config import settings

def migrate():
    engine = create_engine(settings.DATABASE_URL)
    
    with engine.connect() as connection:
        # Start a transaction
        trans = connection.begin()
        try:
            # Add created_at column with default value
            connection.execute(text("""
                ALTER TABLE filesystem_items 
                ADD COLUMN created_at TIMESTAMP WITH TIME ZONE 
                DEFAULT CURRENT_TIMESTAMP NOT NULL
            """))
            
            # Add updated_at column with default value
            connection.execute(text("""
                ALTER TABLE filesystem_items 
                ADD COLUMN updated_at TIMESTAMP WITH TIME ZONE 
                DEFAULT CURRENT_TIMESTAMP NOT NULL
            """))
            
            # Create trigger for auto-updating updated_at
            connection.execute(text("""
                CREATE OR REPLACE FUNCTION update_updated_at_column()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = CURRENT_TIMESTAMP;
                    RETURN NEW;
                END;
                $$ language 'plpgsql'
            """))
            
            connection.execute(text("""
                CREATE TRIGGER update_filesystem_items_updated_at
                    BEFORE UPDATE ON filesystem_items
                    FOR EACH ROW
                    EXECUTE FUNCTION update_updated_at_column()
            """))
            
            print("Migration completed successfully!")
            trans.commit()
            
        except Exception as e:
            print(f"Migration failed: {e}")
            trans.rollback()
            raise

if __name__ == "__main__":
    migrate()