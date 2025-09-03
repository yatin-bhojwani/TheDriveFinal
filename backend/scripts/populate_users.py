#!/usr/bin/env python3
"""
Script to populate users and their data from a folder structure.

Expected folder structure:
/users_folder/
├── user1/
│   ├── info.json
│   └── data/
│       ├── file1.txt
│       └── file2.pdf
├── user2/
│   ├── info.json
│   └── data/
│       └── document.docx
└── ...

info.json format:
{
  "name": "John Doe",
  "password": "P@ssw0rd123",
  "email": "johndoe@example.com"
}
"""

import os
import json
import sys
import uuid
import mimetypes
from pathlib import Path
from typing import Optional

# Add the backend directory to the path so we can import our modules
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from sqlalchemy.orm import Session
from db.database import SessionLocal, engine
from db.models import User, FileSystemItem, Base
from core.security import get_password_hash
import boto3
import httpx
from core.config import settings

# Initialize S3 client
s3_client = boto3.client(
    's3',
    aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
    aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
    region_name=settings.AWS_DEFAULT_REGION
)

def create_tables():
    """Create database tables if they don't exist."""
    Base.metadata.create_all(bind=engine)

def upload_file_to_s3(file_path: str, s3_key: str) -> bool:
    """Upload a file to S3 and return success status."""
    try:
        s3_client.upload_file(file_path, settings.S3_BUCKET_NAME, s3_key)
        print(f"✓ Uploaded {file_path} to S3 as {s3_key}")
        return True
    except Exception as e:
        print(f"✗ Failed to upload {file_path} to S3: {str(e)}")
        return False

def get_file_size(file_path: str) -> int:
    """Get file size in bytes."""
    return os.path.getsize(file_path)

def get_mime_type(file_path: str) -> Optional[str]:
    """Get MIME type of a file."""
    mime_type, _ = mimetypes.guess_type(file_path)
    return mime_type

def create_user(db: Session, email: str, password: str) -> Optional[User]:
    """Create a new user in the database."""
    # Check if user already exists
    existing_user = db.query(User).filter(User.email == email).first()
    if existing_user:
        print(f"⚠ User with email {email} already exists, skipping...")
        return existing_user
    
    # Create new user
    hashed_password = get_password_hash(password)
    user = User(email=email, hashed_password=hashed_password)
    db.add(user)
    db.commit()
    db.refresh(user)
    print(f"✓ Created user: {email}")
    return user

def create_filesystem_item(
    db: Session, 
    item_id: str,
    name: str, 
    item_type: str,
    owner_id: int,
    parent_id: Optional[str] = None,
    s3_key: Optional[str] = None,
    mime_type: Optional[str] = None,
    size_bytes: Optional[int] = None,
    ingestion_status: Optional[str] = None
) -> FileSystemItem:
    """Create a filesystem item (file or folder)."""
    item = FileSystemItem(
        id=item_id,
        name=name,
        type=item_type,
        owner_id=owner_id,
        parent_id=parent_id,
        s3_key=s3_key,
        mime_type=mime_type,
        size_bytes=size_bytes,
        ingestion_status=ingestion_status
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    return item

def ingest_file_in_rag(file_path: str, file_id: str, owner_id: int, parent_id: str):
    """Sends a file to the RAG API for ingestion."""
    mime_type = get_mime_type(file_path)
    file_name = os.path.basename(file_path)
    
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_name, f, mime_type)}
            data = {
                "file_id": file_id,
                "owner_id": str(owner_id),
                "folder_id": parent_id,
            }
            
            with httpx.Client() as client:
                response = client.post(
                    f"{settings.RAG_API_URL}/ingest/file",
                    files=files,
                    data=data,
                    timeout=300
                )
                response.raise_for_status()
            
            print(f"✓ Ingestion request successful for {file_name}")
            return "completed"
            
    except httpx.HTTPStatusError as e:
        print(f"✗ Failed to ingest {file_name}. Status: {e.response.status_code}, Response: {e.response.text}")
        return "failed"
    except Exception as e:
        print(f"✗ An unexpected error occurred during ingestion for {file_name}: {str(e)}")
        return "failed"

def process_user_folder(db: Session, user_folder_path: str) -> bool:
    """Process a single user folder."""
    user_folder = Path(user_folder_path)
    info_json_path = user_folder / "info.json"
    data_folder_path = user_folder / "data"
    
    # Check if info.json exists
    if not info_json_path.exists():
        print(f"✗ No info.json found in {user_folder_path}")
        return False
    
    # Read user info
    try:
        with open(info_json_path, 'r', encoding='utf-8') as f:
            user_info = json.load(f)
    except Exception as e:
        print(f"✗ Failed to read info.json from {user_folder_path}: {str(e)}")
        return False
    
    # Validate required fields
    if 'email' not in user_info or 'password' not in user_info:
        print(f"✗ Missing email or password in {info_json_path}")
        return False
    
    # Create user
    user = create_user(db, user_info['email'], user_info['password'])
    if not user:
        return False
    
    # Process data folder if it exists
    if data_folder_path.exists() and data_folder_path.is_dir():
        process_files_recursive(db, data_folder_path, user.id, None)
    
    return True

def process_files_recursive(db: Session, folder_path: Path, owner_id: int, parent_id: Optional[str]):
    """Recursively process files and folders."""
    for item_path in folder_path.iterdir():
        if item_path.is_file():
            # Process file
            file_id = f"file-{uuid.uuid4()}"
            s3_key = f"{owner_id}/{file_id}/{item_path.name}"
            
            # Upload to S3
            if upload_file_to_s3(str(item_path), s3_key):
                # Create database record with 'pending' status
                item = create_filesystem_item(
                    db=db,
                    item_id=file_id,
                    name=item_path.name,
                    item_type="file",
                    owner_id=owner_id,
                    parent_id=parent_id,
                    s3_key=s3_key,
                    mime_type=get_mime_type(str(item_path)),
                    size_bytes=get_file_size(str(item_path)),
                    ingestion_status="pending"
                )
                print(f"✓ Processed file: {item_path.name}, status: pending")

                # Ingest file and get status
                ingestion_status = ingest_file_in_rag(str(item_path), file_id, owner_id, parent_id)
                
                # Update ingestion status
                item.ingestion_status = ingestion_status
                db.commit()
                print(f"✓ Updated ingestion status to '{ingestion_status}' for {item.name}")

        elif item_path.is_dir():
            # Process folder
            folder_id = f"folder-{uuid.uuid4()}"
            create_filesystem_item(
                db=db,
                item_id=folder_id,
                name=item_path.name,
                item_type="folder",
                owner_id=owner_id,
                parent_id=parent_id
            )
            print(f"✓ Created folder: {item_path.name}")
            
            # Recursively process contents
            process_files_recursive(db, item_path, owner_id, folder_id)

def main():
    """Main function to populate users from folder structure."""
    if len(sys.argv) != 2:
        print("Usage: python populate_users.py <users_folder_path>")
        sys.exit(1)
    
    users_folder = Path(sys.argv[1])
    
    if not users_folder.exists() or not users_folder.is_dir():
        print(f"Error: {users_folder} is not a valid directory")
        sys.exit(1)
    
    print(f"Starting user population from: {users_folder}")
    print("=" * 50)
    
    # Create database tables if they don't exist
    create_tables()
    
    # Get database session
    db = SessionLocal()
    
    try:
        processed_count = 0
        failed_count = 0
        
        # Process each user folder
        for user_folder in users_folder.iterdir():
            if user_folder.is_dir():
                print(f"\nProcessing user folder: {user_folder.name}")
                if process_user_folder(db, str(user_folder)):
                    processed_count += 1
                else:
                    failed_count += 1
        
        print("\n" + "=" * 50)
        print(f"Population complete!")
        print(f"✓ Successfully processed: {processed_count} users")
        print(f"✗ Failed to process: {failed_count} users")
        
    except Exception as e:
        print(f"Error during population: {str(e)}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    main()