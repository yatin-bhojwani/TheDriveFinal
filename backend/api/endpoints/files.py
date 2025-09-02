from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
from typing import List
from sqlalchemy.orm import Session
from db import schemas, models, database
from core.security import get_current_user
from services import s3_service
import uuid
import io
import httpx
import os
from core.config import settings

router = APIRouter()
RAG_API_URL = os.getenv("RAG_API_URL", "http://api:8080")

@router.get("/items", response_model=schemas.FolderResponse)
def get_items(parentId: str = 'root', db: Session = Depends(database.get_db), current_user: models.User = Depends(get_current_user)):
    parent_db_id = None if parentId == 'root' else parentId
    items = db.query(models.FileSystemItem).filter(
        models.FileSystemItem.owner_id == current_user.id,
        models.FileSystemItem.parent_id == parent_db_id
    ).all()

    path = [{"id": "root", "name": "My Drive"}]
    if parentId != 'root':
        current_folder = db.query(models.FileSystemItem).filter_by(id=parentId, owner_id=current_user.id).first()
        if current_folder:
            # This logic can be expanded to build the full path if needed
            path.append({"id": current_folder.id, "name": current_folder.name})

    return {"items": items, "path": path}

@router.post("/upload", response_model=schemas.FileSystemItem, status_code=201)
async def upload_file(
    parentId: str,
    file: UploadFile = File(...),
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    # Validate that parent folder exists (if not root)
    if parentId != 'root':
        parent_folder = db.query(models.FileSystemItem).filter_by(
            id=parentId, 
            owner_id=current_user.id, 
            type='folder'
        ).first()
        if not parent_folder:
            raise HTTPException(status_code=404, detail="Parent folder not found")
    
    file_id = f"file-{uuid.uuid4()}"
    s3_key = f"{current_user.id}/{file_id}/{file.filename}"

    contents = await file.read()
    file_size = len(contents)

    # Step 1: Upload the file to S3 for persistent storage.
    try:
        s3_service.upload_file_obj(
            io.BytesIO(contents),
            settings.S3_BUCKET_NAME,
            s3_key,
            extra_args={"ContentType": file.content_type, "ContentDisposition": "inline"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload to S3: {str(e)}")

    # Step 2: Save file metadata to TheDrive's primary database.
    new_file = models.FileSystemItem(
        id=file_id, name=file.filename, type="file", s3_key=s3_key,
        mime_type=file.content_type, size_bytes=file_size, owner_id=current_user.id,
        parent_id=None if parentId == 'root' else parentId
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    # Step 3: Forward the file to the RAG service for ingestion.
    try:
        files = {'file': (file.filename, contents, file.content_type)}
        data = {'file_id': file_id, 'folder_id': parentId if parentId != 'root' else ''}
        ingest_url = f"{RAG_API_URL}/ingest/file"

        # Use a very long timeout to allow for model processing and ingestion.
        # Set separate timeouts for connect vs read operations
        timeout = httpx.Timeout(connect=100.0, read=6000.0, write=6001.0, pool=None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(ingest_url, files=files, data=data)
            response.raise_for_status()
            print(f"RAG Ingestion successful for file_id {file_id}: {response.json()}")

    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        # If ingestion fails, attempt rollback but don't fail if rollback also fails
        print(f"RAG Ingestion failed for file_id {file_id}. Error: {exc}")
        print("Attempting rollback...")
        
        try:
            # Try to rollback database entry
            db.refresh(new_file)  # Refresh to ensure we have the latest state
            db.delete(new_file)
            db.commit()
            print(f"Database rollback successful for file_id {file_id}")
        except Exception as db_exc:
            print(f"Database rollback failed for file_id {file_id}: {db_exc}")
            # Don't fail the request if rollback fails - log and continue
            
        try:
            # Try to rollback S3 upload
            s3_service.delete_file(settings.S3_BUCKET_NAME, s3_key)
            print(f"S3 rollback successful for file_id {file_id}")
        except Exception as s3_exc:
            print(f"S3 rollback failed for file_id {file_id}: {s3_exc}")
            # Don't fail the request if S3 rollback fails - log and continue
        
        detail = f"Error from RAG service: {exc.response.text}" if isinstance(exc, httpx.HTTPStatusError) else f"Failed to connect to RAG service: {exc}"
        raise HTTPException(status_code=502, detail=detail)

    return new_file

@router.post("/folder", response_model=schemas.FileSystemItem, status_code=201)
def create_folder(folder: schemas.FolderCreate, db: Session = Depends(database.get_db), current_user: models.User = Depends(get_current_user)):
    # Validate that parent folder exists (if not root)
    if folder.parentId != 'root':
        parent_folder = db.query(models.FileSystemItem).filter_by(
            id=folder.parentId, 
            owner_id=current_user.id, 
            type='folder'
        ).first()
        if not parent_folder:
            raise HTTPException(status_code=404, detail="Parent folder not found")
    
    parent_db_id = None if folder.parentId == 'root' else folder.parentId
    
    existing = db.query(models.FileSystemItem).filter_by(
        owner_id=current_user.id, parent_id=parent_db_id, name=folder.name
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="An item with this name already exists")

    new_folder = models.FileSystemItem(
        id=f"folder-{uuid.uuid4()}", name=folder.name, type="folder",
        owner_id=current_user.id, parent_id=parent_db_id
    )
    db.add(new_folder)
    db.commit()
    db.refresh(new_folder)
    return new_folder

@router.put("/item/{item_id}", response_model=schemas.FileSystemItem)
def rename_item(item_id: str, item_update: schemas.ItemUpdate, db: Session = Depends(database.get_db), current_user: models.User = Depends(get_current_user)):
    item = db.query(models.FileSystemItem).filter_by(id=item_id, owner_id=current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    existing = db.query(models.FileSystemItem).filter(
        models.FileSystemItem.id != item_id,
        models.FileSystemItem.parent_id == item.parent_id,
        models.FileSystemItem.name == item_update.name,
        models.FileSystemItem.owner_id == current_user.id
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="An item with this name already exists")

    item.name = item_update.name
    db.commit()
    db.refresh(item)
    return item



@router.delete("/item/{item_id}", status_code=204)
async def delete_item(item_id: str, db: Session = Depends(database.get_db), current_user: models.User = Depends(get_current_user)):
    item = db.query(models.FileSystemItem).filter_by(id=item_id, owner_id=current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")

    def collect_items_to_delete(item_to_check):
        """Recursively collect all items (files and folders) that will be deleted"""
        items_to_delete = [item_to_check]
        if item_to_check.type == 'folder':
            # Get all children recursively
            children = db.query(models.FileSystemItem).filter_by(
                parent_id=item_to_check.id, 
                owner_id=current_user.id
            ).all()
            for child in children:
                items_to_delete.extend(collect_items_to_delete(child))
        return items_to_delete

    # Collect all items that will be deleted (including nested files and folders)
    all_items_to_delete = collect_items_to_delete(item)
    
    # Store info for cleanup operations
    items_info = []
    for item_to_delete in all_items_to_delete:
        items_info.append({
            "id": item_to_delete.id,
            "type": item_to_delete.type,
            "s3_key": item_to_delete.s3_key
        })

    rag_url = os.getenv("RAG_API_URL", "http://localhost:8080")

    # Step 1: RAG cleanup first (while database references still exist)
    try:
        # Delete individual files from RAG service first
        for item_info in items_info:
            if item_info["type"] == 'file':
                try:
                    async with httpx.AsyncClient() as client:
                        url = f"{rag_url}/files/{item_info['id']}"
                        await client.delete(url, timeout=45.0)
                        print(f"Deleted RAG file: {item_info['id']}")
                except Exception as rag_error:
                    print(f"Failed to delete RAG file {item_info['id']}: {rag_error}")

        # If the main item is a folder, delete the folder from RAG service
        if item.type == 'folder':
            try:
                async with httpx.AsyncClient() as client:
                    url = f"{rag_url}/folders/{item.id}"
                    await client.delete(url, timeout=300.0)
                    print(f"Deleted RAG folder: {item.id}")
            except Exception as rag_error:
                print(f"Failed to delete RAG folder {item.id}: {rag_error}")

    except Exception as e:
        print(f"RAG Cleanup Error: Could not remove items from RAG service. Error: {e}")

    # Step 2: S3 cleanup for all files
    try:
        for item_info in items_info:
            if item_info["type"] == 'file' and item_info["s3_key"]:
                try:
                    s3_service.delete_file(settings.S3_BUCKET_NAME, item_info["s3_key"])
                    print(f"Deleted S3 file: {item_info['s3_key']}")
                except Exception as s3_error:
                    print(f"Failed to delete S3 file {item_info['s3_key']}: {s3_error}")
    except Exception as e:
        print(f"S3 Cleanup Error: Could not remove files from S3. Error: {e}")

    # Step 3: Delete from database in correct order (children first, then parent)
    try:
        # First, manually delete all children to ensure proper cleanup
        def delete_children_recursively(parent_item):
            """Delete all children of a folder recursively"""
            children = db.query(models.FileSystemItem).filter_by(
                parent_id=parent_item.id,
                owner_id=current_user.id
            ).all()
            
            for child in children:
                if child.type == 'folder':
                    # Recursively delete children of this child folder
                    delete_children_recursively(child)
                # Delete the child (file or empty folder)
                db.delete(child)
                print(f"Deleted child from database: {child.id}")
        
        # If it's a folder, delete all its children first
        if item.type == 'folder':
            delete_children_recursively(item)
        
        # Finally delete the main item
        db.delete(item)
        db.commit()
        print(f"Deleted from database: {item.id}")
        
    except Exception as db_error:
        print(f"Database deletion failed for {item.id}: {db_error}")
        db.rollback()
        raise HTTPException(status_code=500, detail="Failed to delete item from database")

    return
@router.get("/item/{item_id}/view-link", response_model=schemas.ViewLinkResponse)
def get_view_link(item_id: str, db: Session = Depends(database.get_db), current_user: models.User = Depends(get_current_user)):
    item = db.query(models.FileSystemItem).filter_by(id=item_id, owner_id=current_user.id).first()
    if not item or item.type != 'file' or not item.s3_key:
        raise HTTPException(status_code=404, detail="File not found or is not a viewable file.")
    
    url = s3_service.generate_presigned_url(settings.S3_BUCKET_NAME, item.s3_key )
    return {"url": url}