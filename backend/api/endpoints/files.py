from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Query
from typing import List, Optional
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_, cast, String
from db import schemas, models, database
from core.security import get_current_user
from services import s3_service
import uuid
import io
import httpx
import os
import base64
from datetime import datetime
from core.config import settings
from PIL import Image

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

@router.post("/search", response_model=List[schemas.FileSystemItem])
def search_items(
    filters: schemas.SearchFilters,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """Search and filter filesystem items based on various criteria"""
    query = db.query(models.FileSystemItem).filter(
        models.FileSystemItem.owner_id == current_user.id
    )
    
    # Text search in file/folder names
    if filters.query:
        query = query.filter(
            models.FileSystemItem.name.ilike(f"%{filters.query}%")
        )
    
    # Filter by item type (file/folder)
    if filters.item_type:
        query = query.filter(models.FileSystemItem.type == filters.item_type)
    
    # Filter by mime type (exact match)
    if filters.mime_type:
        query = query.filter(models.FileSystemItem.mime_type == filters.mime_type)
    
    # Filter by file type (extension-based or category-based)
    if filters.file_type:
        if filters.file_type.lower() in ['image', 'images']:
            # Image files
            query = query.filter(
                models.FileSystemItem.mime_type.like('image/%')
            )
        elif filters.file_type.lower() in ['document', 'documents']:
            # Document files
            query = query.filter(
                or_(
                    models.FileSystemItem.mime_type.in_([
                        'application/pdf',
                        'application/msword',
                        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                        'text/plain'
                    ])
                )
            )
        elif filters.file_type.lower() in ['video', 'videos']:
            # Video files
            query = query.filter(
                models.FileSystemItem.mime_type.like('video/%')
            )
        elif filters.file_type.lower() in ['audio']:
            # Audio files
            query = query.filter(
                models.FileSystemItem.mime_type.like('audio/%')
            )
        else:
            # Assume it's a file extension
            query = query.filter(
                models.FileSystemItem.name.ilike(f"%.{filters.file_type}")
            )
    
    # Filter by ingestion status
    if filters.ingestion_status:
        query = query.filter(
            models.FileSystemItem.ingestion_status == filters.ingestion_status
        )
    
    # Filter by date range (created_at)
    if filters.date_from:
        try:
            date_from = datetime.fromisoformat(filters.date_from.replace('Z', '+00:00'))
            query = query.filter(models.FileSystemItem.created_at >= date_from)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from format")
    
    if filters.date_to:
        try:
            date_to = datetime.fromisoformat(filters.date_to.replace('Z', '+00:00'))
            query = query.filter(models.FileSystemItem.created_at <= date_to)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to format")
    
    # Filter by file size range
    if filters.min_size is not None:
        query = query.filter(models.FileSystemItem.size_bytes >= filters.min_size)
    
    if filters.max_size is not None:
        query = query.filter(models.FileSystemItem.size_bytes <= filters.max_size)
    
    # Order by updated date (most recent first)
    query = query.order_by(models.FileSystemItem.updated_at.desc())
    
    # Limit results to prevent overwhelming responses
    results = query.limit(100).all()
    
    return results

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

    # Step 2: Save file metadata to TheDrive's primary database with pending status.
    new_file = models.FileSystemItem(
        id=file_id, name=file.filename, type="file", s3_key=s3_key,
        mime_type=file.content_type, size_bytes=file_size, owner_id=current_user.id,
        parent_id=None if parentId == 'root' else parentId,
        ingestion_status="pending"  # Start with pending status
    )
    db.add(new_file)
    db.commit()
    db.refresh(new_file)

    # Step 3: Start background ingestion process
    import asyncio
    if file.content_type and file.content_type.startswith('image/'):
        # For images, start both RAG ingestion and image analysis
        asyncio.create_task(process_file_ingestion(file_id, file.filename, contents, file.content_type, parentId))
        asyncio.create_task(process_image_analysis(file_id, file.filename, contents, file.content_type))
    else:
        # For non-images, just do RAG ingestion
        asyncio.create_task(process_file_ingestion(file_id, file.filename, contents, file.content_type, parentId))

    return new_file


async def process_file_ingestion(file_id: str, filename: str, contents: bytes, content_type: str, parent_id: str):
    """Background task to handle file ingestion"""
    db = database.SessionLocal()
    try:
        # Update status to processing
        file_item = db.query(models.FileSystemItem).filter_by(id=file_id).first()
        if not file_item:
            print(f"File {file_id} not found for ingestion")
            return
            
        file_item.ingestion_status = "processing"
        db.commit()
        print(f"Started processing ingestion for file_id {file_id}")

        # Process with RAG service
        files = {'file': (filename, contents, content_type)}
        data = {'file_id': file_id, 'folder_id': parent_id if parent_id != 'root' else ''}
        ingest_url = f"{RAG_API_URL}/ingest/file"

        # Use a very long timeout to allow for model processing and ingestion
        timeout = httpx.Timeout(connect=30.0, read=600.0, write=60.0, pool=None)
        async with httpx.AsyncClient(timeout=timeout) as client:
            response = await client.post(ingest_url, files=files, data=data)
            response.raise_for_status()
            print(f"RAG Ingestion successful for file_id {file_id}: {response.json()}")
            
            # Update status to completed
            file_item.ingestion_status = "completed"
            db.commit()
            print(f"Ingestion completed for file_id {file_id}")

    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        print(f"RAG Ingestion failed for file_id {file_id}. Error: {exc}")
        
        # Update status to failed
        try:
            file_item = db.query(models.FileSystemItem).filter_by(id=file_id).first()
            if file_item:
                file_item.ingestion_status = "failed"
                db.commit()
                print(f"Marked ingestion as failed for file_id {file_id}")
        except Exception as db_exc:
            print(f"Failed to update ingestion status for {file_id}: {db_exc}")
            
    except Exception as e:
        print(f"Unexpected error during ingestion for {file_id}: {e}")
        
        # Update status to failed
        try:
            file_item = db.query(models.FileSystemItem).filter_by(id=file_id).first()
            if file_item:
                file_item.ingestion_status = "failed"
                db.commit()
        except Exception as db_exc:
            print(f"Failed to update ingestion status for {file_id}: {db_exc}")
    finally:
        db.close()


@router.get("/item/{item_id}/ingestion-status")
def get_ingestion_status(item_id: str, db: Session = Depends(database.get_db), current_user: models.User = Depends(get_current_user)):
    """Get the ingestion status of a file"""
    item = db.query(models.FileSystemItem).filter_by(id=item_id, owner_id=current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    return {"id": item.id, "ingestion_status": item.ingestion_status}


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

@router.get("/item/{item_id}/image-analysis")
def get_image_analysis(item_id: str, db: Session = Depends(database.get_db), current_user: models.User = Depends(get_current_user)):
    """Get analysis results for an uploaded image"""
    item = db.query(models.FileSystemItem).filter_by(id=item_id, owner_id=current_user.id).first()
    if not item:
        raise HTTPException(status_code=404, detail="Item not found")
    
    if not item.mime_type or not item.mime_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Item is not an image")
    
    # For now, return basic info about the image
    # In a full implementation, you'd fetch stored analysis from a separate table
    return {
        "id": item.id,
        "filename": item.name,
        "mime_type": item.mime_type,
        "size_bytes": item.size_bytes,
        "ingestion_status": item.ingestion_status,
        "is_image": True,
        "analysis_available": item.ingestion_status == "completed",
        "message": "Image analysis data would be retrieved from analysis storage table"
    }

async def process_image_analysis(file_id: str, filename: str, contents: bytes, content_type: str):
    """Background task to analyze uploaded images"""
    db = database.SessionLocal()
    VISION_API_URL = os.getenv("VISION_API_URL", "http://localhost:8081")
    
    try:
        print(f"Starting image analysis for file_id {file_id}")
        
        # Extract basic image metadata
        try:
            image = Image.open(io.BytesIO(contents))
            metadata = {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "size_bytes": len(contents)
            }
        except Exception as e:
            metadata = {"error": f"Failed to extract metadata: {str(e)}"}
        
        # Perform AI analysis if vision service is available
        analysis_result = None
        try:
            image_b64 = base64.b64encode(contents).decode('utf-8')
            payload = {
                "image": image_b64,
                "prompt": "Analyze this image and describe what you see, including objects, text, colors, and overall content.",
                "max_tokens": 500
            }
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(f"{VISION_API_URL}/analyze", json=payload)
                if response.status_code == 200:
                    analysis_result = response.json()
                    print(f"Image analysis successful for {file_id}")
                else:
                    print(f"Vision API returned status {response.status_code}")
                    
        except Exception as e:
            print(f"Vision API call failed for {file_id}: {e}")
            analysis_result = {
                "analysis": "Image uploaded successfully. AI analysis service unavailable.",
                "metadata": metadata
            }
        
        # Store analysis results (could be in a separate table, for now just log)
        print(f"Image analysis completed for {file_id}:")
        print(f"- Metadata: {metadata}")
        if analysis_result:
            print(f"- Analysis: {analysis_result.get('analysis', 'No analysis available')}")
        
        # Update the file status to indicate analysis is complete
        file_item = db.query(models.FileSystemItem).filter_by(id=file_id).first()
        if file_item and file_item.ingestion_status == "pending":
            # Only update if still pending (RAG might have already updated it)
            file_item.ingestion_status = "completed"
            db.commit()
            
    except Exception as e:
        print(f"Image analysis failed for {file_id}: {e}")
        
        # Update status to failed if there was an error
        try:
            file_item = db.query(models.FileSystemItem).filter_by(id=file_id).first()
            if file_item:
                file_item.ingestion_status = "failed"
                db.commit()
        except Exception as db_exc:
            print(f"Failed to update ingestion status for {file_id}: {db_exc}")
            
    finally:
        db.close()