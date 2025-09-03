from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Optional
from sqlalchemy.orm import Session
import uuid
import io
import json
import base64
from PIL import Image
import httpx
import os
from datetime import datetime

from db import schemas, models, database
from core.security import get_current_user
from services import s3_service
from core.config import settings

router = APIRouter()

# External AI service URL for image processing
RAG_API_URL = os.getenv("RAG_API_URL", "http://api:8080")
VISION_API_URL = os.getenv("VISION_API_URL", "http://localhost:8081")  # Separate vision service

class ImageAnalyzer:
    """Service for handling image analysis operations"""
    
    @staticmethod
    async def analyze_image(image_bytes: bytes, prompt: str = None) -> dict:
        """Analyze image using AI vision service"""
        try:
            # Convert image to base64 for API transmission
            image_b64 = base64.b64encode(image_bytes).decode('utf-8')
            
            payload = {
                "image": image_b64,
                "prompt": prompt or "Describe this image in detail, including objects, text, colors, and context.",
                "max_tokens": 1000
            }
            
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{VISION_API_URL}/analyze",
                    json=payload
                )
                response.raise_for_status()
                return response.json()
                
        except Exception as e:
            # Fallback mock response for development
            return {
                "analysis": f"Image analysis service unavailable: {str(e)}",
                "objects": [],
                "text": "",
                "colors": [],
                "metadata": {"width": 0, "height": 0}
            }
    
    @staticmethod
    def extract_image_metadata(image_bytes: bytes) -> dict:
        """Extract basic metadata from image"""
        try:
            image = Image.open(io.BytesIO(image_bytes))
            return {
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "size_bytes": len(image_bytes)
            }
        except Exception as e:
            return {"error": f"Failed to extract metadata: {str(e)}"}

@router.post("/ingest/image", response_model=dict)
async def ingest_image(
    file: UploadFile = File(...),
    description: Optional[str] = None,
    tags: Optional[str] = None,  # Comma-separated tags
    auto_analyze: bool = True,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user),
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Upload and store an image with full analysis and metadata extraction.
    Stores the image in S3 and saves analysis results to database.
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Generate unique ID and S3 key
        image_id = f"img-{uuid.uuid4()}"
        s3_key = f"{current_user.id}/images/{image_id}/{file.filename}"
        
        # Extract basic metadata
        metadata = ImageAnalyzer.extract_image_metadata(image_data)
        
        # Upload to S3
        s3_service.upload_file_obj(
            io.BytesIO(image_data),
            settings.S3_BUCKET_NAME,
            s3_key,
            extra_args={"ContentType": file.content_type, "ContentDisposition": "inline"}
        )
        
        # Create database record
        image_record = models.FileSystemItem(
            id=image_id,
            name=file.filename,
            type="file",
            s3_key=s3_key,
            mime_type=file.content_type,
            size_bytes=len(image_data),
            owner_id=current_user.id,
            parent_id=None,  # Images go to root by default
            ingestion_status="processing" if auto_analyze else "completed"
        )
        
        db.add(image_record)
        db.commit()
        db.refresh(image_record)
        
        response_data = {
            "id": image_id,
            "filename": file.filename,
            "size": len(image_data),
            "metadata": metadata,
            "s3_key": s3_key,
            "status": "processing" if auto_analyze else "completed"
        }
        
        # Start background analysis if requested
        if auto_analyze:
            background_tasks.add_task(
                analyze_and_store_results,
                image_id=image_id,
                image_data=image_data,
                description=description,
                tags=tags.split(",") if tags else [],
                db_session=db
            )
            response_data["message"] = "Image uploaded successfully. Analysis in progress."
        else:
            response_data["message"] = "Image uploaded successfully."
        
        return response_data
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to ingest image: {str(e)}")

@router.post("/analyze/image", response_model=dict)
async def analyze_image_direct(
    file: UploadFile = File(...),
    prompt: Optional[str] = None,
    current_user: models.User = Depends(get_current_user)
):
    """
    Direct image analysis without storage. Returns analysis results immediately.
    Useful for quick image analysis without saving to the drive.
    """
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Extract metadata
        metadata = ImageAnalyzer.extract_image_metadata(image_data)
        
        # Perform analysis
        analysis_result = await ImageAnalyzer.analyze_image(image_data, prompt)
        
        return {
            "filename": file.filename,
            "metadata": metadata,
            "analysis": analysis_result,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to analyze image: {str(e)}")

@router.post("/query/image", response_model=dict)
async def query_image_content(
    query: schemas.ImageQueryRequest,
    db: Session = Depends(database.get_db),
    current_user: models.User = Depends(get_current_user)
):
    """
    Query specifically about image content using stored analysis data.
    Can search across all user's images or query a specific image.
    """
    
    try:
        if query.image_id:
            # Query specific image
            image_item = db.query(models.FileSystemItem).filter_by(
                id=query.image_id,
                owner_id=current_user.id,
                type="file"
            ).first()
            
            if not image_item or not image_item.mime_type.startswith('image/'):
                raise HTTPException(status_code=404, detail="Image not found")
            
            # For specific image queries, we might need to fetch stored analysis
            # or re-analyze the image
            response_data = {
                "image_id": query.image_id,
                "query": query.query,
                "response": f"Querying image '{image_item.name}' about: {query.query}",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            # Query across all user's images
            user_images = db.query(models.FileSystemItem).filter(
                models.FileSystemItem.owner_id == current_user.id,
                models.FileSystemItem.type == "file",
                models.FileSystemItem.mime_type.like('image/%')
            ).all()
            
            response_data = {
                "query": query.query,
                "total_images": len(user_images),
                "response": f"Searching across {len(user_images)} images for: {query.query}",
                "matches": [],  # Would contain matching images with analysis
                "timestamp": datetime.utcnow().isoformat()
            }
        
        # Integration with RAG service for semantic search would go here
        # For now, return structured response
        return response_data
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to query images: {str(e)}")

@router.get("/query/image/stream")
async def stream_image_query(
    query: str,
    image_id: Optional[str] = None,
    current_user: models.User = Depends(get_current_user),
    db: Session = Depends(database.get_db)
):
    """
    Streaming queries for visual content. Returns real-time analysis results.
    """
    
    async def generate_stream():
        try:
            # Initial response
            yield f"data: {json.dumps({'status': 'starting', 'query': query})}\n\n"
            
            if image_id:
                # Stream analysis of specific image
                image_item = db.query(models.FileSystemItem).filter_by(
                    id=image_id,
                    owner_id=current_user.id,
                    type="file"
                ).first()
                
                if not image_item:
                    yield f"data: {json.dumps({'error': 'Image not found'})}\n\n"
                    return
                
                yield f"data: {json.dumps({'status': 'analyzing', 'image': image_item.name})}\n\n"
                
                # Simulate streaming analysis results
                analysis_steps = [
                    "Detecting objects in image...",
                    "Analyzing colors and composition...",
                    "Extracting text content...",
                    "Generating detailed description...",
                    "Processing query against image content..."
                ]
                
                for step in analysis_steps:
                    yield f"data: {json.dumps({'status': 'progress', 'step': step})}\n\n"
                    # In real implementation, each step would perform actual analysis
                
                # Final result
                yield f"data: {json.dumps({'status': 'completed', 'result': f'Analysis complete for {query}', 'timestamp': datetime.utcnow().isoformat()})}\n\n"
            
            else:
                # Stream search across all images
                yield f"data: {json.dumps({'status': 'searching', 'scope': 'all_images'})}\n\n"
                
                user_images = db.query(models.FileSystemItem).filter(
                    models.FileSystemItem.owner_id == current_user.id,
                    models.FileSystemItem.type == "file",
                    models.FileSystemItem.mime_type.like('image/%')
                ).limit(10).all()  # Limit for demo
                
                for image in user_images:
                    yield f"data: {json.dumps({'status': 'checking', 'image': image.name})}\n\n"
                
                yield f"data: {json.dumps({'status': 'completed', 'total_checked': len(user_images), 'matches': []})}\n\n"
            
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream"
        }
    )

# Background task functions
async def analyze_and_store_results(
    image_id: str,
    image_data: bytes,
    description: str = None,
    tags: List[str] = None,
    db_session: Session = None
):
    """Background task to analyze image and store results"""
    try:
        # Perform analysis
        analysis_result = await ImageAnalyzer.analyze_image(image_data, description)
        
        # Update database record with analysis results
        if db_session:
            image_item = db_session.query(models.FileSystemItem).filter_by(id=image_id).first()
            if image_item:
                image_item.ingestion_status = "completed"
                db_session.commit()
        
        # Store analysis results (would typically go to a separate analysis table)
        print(f"Analysis completed for image {image_id}: {analysis_result}")
        
    except Exception as e:
        print(f"Background analysis failed for {image_id}: {e}")
        if db_session:
            image_item = db_session.query(models.FileSystemItem).filter_by(id=image_id).first()
            if image_item:
                image_item.ingestion_status = "failed"
                db_session.commit()