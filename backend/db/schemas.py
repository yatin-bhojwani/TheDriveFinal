from pydantic import BaseModel, EmailStr
from typing import List, Optional, Literal
import datetime

class UserBase(BaseModel):
    email: EmailStr
class UserCreate(UserBase):
    password: str
class User(UserBase):
    id: int
    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str

class FileSystemItemBase(BaseModel):
    id: str
    name: str
    type: Literal['folder', 'file']
    fileType: Optional[str] = None  # Added mime_type
    size: Optional[int] = None 
    ingestion_status: Optional[Literal['pending', 'processing', 'completed', 'failed']] = None
    class Config:
        from_attributes = True
class FileSystemItem(FileSystemItemBase):
    pass

class FolderCreate(BaseModel):
    name: str
    parentId: str
class ItemUpdate(BaseModel):
    name: str

class ChatQuery(BaseModel):
    query: str
    context: dict
class ChatResponse(BaseModel):
    answer: str
    sources: List[dict]

class FolderResponse(BaseModel):
    items: List[FileSystemItem]
    path: List[dict]   

class ViewLinkResponse(BaseModel):
    url: str

class SearchFilters(BaseModel):
    query: Optional[str] = None
    file_type: Optional[str] = None  # e.g., 'pdf', 'docx', 'image'
    mime_type: Optional[str] = None
    item_type: Optional[Literal['file', 'folder']] = None
    ingestion_status: Optional[Literal['pending', 'processing', 'completed', 'failed']] = None
    date_from: Optional[str] = None  # ISO date string
    date_to: Optional[str] = None    # ISO date string
    min_size: Optional[int] = None   # bytes
    max_size: Optional[int] = None   # bytes

class ImageQueryRequest(BaseModel):
    query: str
    image_id: Optional[str] = None  # If provided, query specific image
    max_results: Optional[int] = 10
    include_metadata: bool = True

class ImageAnalysisResponse(BaseModel):
    analysis: str
    objects: List[dict] = []
    text: str = ""
    colors: List[str] = []
    metadata: dict = {}
    timestamp: str

class ImageIngestResponse(BaseModel):
    id: str
    filename: str
    size: int
    metadata: dict
    s3_key: str
    status: str
    message: str