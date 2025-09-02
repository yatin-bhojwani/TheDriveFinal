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