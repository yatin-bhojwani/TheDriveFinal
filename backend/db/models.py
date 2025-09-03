from sqlalchemy import Column, Integer, String, ForeignKey, Enum, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from .database import Base

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    items = relationship("FileSystemItem", back_populates="owner")

class FileSystemItem(Base):
    __tablename__ = "filesystem_items"
    id = Column(String, primary_key=True, index=True)
    name = Column(String, index=True)
    type = Column(Enum("file", "folder", name="item_type_enum"), nullable=False)
    s3_key = Column(String, unique=True, nullable=True)
    mime_type = Column(String, nullable=True)
    size_bytes = Column(Integer, nullable=True)
    ingestion_status = Column(Enum("pending", "processing", "completed", "failed", name="ingestion_status_enum"), nullable=True, default="pending")
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now(), nullable=False)
    
    owner_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    parent_id = Column(String, ForeignKey("filesystem_items.id"), nullable=True)
    
    owner = relationship("User", back_populates="items")
    children = relationship("FileSystemItem", backref="parent", remote_side=[id], cascade="all, delete-orphan",single_parent=True)