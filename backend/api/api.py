from fastapi import APIRouter
from .endpoints import auth, files, chat

api_router = APIRouter()
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(files.router, prefix="/drive", tags=["Drive Operations"])
api_router.include_router(chat.router, prefix="/chat", tags=["AI Chat"])