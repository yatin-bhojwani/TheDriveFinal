from fastapi import APIRouter
from db import schemas

router = APIRouter()

@router.post("/query", response_model=schemas.ChatResponse)
def handle_chat_query(query: schemas.ChatQuery):
    print(f"Received query: '{query.query}' for context: {query.context}")
    return {
        "answer": f"This is a mocked response for your question about '{query.query}'. The budget was $150,000.",
        "sources": [
            {"id": "file-1", "name": "Q3_Budget.xlsx", "relevance": 0.98}
        ]
    }