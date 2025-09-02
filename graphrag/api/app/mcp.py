from typing import Any, Dict
from pydantic import BaseModel
from fastapi import APIRouter, HTTPException
from .rag import rag_answer, get_vs

router = APIRouter(prefix="/mcp", tags=["mcp"])

class RpcRequest(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int
    method: str
    params: Dict[str, Any] | None = None

class RpcResponse(BaseModel):
    jsonrpc: str = "2.0"
    id: str | int
    result: Any | None = None
    error: Dict[str, Any] | None = None

def tool_chat_answer(params: Dict[str, Any]) -> Dict[str, Any]:
    q = params.get("query")
    scope = params.get("scope", "drive")
    folder_id = params.get("folder_id")
    file_id = params.get("file_id")
    if not q:
        raise HTTPException(400, "query is required")
    return rag_answer(q, scope=scope, folder_id=folder_id, file_id=file_id)

def tool_search_semantic(params: Dict[str, Any]) -> Dict[str, Any]:
    q = params.get("query")
    scope = params.get("scope", "drive")
    folder_id = params.get("folder_id")
    file_id = params.get("file_id")
    k = int(params.get("k", 10))
    if not q:
        raise HTTPException(400, "query is required")

    vs = get_vs()
    where = {}
    if scope == "folder" and folder_id:
        where["folder_id"] = folder_id
    if scope == "file" and file_id:
        where["file_id"] = file_id

    try:
        docs = vs.max_marginal_relevance_search(q, k=k, fetch_k=max(20, k*3), filter=where or None)
    except Exception:
        docs = vs.similarity_search(q, k=k, filter=where or None)

    return {
        "hits": [
            {
                "text": d.page_content,
                "file_id": d.metadata.get("file_id"),
                "chunk_no": d.metadata.get("chunk_no"),
                "source": d.metadata.get("source"),
            } for d in docs
        ]
    }

def tool_highlight_text(params: Dict[str, Any]) -> Dict[str, Any]:
    text = params.get("text")
    scope = params.get("scope", "file")
    file_id = params.get("file_id")
    if not text:
        raise HTTPException(400, "text is required")
    return rag_answer(f"Explain briefly: {text}", scope=scope, file_id=file_id)

TOOLS = {
    "chat.answer": tool_chat_answer,
    "search.semantic": tool_search_semantic,
    "highlight.text": tool_highlight_text,
}

@router.post("")
def rpc(req: RpcRequest) -> RpcResponse:
    try:
        if req.method not in TOOLS:
            return RpcResponse(id=req.id, error={"code": -32601, "message": "Method not found"})
        result = TOOLS[req.method](req.params or {})
        return RpcResponse(id=req.id, result=result)
    except HTTPException as e:
        return RpcResponse(id=req.id, error={"code": e.status_code, "message": e.detail})
    except Exception as e:
        return RpcResponse(id=req.id, error={"code": -32000, "message": str(e)})
