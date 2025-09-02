from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from jinja2 import TemplateNotFound
from typing import AsyncGenerator, List, Optional, Dict
import json
import os

# ========== Request Models ==========

class BulkDeleteRequest(BaseModel):
    file_ids: List[str] = Field(..., description="List of file IDs to delete")

from .rag import (
    ingest_pdf, ingest_file, rag_answer, build_prompt_and_citations, stream_gemini, 
    get_graph_stats, clear_graph, cleanup_low_confidence_entities, 
    update_confidence_thresholds, get_graphrag_config,
    clear_cache_collection, clear_main_collection, clear_all_collections, get_chroma_stats,
    detect_file_type, conversation_aware_rag
)
from .mcp import router as mcp_router
from .delt import (
    delete_file_completely, delete_folder_completely,
    get_file_deletion_preview, get_folder_deletion_preview
)
from .bulkdel import (
    delete_multiple_files, get_orphaned_data_preview
)

# GraphRAG will be accessed through rag module to avoid circular imports
GRAPHRAG_AVAILABLE = os.getenv("ENABLE_GRAPHRAG", "0") == "1"

from dataclasses import dataclass
from datetime import datetime, timedelta
from uuid import uuid4
import json

app = FastAPI(title="TheDrive (Local RAG)")

# Add CORS middleware for SSE support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, be more restrictive
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(mcp_router)

BASE_DIR = os.path.dirname(__file__)
from fastapi.templating import Jinja2Templates
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))


# ========== Conversation Models ==========

@dataclass
class ConversationTurn:
    query: str
    answer: str
    timestamp: datetime
    scope: str = "drive"
    file_id: Optional[str] = None
    folder_id: Optional[str] = None
    confidence: float = 0.0
    citations: List[Dict] = None
    
    def __post_init__(self):
        if self.citations is None:
            self.citations = []
    
    def dict(self):
        return {
            "query": self.query,
            "answer": self.answer,
            "timestamp": self.timestamp.isoformat(),
            "scope": self.scope,
            "file_id": self.file_id,
            "folder_id": self.folder_id,
            "confidence": self.confidence,
            "citations": self.citations
        }


class ConversationRequest(BaseModel):
    query: str
    conversation_history: List[Dict] = Field(default_factory=list)
    scope: str = "drive"
    folder_id: Optional[str] = None
    file_id: Optional[str] = None
    k: int = 10


# Conversation Memory Store (in production, use Redis or database)
conversation_store: Dict[str, List[ConversationTurn]] = {}

class ChatRequest(BaseModel):
    query: str
    scope: str = "drive"          # drive | folder | file
    folder_id: str | None = None
    file_id: str | None = None
    k: int = 10

@app.get("/health")
def health():
    return {"ok": True}

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    try:
        return templates.TemplateResponse("index.html", {"request": request})
    except TemplateNotFound:
        return HTMLResponse("<h1>TheDrive API</h1><p>Templates missing.</p>")

@app.post("/ingest/pdf")
async def ingest_pdf_endpoint(file: UploadFile, file_id: str = Form(...), folder_id: str | None = Form(None)):
    """Legacy PDF ingestion endpoint for backward compatibility."""
    path = f"/workspace/data/{file.filename}"
    with open(path, "wb") as f:
        f.write(await file.read())
    try:
        chunks = ingest_pdf(path, file_id=file_id, folder_id=folder_id)
        return {"indexed_chunks": chunks, "file_id": file_id, "folder_id": folder_id, "file_type": "pdf"}
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ingest/file")
async def ingest_file_endpoint(file: UploadFile, file_id: str = Form(...), folder_id: str | None = Form(None)):
    """Universal file ingestion endpoint supporting PDF, TXT, DOCX, CSV."""
    # Create data directory if it doesn't exist
    os.makedirs("/workspace/data", exist_ok=True)
    
    path = f"/workspace/data/{file.filename}"
    
    # Save uploaded file
    with open(path, "wb") as f:
        f.write(await file.read())
    
    try:
        # Auto-detect file type
        file_type = detect_file_type(path, file.content_type)
        
        # Ingest file
        chunks = ingest_file(path, file_id=file_id, folder_id=folder_id, file_type=file_type)
        
        return {
            "indexed_chunks": chunks, 
            "file_id": file_id, 
            "folder_id": folder_id,
            "file_type": file_type,
            "filename": file.filename,
            "content_type": file.content_type
        }
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ingest/text")
async def ingest_text_endpoint(file: UploadFile, file_id: str = Form(...), folder_id: str | None = Form(None)):
    """Text file ingestion endpoint (.txt, .md, .rst)."""
    os.makedirs("/workspace/data", exist_ok=True)
    path = f"/workspace/data/{file.filename}"
    
    with open(path, "wb") as f:
        f.write(await file.read())
    
    try:
        chunks = ingest_file(path, file_id=file_id, folder_id=folder_id, file_type="text")
        return {"indexed_chunks": chunks, "file_id": file_id, "folder_id": folder_id, "file_type": "text"}
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ingest/docx")
async def ingest_docx_endpoint(file: UploadFile, file_id: str = Form(...), folder_id: str | None = Form(None)):
    """DOCX file ingestion endpoint."""
    os.makedirs("/workspace/data", exist_ok=True)
    path = f"/workspace/data/{file.filename}"
    
    with open(path, "wb") as f:
        f.write(await file.read())
    
    try:
        chunks = ingest_file(path, file_id=file_id, folder_id=folder_id, file_type="docx")
        return {"indexed_chunks": chunks, "file_id": file_id, "folder_id": folder_id, "file_type": "docx"}
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/ingest/csv")
async def ingest_csv_endpoint(file: UploadFile, file_id: str = Form(...), folder_id: str | None = Form(None)):
    """CSV file ingestion endpoint."""
    os.makedirs("/workspace/data", exist_ok=True)
    path = f"/workspace/data/{file.filename}"
    
    with open(path, "wb") as f:
        f.write(await file.read())
    
    try:
        chunks = ingest_file(path, file_id=file_id, folder_id=folder_id, file_type="csv")
        return {"indexed_chunks": chunks, "file_id": file_id, "folder_id": folder_id, "file_type": "csv"}
    except Exception as e:
        import traceback, sys
        traceback.print_exc(file=sys.stderr)
        return JSONResponse({"error": str(e)}, status_code=500)


# ========== Conversation-Aware Chat Endpoints ==========

@app.post("/chat/conversation")
async def chat_conversation_aware(request: ConversationRequest):
    """
    Chat endpoint with conversation awareness and follow-up detection.
    """
    try:
        result = conversation_aware_rag(
            query=request.query,
            conversation_history=request.conversation_history,
            scope=request.scope,
            folder_id=request.folder_id,
            file_id=request.file_id,
            k=request.k
        )
        return {"data": result}
    except Exception as e:
        print(f"Error in conversation chat: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.post("/chat/conversation/{session_id}")
async def chat_with_session(session_id: str, request: ChatRequest):
    """
    Chat endpoint that maintains conversation history by session ID.
    """
    try:
        # Get conversation history for this session
        conversation_history = conversation_store.get(session_id, [])
        
        # Convert to dict format for rag function
        history_dicts = [turn.dict() for turn in conversation_history]
        
        result = conversation_aware_rag(
            query=request.query,
            conversation_history=history_dicts,
            scope=request.scope,
            folder_id=request.folder_id,
            file_id=request.file_id,
            k=request.k
        )
        
        # Store this turn in conversation history
        new_turn = ConversationTurn(
            query=request.query,
            answer=result["answer"],
            timestamp=datetime.now(),
            scope=result.get("final_scope", {}).get("scope", request.scope),
            file_id=result.get("final_scope", {}).get("file_id"),
            folder_id=result.get("final_scope", {}).get("folder_id"),
            confidence=result.get("confidence", 0.0),
            citations=result.get("citations", [])
        )
        
        # Maintain conversation history (keep last 10 turns)
        if session_id not in conversation_store:
            conversation_store[session_id] = []
        conversation_store[session_id].append(new_turn)
        if len(conversation_store[session_id]) > 10:
            conversation_store[session_id] = conversation_store[session_id][-10:]
        
        return {"data": result}
    except Exception as e:
        print(f"Error in session chat: {str(e)}")
        return JSONResponse({"error": str(e)}, status_code=500)


@app.get("/chat/sessions/{session_id}/history")
async def get_conversation_history(session_id: str):
    """
    Get conversation history for a session.
    """
    history = conversation_store.get(session_id, [])
    return {"session_id": session_id, "history": [turn.dict() for turn in history]}


@app.delete("/chat/sessions/{session_id}")
async def clear_conversation_session(session_id: str):
    """
    Clear conversation history for a session.
    """
    if session_id in conversation_store:
        del conversation_store[session_id]
        return {"message": f"Cleared conversation history for session {session_id}"}
    return {"message": f"No conversation history found for session {session_id}"}


@app.get("/chat/sessions")
async def list_conversation_sessions():
    """
    List all active conversation sessions.
    """
    sessions = []
    for session_id, history in conversation_store.items():
        sessions.append({
            "session_id": session_id,
            "turn_count": len(history),
            "last_activity": history[-1].timestamp if history else None
        })
    return {"sessions": sessions}


@app.get("/chat/conversation/{session_id}/stream")
async def chat_conversation_stream(
    request: Request,
    session_id: str,
    query: str,
    scope: str = "drive",
    folder_id: str | None = None,
    file_id: str | None = None,
    k: int = 10
):
    """
    Streaming conversation-aware chat with session management.
    """
    async def event_gen() -> AsyncGenerator[str, None]:
        try:
            print(f"[ConversationStream] Starting for session: {session_id}, query: {query}")
            yield 'event: status\ndata: {"msg":"analyzing conversation context"}\n\n'
            
            # Get conversation history for this session
            conversation_history = conversation_store.get(session_id, [])
            history_dicts = [turn.dict() for turn in conversation_history]
            print(f"[ConversationStream] Found {len(conversation_history)} previous turns")
            
            yield 'event: status\ndata: {"msg":"detecting follow-up patterns"}\n\n'
            
            # Store original values for comparison
            original_scope, original_file_id, original_folder_id = scope, file_id, folder_id
            
            # Use conversation-aware RAG instead of the regular one
            print("[ConversationStream] Using conversation-aware RAG...")
            from .rag import conversation_aware_rag
            
            # Get the RAG result with conversation awareness
            rag_result = conversation_aware_rag(
                query=query,
                conversation_history=history_dicts,
                scope=scope,
                folder_id=folder_id,
                file_id=file_id,
                k=k
            )
            
            # Extract components from the result
            followup_analysis = rag_result.get("conversation_analysis", {})
            scope_adjusted = rag_result.get("scope_adjusted", False)
            final_scope = rag_result.get("final_scope", {"scope": scope, "file_id": file_id, "folder_id": folder_id})
            
            # Update scope variables from the result
            final_scope_name = final_scope.get("scope", scope)
            final_file_id = final_scope.get("file_id", file_id)
            final_folder_id = final_scope.get("folder_id", folder_id)
            
            print(f"[ConversationStream] Follow-up analysis: {followup_analysis}")
            print(f"[ConversationStream] Scope adjusted: {scope_adjusted}")
            
            # Send follow-up analysis as debug info
            yield f'event: followup\ndata: {json.dumps(followup_analysis)}\n\n'
            
            if scope_adjusted:
                yield f'event: scope_adjusted\ndata: {json.dumps({"original": {"scope": original_scope, "file_id": original_file_id, "folder_id": original_folder_id}, "adjusted": {"scope": final_scope_name, "file_id": final_file_id, "folder_id": final_folder_id}})}\n\n'
            
            yield 'event: status\ndata: {"msg":"retrieving context"}\n\n'
            
            if followup_analysis.get("is_followup", False):
                yield 'event: enhanced_query\ndata: {"msg":"Enhanced query with conversation context"}\n\n'
            
            # Build the full prompt using the existing infrastructure
            from .rag import build_prompt_and_citations, get_conversation_context
            citations = rag_result.get("citations", [])
            rewritten = rag_result.get("rewritten_query", query)
            confidence = rag_result.get("confidence", 0.0)
            
            # Build the prompt with proper context
            prompt, prompt_citations, prompt_rewritten, graphrag_info = build_prompt_and_citations(
                query=query, scope=final_scope_name, folder_id=final_folder_id, file_id=final_file_id, k=k
            )
            
            # If this is a follow-up, enhance the prompt with conversation context
            if followup_analysis.get("is_followup", False) and history_dicts:
                conversation_context = get_conversation_context(history_dicts, 800)
                
                # Add conversation awareness to the system prompt
                enhanced_system_part = "\n\nIMPORTANT: This is a follow-up question to a previous conversation. Use the conversation context provided to maintain consistency and avoid repeating information already discussed."
                conversation_prompt_context = f"\n\n# Previous Conversation Context\n{conversation_context}\n"
                
                # Insert the enhanced instructions and context
                prompt = prompt.replace("# Question", enhanced_system_part + conversation_prompt_context + "\n\n# Question")
            
            # Use the most comprehensive citations available
            citations = prompt_citations or citations
            
            # Send metadata including conversation info
            meta = {
                "citations": citations, 
                "rewritten_query": rewritten, 
                "conversation_analysis": followup_analysis,
                "scope_adjusted": scope_adjusted,
                "final_scope": final_scope,
                "graphrag_info": graphrag_info,
                "session_id": session_id,
                "conversation_turns": len(conversation_history),
                "confidence": confidence
            }
            yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

            yield 'event: status\ndata: {"msg":"generating response"}\n\n'
            
            # Stream the response
            from .rag import stream_gemini
            full_answer = ""
            async for tok in stream_gemini(prompt):
                if await request.is_disconnected():
                    break
                yield f'event: token\ndata: {json.dumps({"t": tok})}\n\n'
                full_answer += tok
            
            # Store this turn in conversation history
            new_turn = ConversationTurn(
                query=query,
                answer=full_answer,
                timestamp=datetime.now(),
                scope=final_scope_name,
                file_id=final_file_id,
                folder_id=final_folder_id,
                confidence=confidence,
                citations=citations
            )
            
            # Maintain conversation history (keep last 10 turns)
            if session_id not in conversation_store:
                conversation_store[session_id] = []
            conversation_store[session_id].append(new_turn)
            if len(conversation_store[session_id]) > 10:
                conversation_store[session_id] = conversation_store[session_id][-10:]
            
            yield 'event: conversation_updated\ndata: {"msg":"Conversation history updated"}\n\n'
            yield 'event: done\ndata: {"msg":"complete"}\n\n'
            
        except Exception as e:
            print(f"[ConversationStream] ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            yield f'event: error\ndata: {json.dumps({"error": str(e)})}\n\n'

    return StreamingResponse(
        event_gen(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@app.get("/test-sse")
async def test_sse():
    """Simple SSE test endpoint"""
    import asyncio
    async def simple_gen():
        for i in range(5):
            yield f'event: test\ndata: {{"msg":"test message {i}"}}\n\n'
            await asyncio.sleep(0.5)
        yield 'event: done\ndata: {"msg":"test complete"}\n\n'
    
    return StreamingResponse(simple_gen(), media_type="text/event-stream")


# ========== Original Chat Endpoints ==========

@app.post("/chat")
async def chat(req: ChatRequest):
    """Original chat endpoint (non-conversation aware)"""
    result = rag_answer(req.query, scope=req.scope, folder_id=req.folder_id, file_id=req.file_id, k=req.k)
    return JSONResponse(result)

#(SSE) 
@app.get("/chat/stream")
async def chat_stream(request: Request,
                      query: str,
                      scope: str = "drive",
                      folder_id: str | None = None,
                      file_id: str | None = None,
                      k: int = 10):
    """Original streaming chat endpoint (non-conversation aware)"""
    async def event_gen() -> AsyncGenerator[str, None]:
        yield 'event: status\ndata: {"msg":"retrieving context"}\n\n'
        prompt, citations, rewritten, graphrag_info = build_prompt_and_citations(
            query=query, scope=scope, folder_id=folder_id, file_id=file_id, k=k
        )
        # MODIFICATION: Add the full prompt and GraphRAG info to the metadata for debugging
        meta = {
            "citations": citations, 
            "rewritten_query": rewritten, 
            "DEBUG_PROMPT": prompt,
            "graphrag_info": graphrag_info
        }
        yield f"event: meta\ndata: {json.dumps(meta)}\n\n"

        yield 'event: status\ndata: {"msg":"generating"}\n\n'
        async for tok in stream_gemini(prompt):
            if await request.is_disconnected():
                break
            yield f"event: token\ndata: {json.dumps({'t': tok})}\n\n"

        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(
        event_gen(), 
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

# ========== GraphRAG Management Endpoints ==========

@app.get("/graphrag/stats")
async def graphrag_stats():
    """Get GraphRAG statistics including confidence metrics."""
    if not GRAPHRAG_AVAILABLE:
        return JSONResponse({"error": "GraphRAG not available"}, status_code=503)
    
    try:
        stats = get_graph_stats()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/graphrag/clear")
async def graphrag_clear():
    """Clear the GraphRAG knowledge graph."""
    if not GRAPHRAG_AVAILABLE:
        return JSONResponse({"error": "GraphRAG not available"}, status_code=503)
    
    try:
        success = clear_graph()
        if success:
            return JSONResponse({"message": "GraphRAG knowledge graph cleared successfully"})
        else:
            return JSONResponse({"error": "Failed to clear graph"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/graphrag/cleanup")
async def graphrag_cleanup(confidence_threshold: float = None):
    """Clean up low-confidence entities and relations from the graph."""
    if not GRAPHRAG_AVAILABLE:
        return JSONResponse({"error": "GraphRAG not available"}, status_code=503)
    
    try:
        result = cleanup_low_confidence_entities(confidence_threshold)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/graphrag/thresholds")
async def update_graphrag_thresholds(entity_threshold: float = None, relation_threshold: float = None):
    """Update confidence thresholds for entity and relation filtering."""
    if not GRAPHRAG_AVAILABLE:
        return JSONResponse({"error": "GraphRAG not available"}, status_code=503)
    
    try:
        result = update_confidence_thresholds(entity_threshold, relation_threshold)
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/graphrag/cleanup-orphaned")
async def cleanup_orphaned_entities_endpoint():
    """Clean up entities that have no mention relationships from any chunks."""
    if not GRAPHRAG_AVAILABLE:
        return JSONResponse({"error": "GraphRAG not available"}, status_code=503)
    
    try:
        from .delt import cleanup_orphaned_entities
        result = cleanup_orphaned_entities()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/graphrag/config")
async def get_graphrag_config_endpoint():
    """Get current GraphRAG configuration including confidence thresholds."""
    if not GRAPHRAG_AVAILABLE:
        return JSONResponse({"error": "GraphRAG not available"}, status_code=503)
    
    try:
        config = get_graphrag_config()
        return JSONResponse(config)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ========== Chroma Management Endpoints ==========

@app.get("/chroma/stats")
async def chroma_stats():
    """Get Chroma database statistics."""
    try:
        stats = get_chroma_stats()
        return JSONResponse(stats)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chroma/clear")
async def chroma_clear_all():
    """Clear all Chroma collections (main and cache)."""
    try:
        success = clear_all_collections()
        if success:
            return JSONResponse({"message": "All Chroma collections cleared successfully"})
        else:
            return JSONResponse({"error": "Failed to clear all collections"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chroma/clear/main")
async def chroma_clear_main():
    """Clear the main Chroma collection containing documents."""
    try:
        success = clear_main_collection()
        if success:
            return JSONResponse({"message": "Main Chroma collection cleared successfully"})
        else:
            return JSONResponse({"error": "Failed to clear main collection"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chroma/clear/cache")
async def chroma_clear_cache():
    """Clear the Chroma cache collection."""
    try:
        success = clear_cache_collection()
        if success:
            return JSONResponse({"message": "Chroma cache collection cleared successfully"})
        else:
            return JSONResponse({"error": "Failed to clear cache collection"}, status_code=500)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ========== File and Folder Deletion Endpoints ==========

@app.delete("/files/{file_id}")
async def delete_file_endpoint(file_id: str):
    """
    Delete a specific file and all its associated data from both vector database and knowledge graph.
    
    Args:
        file_id: The unique identifier of the file to delete
    
    Returns:
        Comprehensive deletion results including success status and details
    """
    try:
        result = delete_file_completely(file_id)
        
        if result["overall_success"]:
            return JSONResponse({
                "success": True,
                "message": f"File {file_id} deleted successfully",
                "details": result
            })
        else:
            return JSONResponse({
                "success": False,
                "message": f"Failed to delete file {file_id} or file not found",
                "details": result
            }, status_code=404)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "file_id": file_id
        }, status_code=500)


@app.delete("/folders/{folder_id}")
async def delete_folder_endpoint(folder_id: str):
    """
    Delete a specific folder and all files within it from both vector database and knowledge graph.
    
    Args:
        folder_id: The unique identifier of the folder to delete
    
    Returns:
        Comprehensive deletion results including success status and details
    """
    try:
        result = delete_folder_completely(folder_id)
        
        if result["overall_success"]:
            return JSONResponse({
                "success": True,
                "message": f"Folder {folder_id} and all its files deleted successfully",
                "details": result
            })
        else:
            return JSONResponse({
                "success": False,
                "message": f"Failed to delete folder {folder_id} or folder not found",
                "details": result
            }, status_code=404)
            
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "folder_id": folder_id
        }, status_code=500)


@app.get("/files/{file_id}/deletion-preview")
async def preview_file_deletion(file_id: str):
    """
    Preview what would be deleted for a specific file without actually performing the deletion.
    
    Args:
        file_id: The unique identifier of the file to preview
    
    Returns:
        Preview information showing what would be deleted
    """
    try:
        preview = get_file_deletion_preview(file_id)
        return JSONResponse({
            "success": True,
            "preview": preview
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "file_id": file_id
        }, status_code=500)


@app.get("/folders/{folder_id}/deletion-preview")
async def preview_folder_deletion(folder_id: str):
    """
    Preview what would be deleted for a specific folder without actually performing the deletion.
    
    Args:
        folder_id: The unique identifier of the folder to preview
    
    Returns:
        Preview information showing what would be deleted
    """
    try:
        preview = get_folder_deletion_preview(folder_id)
        return JSONResponse({
            "success": True,
            "preview": preview
        })
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e),
            "folder_id": folder_id
        }, status_code=500)


@app.post("/files/delete-multiple")
async def delete_multiple_files_endpoint(request: BulkDeleteRequest):
    """
    Delete multiple files at once.
    
    Request body should be a JSON object with file_ids array: {"file_ids": ["file1", "file2", "file3"]}
    
    Returns:
        Bulk deletion results with summary and individual results
    """
    if not request.file_ids:
        return JSONResponse({
            "success": False,
            "error": "No file IDs provided"
        }, status_code=400)
    
    try:
        result = delete_multiple_files(request.file_ids)
        
        return JSONResponse({
            "success": result["successful_deletions"] > 0,
            "message": f"Processed {result['total_files']} files. {result['successful_deletions']} successful, {result['failed_deletions']} failed.",
            "summary": {
                "total_files": result["total_files"],
                "successful_deletions": result["successful_deletions"],
                "failed_deletions": result["failed_deletions"]
            },
            "details": result["results"]
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


@app.get("/admin/orphaned-data")
async def get_orphaned_data():
    """
    Get information about data that might be orphaned in the databases.
    This helps identify data that exists in the vector DB or knowledge graph
    but may no longer have corresponding files.
    
    Returns:
        Information about orphaned data across all databases
    """
    try:
        orphaned_data = get_orphaned_data_preview()
        
        return JSONResponse({
            "success": True,
            "message": "Orphaned data analysis completed",
            "data": orphaned_data
        })
        
    except Exception as e:
        return JSONResponse({
            "success": False,
            "error": str(e)
        }, status_code=500)


# ========== Combined Clear Endpoint ==========

@app.post("/clear-all")
async def clear_all_databases():
    """Clear both GraphRAG (Neo4j) and Chroma databases completely."""
    results = {
        "graphrag": {"success": False, "message": ""},
        "chroma": {"success": False, "message": ""}
    }
    
    # Clear GraphRAG
    if GRAPHRAG_AVAILABLE:
        try:
            graphrag_success = clear_graph()
            results["graphrag"]["success"] = graphrag_success
            results["graphrag"]["message"] = "GraphRAG cleared successfully" if graphrag_success else "Failed to clear GraphRAG"
        except Exception as e:
            results["graphrag"]["message"] = f"GraphRAG error: {str(e)}"
    else:
        results["graphrag"]["message"] = "GraphRAG not available"
    
    # Clear Chroma
    try:
        chroma_success = clear_all_collections()
        results["chroma"]["success"] = chroma_success
        results["chroma"]["message"] = "Chroma collections cleared successfully" if chroma_success else "Failed to clear Chroma collections"
    except Exception as e:
        results["chroma"]["message"] = f"Chroma error: {str(e)}"
    
    # Determine overall success
    overall_success = results["chroma"]["success"] and (results["graphrag"]["success"] or not GRAPHRAG_AVAILABLE)
    
    return JSONResponse({
        "success": overall_success,
        "message": "Database clearing completed",
        "details": results
    })
