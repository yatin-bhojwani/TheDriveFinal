from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from langchain_community.document_loaders.pdf import PyPDFLoader
from langchain_community.document_loaders import TextLoader, CSVLoader
from langchain_community.document_loaders.word_document import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.documents import Document
import os
import json
import httpx
import time
import hashlib
import re
import mimetypes
from pathlib import Path
from datetime import datetime, timedelta

from . import settings

# neo4j
_GR_ENABLED = os.getenv("ENABLE_GRAPHRAG", "0") == "1"
_GRAPHRAG_FUNCTIONS = None
_GRAPHRAG_MODULE = None 

def _get_graphrag_module():
    """Helper to robustly import the graphrag module using an absolute path."""
    global _GRAPHRAG_MODULE
    if _GRAPHRAG_MODULE is None and _GR_ENABLED:
        try:
            import importlib
           
            _GRAPHRAG_MODULE = importlib.import_module('app.graphrag')
        except ImportError as e:
            print(f"[GraphRAG] CRITICAL: Failed to import 'app.graphrag' module: {e}")
            
            _GRAPHRAG_MODULE = False
    return _GRAPHRAG_MODULE

def get_graphrag_functions():
    global _GRAPHRAG_FUNCTIONS
    if _GRAPHRAG_FUNCTIONS is None and _GR_ENABLED:
        graphrag_module = _get_graphrag_module()
        if graphrag_module:
            _GRAPHRAG_FUNCTIONS = {
                # Core extraction and storage functions
                'upsert_entities_and_relations': graphrag_module.upsert_entities_and_relations,
                'extract_query_entities': graphrag_module.extract_query_entities,
                'get_subgraph_facts': graphrag_module.get_subgraph_facts,
                
                # New confidence-based functions
                'cleanup_low_confidence_entities': graphrag_module.cleanup_low_confidence_entities,
                'update_confidence_thresholds': graphrag_module.update_confidence_thresholds,
                'get_graph_stats': graphrag_module.get_graph_stats,
                'clear_graph': graphrag_module.clear_graph,
                
                # Configuration access
                'get_entity_threshold': lambda: graphrag_module.ENTITY_CONFIDENCE_THRESHOLD,
                'get_relation_threshold': lambda: graphrag_module.RELATION_CONFIDENCE_THRESHOLD,
                'get_validation_enabled': lambda: graphrag_module.ENABLE_EMBEDDING_VALIDATION,
            }
            print("[GraphRAG] Successfully loaded GraphRAG functions (including confidence features)")
        else:
            print("[GraphRAG] GraphRAG functions not loaded due to import failure.")
            _GRAPHRAG_FUNCTIONS = False
    return _GRAPHRAG_FUNCTIONS


# models 

def embedder():
    return GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GEMINI_API_KEY
    )

def llm():
    """LLM for processing tasks (entity extraction, rewriting, etc.)"""
    return ChatGoogleGenerativeAI(
        model=settings.MODEL_NAME,
        google_api_key=settings.GEMINI_API_KEY,
        temperature=0.2
    )

def response_llm():
    """Separate LLM for final response generation"""
    return ChatGoogleGenerativeAI(
        model=settings.RESPONSE_MODEL,
        google_api_key=settings.RESPONSE_API_KEY,
        temperature=0.2
    )


# vector sttores

def _http_host_port():
    url = settings.CHROMA_URL.rstrip("/")
    url_parts = url.replace("http://", "").replace("https://", "")
    if ":" in url_parts:
        host, port_s = url_parts.split(":", 1)
        port = int(port_s)
    else:
        host, port = url_parts, 8000
    return host, port

def get_vs() -> Chroma:
    import chromadb
    host, port = _http_host_port()
    client = chromadb.HttpClient(host=host, port=port)
    return Chroma(
        collection_name=settings.COLLECTION,
        embedding_function=embedder(),
        client=client,
    )

def get_cache_vs() -> Chroma:
    import chromadb
    host, port = _http_host_port()
    client = chromadb.HttpClient(host=host, port=port)
    return Chroma(
        collection_name=f"{settings.COLLECTION}_cache",
        embedding_function=embedder(),
        client=client,
    )


# ingest

@dataclass
class ChunkMeta:
    file_id: str
    folder_id: Optional[str]
    page: Optional[int]
    chunk_no: int
    type: str = "unknown"
    source: str = ""

# File type detection
def detect_file_type(file_path: str, content_type: Optional[str] = None) -> str:
    """Detect file type from extension and MIME type."""
    path = Path(file_path)
    extension = path.suffix.lower()
    
    # Primary detection by extension
    if extension == '.pdf':
        return 'pdf'
    elif extension in ['.txt', '.md', '.rst']:
        return 'text'
    elif extension in ['.docx', '.doc']:
        return 'docx'
    elif extension == '.csv':
        return 'csv'
    
    # Fallback to MIME type
    if content_type:
        if 'pdf' in content_type:
            return 'pdf'
        elif 'text' in content_type:
            return 'text'
        elif 'spreadsheet' in content_type or 'csv' in content_type:
            return 'csv'
        elif 'word' in content_type or 'document' in content_type:
            return 'docx'
    
    # Final fallback
    return 'text'

# Specialized loaders for each file type
def load_pdf(path: str) -> List[Document]:
    """Load PDF using PyPDFLoader with enhanced metadata."""
    try:
        loader = PyPDFLoader(path)
        docs = loader.load()
        # Enhance metadata
        for doc in docs:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata['file_type'] = 'pdf'
            doc.metadata['loader'] = 'PyPDFLoader'
        return docs
    except Exception as e:
        print(f"[Ingest] Error loading PDF {path}: {e}")
        return []

def load_text(path: str) -> List[Document]:
    """Load text files with encoding detection."""
    try:
        # Try UTF-8 first
        encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    content = f.read()
                
                doc = Document(
                    page_content=content,
                    metadata={
                        'source': os.path.basename(path),
                        'file_type': 'text',
                        'encoding': encoding,
                        'loader': 'TextLoader'
                    }
                )
                return [doc]
            except UnicodeDecodeError:
                continue
        
        # If all encodings fail, try with errors='ignore'
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
        
        doc = Document(
            page_content=content,
            metadata={
                'source': os.path.basename(path),
                'file_type': 'text',
                'encoding': 'utf-8-ignore',
                'loader': 'TextLoader'
            }
        )
        return [doc]
        
    except Exception as e:
        print(f"[Ingest] Error loading text file {path}: {e}")
        return []

def load_docx(path: str) -> List[Document]:
    """Load DOCX files using Docx2txtLoader."""
    try:
        loader = Docx2txtLoader(path)
        docs = loader.load()
        # Enhance metadata
        for doc in docs:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata['file_type'] = 'docx'
            doc.metadata['loader'] = 'Docx2txtLoader'
        return docs
    except Exception as e:
        print(f"[Ingest] Error loading DOCX {path}: {e}")
        # Fallback: try to extract as zip and read document.xml
        try:
            import zipfile
            import xml.etree.ElementTree as ET
            
            with zipfile.ZipFile(path, 'r') as zip_file:
                try:
                    xml_content = zip_file.read('word/document.xml')
                    root = ET.fromstring(xml_content)
                    
                    # Extract text from XML (basic extraction)
                    text_content = ""
                    for text_elem in root.iter():
                        if text_elem.text:
                            text_content += text_elem.text + " "
                    
                    doc = Document(
                        page_content=text_content.strip(),
                        metadata={
                            'source': os.path.basename(path),
                            'file_type': 'docx',
                            'loader': 'ZipFallback'
                        }
                    )
                    return [doc]
                except:
                    pass
        except:
            pass
        return []

def load_csv(path: str) -> List[Document]:
    """Load CSV files with intelligent processing."""
    try:
        import pandas as pd
        
        # Try to read CSV with different parameters
        csv_params = [
            {'encoding': 'utf-8'},
            {'encoding': 'latin-1'},
            {'encoding': 'cp1252'},
            {'encoding': 'utf-8', 'sep': ';'},
            {'encoding': 'latin-1', 'sep': ';'},
        ]
        
        df = None
        used_params = None
        
        for params in csv_params:
            try:
                df = pd.read_csv(path, **params)
                used_params = params
                break
            except:
                continue
        
        if df is None:
            print(f"[Ingest] Could not read CSV file {path}")
            return []
        
        docs = []
        
        # Strategy 1: Convert entire CSV to text representation
        csv_text = f"CSV File: {os.path.basename(path)}\n"
        csv_text += f"Columns: {', '.join(df.columns.tolist())}\n"
        csv_text += f"Rows: {len(df)}\n\n"
        
        # Add column descriptions
        for col in df.columns:
            col_info = f"Column '{col}':\n"
            if df[col].dtype == 'object':
                unique_vals = df[col].dropna().unique()[:10]
                col_info += f"  Type: Text/Categorical\n"
                col_info += f"  Sample values: {', '.join(map(str, unique_vals))}\n"
            elif df[col].dtype in ['int64', 'float64']:
                col_info += f"  Type: Numeric\n"
                col_info += f"  Range: {df[col].min()} to {df[col].max()}\n"
                col_info += f"  Mean: {df[col].mean():.2f}\n"
            
            csv_text += col_info + "\n"
        
        # Add sample rows
        csv_text += "Sample rows:\n"
        csv_text += df.head(5).to_string(index=False)
        
        doc = Document(
            page_content=csv_text,
            metadata={
                'source': os.path.basename(path),
                'file_type': 'csv',
                'loader': 'PandasCSV',
                'columns': df.columns.tolist(),
                'rows': len(df),
                'encoding': used_params.get('encoding', 'utf-8')
            }
        )
        docs.append(doc)
        
        # Strategy 2: Create individual documents for each row (if reasonable size)
        if len(df) <= 1000:  # Only for smaller CSVs
            for idx, row in df.iterrows():
                row_text = f"Row {idx + 1} from {os.path.basename(path)}:\n"
                for col, val in row.items():
                    if pd.notna(val):
                        row_text += f"{col}: {val}\n"
                
                row_doc = Document(
                    page_content=row_text,
                    metadata={
                        'source': os.path.basename(path),
                        'file_type': 'csv',
                        'loader': 'PandasCSV',
                        'row_number': idx + 1,
                        'is_row_data': True
                    }
                )
                docs.append(row_doc)
        
        return docs
        
    except Exception as e:
        print(f"[Ingest] Error loading CSV {path}: {e}")
        # Fallback to basic text loading
        try:
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            doc = Document(
                page_content=content,
                metadata={
                    'source': os.path.basename(path),
                    'file_type': 'csv',
                    'loader': 'TextFallback'
                }
            )
            return [doc]
        except:
            return []

# Enhanced chunking with file-type awareness
def chunk_docs(docs: List[Document], file_type: str = "text", chunk_size: int = 700, overlap: int = 140) -> List[Document]:
    """Chunk documents with file-type specific strategies."""
    
    # Adjust chunking parameters based on file type
    if file_type == 'csv':
        # CSVs often need larger chunks to preserve row context
        chunk_size = 1500
        overlap = 200
    elif file_type == 'pdf':
        # PDFs might benefit from slightly larger chunks
        chunk_size = 800
        overlap = 160
    elif file_type == 'docx':
        # Word docs usually have structured content
        chunk_size = 900
        overlap = 180
    
    # Use appropriate separators based on file type
    if file_type == 'csv':
        separators = ["\n\n", "\n", ",", " "]
    elif file_type in ['pdf', 'docx']:
        separators = ["\n\n", "\n", ". ", " "]
    else:
        separators = ["\n\n", "\n", ". ", " "]
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=separators
    )
    
    return splitter.split_documents(docs)

# Universal file ingestion function
def ingest_file(path: str, file_id: str, folder_id: Optional[str] = None, file_type: Optional[str] = None) -> int:
    """
    Universal file ingestion supporting PDF, TXT, DOCX, CSV.
    Automatically detects file type and applies appropriate processing.
    """
    
    # Detect file type if not provided
    if file_type is None:
        file_type = detect_file_type(path)
    
    print(f"[Ingest] Processing {file_type.upper()} file: {path}")
    
    # Load documents based on file type
    if file_type == 'pdf':
        raw_docs = load_pdf(path)
    elif file_type == 'text':
        raw_docs = load_text(path)
    elif file_type == 'docx':
        raw_docs = load_docx(path)
    elif file_type == 'csv':
        raw_docs = load_csv(path)
    else:
        print(f"[Ingest] Unsupported file type: {file_type}")
        return 0
    
    if not raw_docs:
        print(f"[Ingest] No content extracted from {path}")
        return 0
    
    print(f"[Ingest] Extracted {len(raw_docs)} raw documents from {file_type.upper()}")
    
    # Chunk documents
    chunks = chunk_docs(raw_docs, file_type)
    print(f"[Ingest] Created {len(chunks)} chunks")
    
    # Prepare documents for vector store
    vs = get_vs()
    docs = []
    
    for i, chunk in enumerate(chunks):
        # Extract page number if available (mainly for PDFs)
        page = None
        if chunk.metadata:
            page = chunk.metadata.get("page")
            if isinstance(page, str):
                try:
                    page = int(page)
                except:
                    page = None
        
        # Create comprehensive metadata
        meta = {
            "file_id": file_id,
            "chunk_no": i,
            "type": file_type,
            "source": os.path.basename(path),
            "original_loader": chunk.metadata.get("loader", "unknown") if chunk.metadata else "unknown"
        }
        
        # Add page number if available
        if page is not None:
            meta["page"] = page
        
        # Add folder if provided
        if folder_id:
            meta["folder_id"] = folder_id
        
        # Add file-type specific metadata
        if chunk.metadata:
            if file_type == 'csv':
                if 'columns' in chunk.metadata:
                    meta['columns'] = chunk.metadata['columns']
                if 'rows' in chunk.metadata:
                    meta['total_rows'] = chunk.metadata['rows']
                if 'is_row_data' in chunk.metadata:
                    meta['is_row_data'] = True
                    meta['row_number'] = chunk.metadata.get('row_number')
            elif file_type == 'text':
                if 'encoding' in chunk.metadata:
                    meta['encoding'] = chunk.metadata['encoding']
        
        docs.append(Document(page_content=chunk.page_content, metadata=meta))
    
    # Store in vector database
    try:
        vs.add_documents(docs)
        print(f"[Ingest] Added {len(docs)} chunks to vector store")
    except Exception as e:
        print(f"[Ingest] Error adding to vector store: {e}")
        return 0
    
    # GraphRAG processing (if enabled)
    graphrag_funcs = get_graphrag_functions()
    if _GR_ENABLED and graphrag_funcs:
        print(f"[GraphRAG] Starting processing for {len(chunks)} chunks of type {file_type}")
        
        for i, chunk in enumerate(chunks):
            try:
                page = chunk.metadata.get("page") if chunk.metadata else None
                print(f"[GraphRAG] Processing entities and relations for file={file_id}, chunk={i}, type={file_type}")
                
                # Limit text length for better processing
                text_content = chunk.page_content[:1500]
                
                graphrag_funcs['upsert_entities_and_relations'](
                    file_id=file_id,
                    page=page,
                    chunk_no=i,
                    source=os.path.basename(path),
                    text=text_content
                )
            except Exception as e:
                print(f"[GraphRAG] Entity/relation extraction failed for chunk {i}: {e}")
                continue
    
    return len(docs)

# Legacy function for backward compatibility
def ingest_pdf(path: str, file_id: str, folder_id: Optional[str] = None) -> int:
    """Legacy PDF ingestion function - now uses universal ingest_file."""
    return ingest_file(path, file_id, folder_id, file_type='pdf')


#raggging
SYSTEM_PROMPT = """You are a specialized AI assistant. Your primary function is to provide a detailed and accurate answer to the user's question based exclusively on the information contained within the provided context.

Core Instructions

    Strictly Grounded: Your entire answer must be derived solely from the provided text. Do not introduce any external information, assumptions, or prior knowledge. Your task is to synthesize, not to create.

    Handle Insufficient Information: If the context does not contain the information required to fully and accurately answer the question, you must state that the answer cannot be found in the provided documents. Do not attempt to infer, guess, or fabricate an answer.

Response Style and Formatting

    Tone: Maintain a clear, professional, and helpful tone.

    Content: The response should be comprehensive yet concise, directly addressing the user's query. Synthesize relevant points from the text into a coherent, well-structured explanation.

    Length: Ensure the response is at least 200 words.

    Exclusions: Do not include citations, document metadata, or introductory phrases like "Based on the provided context..." or "The document states...". Integrate the information naturally into your response."""


def _cite(d: Document) -> str:
    m = d.metadata or {}
    return f"[file:{m.get('file_id')} chunk:{m.get('chunk_no')}]"

def _context_block(docs: List[Document], max_chars: int) -> str:
    out, used = [], 0
    for d in docs:
        add = f"{d.page_content.strip()}\n{_cite(d)}\n---\n"
        if used + len(add) > max_chars:
            break
        out.append(add)
        used += len(add)
    return "".join(out)

# ========== Conversation Intelligence ==========

FOLLOW_UP_DETECTION_PROMPT = """
You are a conversation context analyzer. Determine if the current query is a follow-up question to the previous conversation.

Previous conversation context:
Query: {prev_query}
Response: {prev_response}

Current query: {current_query}

Analyze if the current query is:
1. A follow-up question referring to the same topic/document/context
2. A continuation or clarification of the previous discussion
3. Uses pronouns (it, this, that, they, etc.) referring to previous context
4. Asks for more details about something mentioned previously

Return ONLY a JSON object with:
{
  "is_followup": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "brief explanation",
  "should_preserve_context": true/false
}
"""

def detect_followup_question(current_query: str, conversation_history: List[Dict]) -> Dict:
    """
    Detect if the current query is a follow-up to previous conversation.
    Returns analysis with follow-up detection and context preservation recommendation.
    """
    if not conversation_history:
        return {
            "is_followup": False,
            "confidence": 0.0,
            "reasoning": "No previous conversation",
            "should_preserve_context": False
        }
    
    # Get the most recent conversation turn
    last_turn = conversation_history[-1]
    prev_query = last_turn.get("query", "")
    prev_response = last_turn.get("response", "")[:500]  # Limit response length
    
    # Simple heuristic checks first
    current_lower = current_query.lower()
    
    # Check for obvious follow-up indicators
    followup_indicators = [
        "what about", "how about", "tell me more", "can you explain", "expand on",
        "what does that mean", "what is that", "what are they", "what is it",
        "more details", "elaborate", "clarify", "continue", "also",
        "in addition", "furthermore", "what else", "anything else"
    ]
    
    pronoun_indicators = [
        " it ", " this ", " that ", " they ", " them ", " these ", " those ",
        "the above", "mentioned", "previous", "earlier"
    ]
    
    # Quick heuristic scoring
    heuristic_score = 0.0
    reasons = []
    
    for indicator in followup_indicators:
        if indicator in current_lower:
            heuristic_score += 0.3
            reasons.append(f"Contains follow-up phrase: '{indicator}'")
    
    for pronoun in pronoun_indicators:
        if pronoun in current_lower:
            heuristic_score += 0.4
            reasons.append(f"Contains reference pronoun: '{pronoun.strip()}'")
    
    # Check if query is very short (likely a follow-up)
    if len(current_query.split()) <= 5 and heuristic_score > 0:
        heuristic_score += 0.2
        reasons.append("Short query with follow-up indicators")
    
    # Check for question words in short queries
    question_words = ["what", "how", "why", "when", "where", "which", "who"]
    if any(current_lower.startswith(qw) for qw in question_words) and len(current_query.split()) <= 8:
        heuristic_score += 0.1
        reasons.append("Short question likely referencing previous context")
    
    # If heuristic score is high enough, consider it a follow-up
    if heuristic_score >= 0.4:
        return {
            "is_followup": True,
            "confidence": min(0.9, heuristic_score),
            "reasoning": "; ".join(reasons),
            "should_preserve_context": True
        }
    
    # For borderline cases, use LLM analysis
    if heuristic_score > 0.1 or len(current_query.split()) <= 6:
        try:
            prompt = FOLLOW_UP_DETECTION_PROMPT.format(
                prev_query=prev_query,
                prev_response=prev_response,
                current_query=current_query
            )
            
            llm_instance = llm()
            response = llm_instance.invoke(prompt).content.strip()
            
            # Parse LLM response
            llm_analysis = _safe_extract_json(response, dict)
            if llm_analysis:
                return llm_analysis
                
        except Exception as e:
            print(f"[Conversation] LLM follow-up detection failed: {e}")
    
    # Default: not a follow-up
    return {
        "is_followup": False,
        "confidence": 1.0 - heuristic_score,
        "reasoning": "No clear follow-up indicators found",
        "should_preserve_context": False
    }

def get_conversation_context(conversation_history: List[Dict], max_context_chars: int = 2000) -> str:
    """
    Build conversation context string from history.
    """
    if not conversation_history:
        return ""
    
    context_parts = []
    current_length = 0
    
    # Start from most recent and work backwards
    for turn in reversed(conversation_history[-3:]):  # Last 3 turns max
        query = turn.get("query", "")
        response = turn.get("response", "")
        
        turn_context = f"Q: {query}\nA: {response[:300]}...\n---\n"
        
        if current_length + len(turn_context) > max_context_chars:
            break
            
        context_parts.insert(0, turn_context)
        current_length += len(turn_context)
    
    return "".join(context_parts)

def preserve_conversation_scope(conversation_history: List[Dict]) -> Tuple[str, Optional[str], Optional[str]]:
    """
    Extract scope information from conversation history to maintain context.
    Returns (scope, file_id, folder_id) from the most relevant previous interaction.
    """
    if not conversation_history:
        return "drive", None, None
    
    # Look for the most recent interaction with specific scope
    for turn in reversed(conversation_history):
        scope = turn.get("scope", "drive")
        file_id = turn.get("file_id")
        folder_id = turn.get("folder_id")
        
        # Prefer file-specific scope over folder or drive
        if scope == "file" and file_id:
            return scope, file_id, folder_id
        elif scope == "folder" and folder_id:
            return scope, file_id, folder_id
    
    # Fallback to drive scope
    return "drive", None, None

def _safe_extract_json(response: str, expected_type: type = dict):
    """
    Safely extracts a JSON object from a string response.
    """
    if not response:
        return expected_type()

    # Try to find JSON in the response
    start_pos = -1
    if expected_type == dict:
        start_pos = response.find('{')
        end_char = '}'
    elif expected_type == list:
        start_pos = response.find('[')
        end_char = ']'
    
    if start_pos == -1:
        return expected_type()
    
    end_pos = response.rfind(end_char)
    if end_pos == -1:
        return expected_type()
    
    json_str = response[start_pos:end_pos + 1]
    
    try:
        data = json.loads(json_str)
        if isinstance(data, expected_type):
            return data
        return expected_type()
    except json.JSONDecodeError:
        return expected_type()

def _keywords(q: str) -> List[str]:
    toks = [t for t in re.split(r"[^a-zA-Z0-9]+", q.lower()) if len(t) > 2]
    seen, out = set(), []
    for t in toks:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out[:6]

def _rewrite_query(q: str) -> str:
    try:
        rewritten = llm().invoke(f"Rewrite as a concise search query (no fluff, <=12 words): {q}").content.strip()
        bad_tokens = ["please", "i don't", "provide", "share", "pdf"]
        if any(tok in rewritten.lower() for tok in bad_tokens) or len(rewritten.split()) > 14:
            return q
        return rewritten
    except Exception:
        return q

def _extract_file_references(query: str) -> List[str]:
    """
    Extract potential file references from the query text.
    Looks for patterns like:
    - "file X", "file named X", "document X"
    - "X file", "X document" 
    - quoted strings that might be filenames
    - words ending in common file extensions
    - partial file names (3+ characters)
    """
    import re
    
    query_lower = query.lower()
    file_candidates = []
    
    # Pattern 1: "file X", "document X", "pdf X", etc.
    patterns = [
        r'\b(?:file|document|pdf|doc)\s+(?:named\s+|called\s+)?["\']?([a-zA-Z0-9_\-\.]+)["\']?',
        r'\b([a-zA-Z0-9_\-\.]+)\s+(?:file|document|pdf|doc)\b',
        r'\b(?:saved\s+as|named\s+as|called\s+)["\']([a-zA-Z0-9_\-\.]+)["\']',
        r'["\']([a-zA-Z0-9_\-\.]+)["\']',  # Any quoted strings
        r'\b(?:in\s+(?:the\s+)?|from\s+(?:the\s+)?|about\s+(?:the\s+)?|regarding\s+(?:the\s+)?)([a-zA-Z0-9_\-\.]{3,})\s+(?:file|document|pdf|doc)\b',
        r'\b(?:file|document|pdf|doc)\s+(?:named\s+|called\s+)?([a-zA-Z0-9_\-\.]{3,})\b',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, query_lower, re.IGNORECASE)
        file_candidates.extend(matches)
    
    # Pattern 2: Words with file extensions
    extension_pattern = r'\b([a-zA-Z0-9_\-]+\.(?:pdf|doc|docx|txt|md))\b'
    extension_matches = re.findall(extension_pattern, query_lower, re.IGNORECASE)
    file_candidates.extend(extension_matches)
    
    # Pattern 3: Standalone words that could be file names (3+ chars, not common words)
    # Look for alphanumeric sequences that could be file identifiers
    common_words = {'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had', 'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his', 'how', 'man', 'new', 'now', 'old', 'see', 'two', 'way', 'who', 'boy', 'did', 'its', 'let', 'put', 'say', 'she', 'too', 'use'}
    
    # Extract potential file identifiers (3+ chars, contains numbers or mixed case patterns)
    potential_files = re.findall(r'\b([a-zA-Z0-9_\-]{3,})\b', query_lower)
    for candidate in potential_files:
        # Skip common English words
        if candidate.lower() in common_words:
            continue
        # Include if it has numbers, underscores, hyphens, or mixed patterns
        if (any(c.isdigit() for c in candidate) or 
            '_' in candidate or '-' in candidate or
            len(candidate) >= 5):  # Longer words are more likely to be file names
            file_candidates.append(candidate)
    
    # Clean and deduplicate
    cleaned_candidates = []
    for candidate in file_candidates:
        candidate = candidate.strip(' "\',.')
        if candidate and len(candidate) >= 3 and candidate not in cleaned_candidates:
            cleaned_candidates.append(candidate)
    
    return cleaned_candidates[:5]  # Return top 5 candidates

def _calculate_string_similarity(s1: str, s2: str) -> float:
    """
    Calculate similarity between two strings using multiple methods.
    Returns a score between 0.0 and 1.0.
    """
    s1, s2 = s1.lower(), s2.lower()
    
    if s1 == s2:
        return 1.0
    
    # Method 1: Substring matching
    if s1 in s2 or s2 in s1:
        shorter, longer = (s1, s2) if len(s1) < len(s2) else (s2, s1)
        substring_score = len(shorter) / len(longer)
    else:
        substring_score = 0.0
    
    # Method 2: Common prefix/suffix
    prefix_len = 0
    for i in range(min(len(s1), len(s2))):
        if s1[i] == s2[i]:
            prefix_len += 1
        else:
            break
    
    suffix_len = 0
    for i in range(1, min(len(s1), len(s2)) + 1):
        if s1[-i] == s2[-i]:
            suffix_len += 1
        else:
            break
    
    prefix_suffix_score = (prefix_len + suffix_len) / max(len(s1), len(s2))
    
    # Method 3: Character overlap (Jaccard similarity)
    set1, set2 = set(s1), set(s2)
    jaccard_score = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0
    
    # Combine scores with weights
    final_score = (
        substring_score * 0.5 +
        prefix_suffix_score * 0.3 +
        jaccard_score * 0.2
    )
    
    return min(1.0, final_score)

async def _check_file_exists_in_db(file_candidates: List[str], vs: Chroma) -> Optional[str]:
    """
    Check if any of the file candidates actually exist in the database.
    Uses fuzzy matching to find the best match.
    Returns the matching file_id found.
    """
    if not file_candidates:
        return None
    
    try:
        print(f"[RAG] Checking file candidates: {file_candidates}")
        
        # Get a sample of documents to check against
        sample_docs = vs.similarity_search("", k=50)  # Get some documents to check file_ids
        
        available_files = {}  # {file_id: [source, sample_content]}
        for doc in sample_docs:
            if doc.metadata:
                file_id = doc.metadata.get("file_id", "")
                source = doc.metadata.get("source", "")
                if file_id:
                    available_files[file_id] = [source, doc.page_content[:100]]
        
        print(f"[RAG] Available files in database: {list(available_files.keys())}")
        
        best_match = None
        best_score = 0.0
        
        # Check each candidate against available files
        for candidate in file_candidates:
            for file_id, (source, content) in available_files.items():
                # Calculate similarity scores
                file_id_score = _calculate_string_similarity(candidate, file_id)
                source_score = _calculate_string_similarity(candidate, source.replace('.pdf', '').replace('.txt', ''))
                
                # Take the best score
                score = max(file_id_score, source_score)
                
                # Bonus for exact substring matches
                if candidate.lower() in file_id.lower() or candidate.lower() in source.lower():
                    score += 0.2
                if file_id.lower() in candidate.lower() or source.replace('.pdf', '').lower() in candidate.lower():
                    score += 0.2
                
                score = min(1.0, score)
                
                print(f"[RAG] Candidate '{candidate}' vs file '{file_id}' (source: {source}): score = {score:.3f}")
                
                if score > best_score and score >= 0.4:  # Minimum threshold
                    best_score = score
                    best_match = file_id
        
        if best_match:
            print(f"[RAG] Best file match: '{best_match}' with score {best_score:.3f}")
            return best_match
        else:
            print("[RAG] No good file matches found")
                
    except Exception as e:
        print(f"[RAG] Error checking file existence: {e}")
    
    return None

def _auto_detect_file_scope(query: str, vs: Chroma, current_scope: str, current_file_id: Optional[str]) -> Tuple[str, Optional[str]]:
    """
    Automatically detect if the query is asking about a specific file and adjust scope accordingly.
    Returns (new_scope, new_file_id)
    """
    # If already scoped to a specific file, don't change
    if current_scope == "file" and current_file_id:
        return current_scope, current_file_id
    
    # Extract potential file references from query
    file_candidates = _extract_file_references(query)
    
    if not file_candidates:
        return current_scope, current_file_id
    
    print(f"[RAG] Detected potential file references in query: {file_candidates}")
    
    # Check if any candidates exist in the database using improved matching
    try:
        print(f"[RAG] Checking file candidates: {file_candidates}")
        
        # Get a sample of documents to check against
        sample_docs = vs.similarity_search("", k=50)  # Get some documents to check file_ids
        
        available_files = {}  # {file_id: [source, sample_content]}
        for doc in sample_docs:
            if doc.metadata:
                file_id = doc.metadata.get("file_id", "")
                source = doc.metadata.get("source", "")
                if file_id:
                    available_files[file_id] = [source, doc.page_content[:100]]
        
        print(f"[RAG] Available files in database: {list(available_files.keys())}")
        
        best_match = None
        best_score = 0.0
        
        # Check each candidate against available files
        for candidate in file_candidates:
            for file_id, (source, content) in available_files.items():
                # Calculate similarity scores
                file_id_score = _calculate_string_similarity(candidate, file_id)
                source_score = _calculate_string_similarity(candidate, source.replace('.pdf', '').replace('.txt', ''))
                
                # Take the best score
                score = max(file_id_score, source_score)
                
                # Bonus for exact substring matches
                if candidate.lower() in file_id.lower() or candidate.lower() in source.lower():
                    score += 0.2
                if file_id.lower() in candidate.lower() or source.replace('.pdf', '').lower() in candidate.lower():
                    score += 0.2
                
                score = min(1.0, score)
                
                print(f"[RAG] Candidate '{candidate}' vs file '{file_id}' (source: {source}): score = {score:.3f}")
                
                if score > best_score and score >= 0.4:  # Minimum threshold
                    best_score = score
                    best_match = file_id
        
        if best_match:
            print(f"[RAG] Auto-detected file scope: file_id='{best_match}' with score {best_score:.3f}")
            return "file", best_match
        else:
            print("[RAG] No good file matches found")
                
    except Exception as e:
        print(f"[RAG] Error in auto file detection: {e}")
    
    # If no matching files found, keep original scope
    return current_scope, current_file_id

def _search_docs_with_scores(vs: Chroma, query: str, where: Dict, k: int):
    print(f"[ChromaDB] Searching with query='{query[:50]}...', filter={where}, k={k}")
    try:
        results = vs.similarity_search_with_score(query, k=k, filter=where or None)
        print(f"[ChromaDB] Found {len(results)} results using similarity_search_with_score")
        return results
    except Exception as e:
        print(f"[ChromaDB] similarity_search_with_score failed: {e}, falling back to max_marginal_relevance_search")
        docs = vs.max_marginal_relevance_search(query, k=k, fetch_k=max(20, k*3), filter=where or None)
        print(f"[ChromaDB] Found {len(docs)} results using max_marginal_relevance_search")
        # fallback has no distance -> assign neutral distance
        return [(d, 1.0) for d in docs]

# distances: cosine distance
CONF_DISTANCE_MAX = 0.40
CACHE_DISTANCE_MAX = 0.12

def _confidence_from_distances(distances: List[float]) -> float:
    if not distances:
        return 0.0
    # map distance -> similarity ~ (1 - distance), clamp to [0,1]
    sims = [max(0.0, 1.0 - min(1.0, d)) for d in distances]
    return sum(sims) / len(sims)

def rag_answer(query: str, scope: str = "drive", folder_id: Optional[str] = None,
               file_id: Optional[str] = None, k: int = 10) -> Dict:
    """
    Non-streaming answer. (Your UI can still prefer SSE.)
    Includes namespaced cache and confidence gating.
    """
    vs = get_vs()
    cache_vs = get_cache_vs()

    # Auto-detect file scope based on query content
    scope, file_id = _auto_detect_file_scope(query, vs, scope, file_id)

    # scope filter
    where: Dict[str, str] = {}
    if scope == "folder" and folder_id:
        where["folder_id"] = folder_id
    if scope == "file" and file_id:
        where["file_id"] = file_id
    
    print(f"[RAG] rag_answer called with scope={scope}, file_id={file_id}, folder_id={folder_id}")
    print(f"[RAG] Where filter: {where}")

    # namespaced cache key/query
    cache_query = f"scope={scope}|folder={folder_id}|file={file_id}|q={query}"
    try:
        ch = cache_vs.similarity_search_with_score(cache_query, k=1)
        if ch:
            doc, dist = ch[0]
            if dist <= CACHE_DISTANCE_MAX:
                meta = doc.metadata or {}
                if "answer_json" in meta:
                    return json.loads(meta["answer_json"])
    except Exception:
        pass

    # retrieval
    q_lower = query.lower()
    if ("summarize" in q_lower or "summary" in q_lower) and (scope == "file" and file_id):
        rewritten_query = query
        pairs = _search_docs_with_scores(vs, "summary", where, k=max(k, 30))
    else:
        rewritten_query = _rewrite_query(query)
        kw = " ".join(_keywords(query))
        boosted = f"{rewritten_query} {kw}".strip()
        pairs = _search_docs_with_scores(vs, boosted, where, k=k)

    docs = [d for d, _ in pairs]
    distances = [dist for _, dist in pairs]
    avg_distance = (sum(distances)/len(distances)) if distances else 1.0
    confidence = round(_confidence_from_distances(distances), 3)

    if not docs or avg_distance > CONF_DISTANCE_MAX:
        return {
            "answer": "I don't know based on the indexed files.",
            "citations": [],
            "rewritten_query": rewritten_query,
            "confidence": confidence,
        }

    # build base context from vector hits
    ctx = _context_block(docs, settings.MAX_CONTEXT_CHARS)

    # prepend Graph Facts when enabled
    graphrag_funcs = get_graphrag_functions()
    if _GR_ENABLED and graphrag_funcs:
        try:
            ents = graphrag_funcs['extract_query_entities'](query)
            # Use confidence-based graph fact retrieval with minimum confidence filtering
            gctx = graphrag_funcs['get_subgraph_facts'](
                ents,
                file_id=file_id if scope == "file" else None,
                folder_id=folder_id if scope == "folder" else None,
                max_facts=settings.MAX_CONTEXT_CHARS // 100,  # Adjust based on context limit
                min_confidence=0.6  # Only include high-confidence facts
            )
            if gctx:
                ctx = f"# Graph Facts\n{gctx}\n---\n" + ctx
        except Exception:
            pass

    prompt = f"{SYSTEM_PROMPT}\n\n# Question\n{query}\n\n# Context\n{ctx}\n# Answer"
    answer_text = response_llm().invoke(prompt).content.strip()  # Use separate response model

    citations = []
    for d in docs:
        m = d.metadata or {}
        citations.append({
            "file_id": m.get("file_id"),
            "chunk_no": m.get("chunk_no"),
            "page": m.get("page"),
            "source": m.get("source"),
        })

    result = {
        "answer": answer_text,
        "citations": citations,
        "rewritten_query": rewritten_query,
        "confidence": confidence,
    }

    # store result in cache
    try:
        qid = hashlib.sha256(cache_query.encode("utf-8")).hexdigest()[:24]
        cache_vs.add_texts(
            texts=[cache_query],
            metadatas=[{
                "answer_json": json.dumps(result, ensure_ascii=False),
                "ts": int(time.time()),
                "scope": scope, "folder_id": folder_id, "file_id": file_id
            }],
            ids=[qid],
        )
    except Exception:
        pass

    return result


# ========== Conversation-Aware RAG ==========

def conversation_aware_rag(
    query: str, 
    conversation_history: List[Dict] = None,
    scope: str = "drive", 
    folder_id: Optional[str] = None,
    file_id: Optional[str] = None, 
    k: int = 10
) -> Dict:
    """
    Conversation-aware RAG that maintains context and detects follow-up questions.
    """
    if conversation_history is None:
        conversation_history = []
    
    print(f"[ConversationRAG] Processing query with {len(conversation_history)} previous turns")
    
    # Detect if this is a follow-up question
    followup_analysis = detect_followup_question(query, conversation_history)
    print(f"[ConversationRAG] Follow-up analysis: {followup_analysis}")
    
    # Adjust scope based on conversation context if it's a follow-up
    original_scope, original_file_id, original_folder_id = scope, file_id, folder_id
    
    if followup_analysis["is_followup"] and followup_analysis["should_preserve_context"]:
        # Preserve scope from conversation history
        conv_scope, conv_file_id, conv_folder_id = preserve_conversation_scope(conversation_history)
        
        # Only override if current scope is broader than conversation scope
        if scope == "drive" and conv_scope in ["file", "folder"]:
            scope, file_id, folder_id = conv_scope, conv_file_id, conv_folder_id
            print(f"[ConversationRAG] Preserved conversation scope: {scope}, file_id={file_id}, folder_id={folder_id}")
        elif scope == "folder" and conv_scope == "file":
            scope, file_id = conv_scope, conv_file_id
            print(f"[ConversationRAG] Narrowed scope to file: {file_id}")
    
    # Build enhanced query with conversation context for better retrieval
    enhanced_query = query
    if followup_analysis["is_followup"] and conversation_history:
        # Add conversation context to improve retrieval
        conversation_context = get_conversation_context(conversation_history, max_context_chars=500)
        enhanced_query = f"{conversation_context}\n\nCurrent question: {query}"
        print(f"[ConversationRAG] Enhanced query with conversation context")
    
    # Use regular RAG with potentially adjusted scope and enhanced query
    vs = get_vs()
    cache_vs = get_cache_vs()

    # Auto-detect file scope based on query content (existing functionality)
    scope, file_id = _auto_detect_file_scope(query, vs, scope, file_id)

    # scope filter
    where: Dict[str, str] = {}
    if scope == "folder" and folder_id:
        where["folder_id"] = folder_id
    if scope == "file" and file_id:
        where["file_id"] = file_id
    
    print(f"[ConversationRAG] Final scope: {scope}, file_id={file_id}, folder_id={folder_id}")
    print(f"[ConversationRAG] Where filter: {where}")

    # Namespaced cache key (including conversation context)
    conversation_context_hash = hashlib.md5(str(conversation_history).encode()).hexdigest()[:8] if conversation_history else "none"
    cache_query = f"scope={scope}|folder={folder_id}|file={file_id}|conv={conversation_context_hash}|q={query}"
    
    try:
        ch = cache_vs.similarity_search_with_score(cache_query, k=1)
        if ch:
            doc, dist = ch[0]
            if dist <= CACHE_DISTANCE_MAX:
                meta = doc.metadata or {}
                if "answer_json" in meta:
                    cached_result = json.loads(meta["answer_json"])
                    cached_result["conversation_analysis"] = followup_analysis
                    cached_result["scope_adjusted"] = (scope != original_scope or file_id != original_file_id)
                    return cached_result
    except Exception:
        pass

    # Retrieval with enhanced query for follow-ups
    search_query = enhanced_query if followup_analysis["is_followup"] else query
    
    q_lower = search_query.lower()
    if ("summarize" in q_lower or "summary" in q_lower) and (scope == "file" and file_id):
        rewritten_query = search_query
        pairs = _search_docs_with_scores(vs, "summary", where, k=max(k, 30))
    else:
        rewritten_query = _rewrite_query(search_query)
        kw = " ".join(_keywords(search_query))
        boosted = f"{rewritten_query} {kw}".strip()
        pairs = _search_docs_with_scores(vs, boosted, where, k=k)

    docs = [d for d, _ in pairs]
    distances = [dist for _, dist in pairs]
    avg_distance = (sum(distances)/len(distances)) if distances else 1.0
    confidence = round(_confidence_from_distances(distances), 3)

    if not docs or avg_distance > CONF_DISTANCE_MAX:
        return {
            "answer": "I don't know based on the indexed files and conversation context.",
            "citations": [],
            "rewritten_query": rewritten_query,
            "confidence": confidence,
            "conversation_analysis": followup_analysis,
            "scope_adjusted": (scope != original_scope or file_id != original_file_id)
        }

    # Build context
    ctx = _context_block(docs, settings.MAX_CONTEXT_CHARS)

    # Add conversation context to the prompt for follow-ups
    conversation_prompt_context = ""
    if followup_analysis["is_followup"] and conversation_history:
        conversation_prompt_context = f"\n\n# Previous Conversation Context\n{get_conversation_context(conversation_history, 800)}\n"

    # GraphRAG integration
    graphrag_funcs = get_graphrag_functions()
    if _GR_ENABLED and graphrag_funcs:
        try:
            ents = graphrag_funcs['extract_query_entities'](query)
            gctx = graphrag_funcs['get_subgraph_facts'](
                ents,
                file_id=file_id if scope == "file" else None,
                folder_id=folder_id if scope == "folder" else None,
                max_facts=settings.MAX_CONTEXT_CHARS // 100,
                min_confidence=0.6
            )
            if gctx:
                ctx = f"# Graph Facts\n{gctx}\n---\n" + ctx
        except Exception:
            pass

    # Enhanced system prompt for conversation awareness
    enhanced_system_prompt = SYSTEM_PROMPT
    if followup_analysis["is_followup"]:
        enhanced_system_prompt += "\n\nIMPORTANT: This is a follow-up question to a previous conversation. Use the conversation context provided to maintain consistency and avoid repeating information already discussed."

    prompt = f"{enhanced_system_prompt}{conversation_prompt_context}\n\n# Question\n{query}\n\n# Context\n{ctx}\n# Answer"
    answer_text = response_llm().invoke(prompt).content.strip()  # Use separate response model

    citations = []
    for d in docs:
        m = d.metadata or {}
        citations.append({
            "file_id": m.get("file_id"),
            "chunk_no": m.get("chunk_no"),
            "page": m.get("page"),
            "source": m.get("source"),
        })

    result = {
        "answer": answer_text,
        "citations": citations,
        "rewritten_query": rewritten_query,
        "confidence": confidence,
        "conversation_analysis": followup_analysis,
        "scope_adjusted": (scope != original_scope or file_id != original_file_id),
        "final_scope": {"scope": scope, "file_id": file_id, "folder_id": folder_id}
    }

    # Store result in cache
    try:
        qid = hashlib.sha256(cache_query.encode("utf-8")).hexdigest()[:24]
        cache_vs.add_texts(
            texts=[cache_query],
            metadatas=[{
                "answer_json": json.dumps(result, ensure_ascii=False),
                "ts": int(time.time()),
                "scope": scope, "folder_id": folder_id, "file_id": file_id,
                "is_followup": followup_analysis["is_followup"]
            }],
            ids=[qid],
        )
    except Exception:
        pass

    return result


# SSE

def build_prompt_and_citations(query: str, scope: str = "drive",
                               folder_id: Optional[str] = None,
                               file_id: Optional[str] = None,
                               k: int = 10) -> Tuple[str, list, str, dict]:
    """
    Build the final prompt (+context) and the citations list for SSE streaming.
    
    Follows the refined GraphRAG pipeline:
    1. Similarity Search: Find relevant raw text chunks via ChromaDB
    2. Query Analysis: Extract key entities from the query using LLM
    3. Graph Retrieval: Query Neo4j for facts connected to those entities
    4. Context Enrichment: Place structured graph facts before unstructured text
    5. Final Answer Generation: Combine everything into the prompt
    
    Returns: (prompt, citations, rewritten_query, graphrag_info)
    """
    print(f"[RAG Pipeline] Starting for query: '{query[:50]}...'")
    
    vs = get_vs()
    
    # Auto-detect file scope based on query content
    original_scope, original_file_id = scope, file_id
    scope, file_id = _auto_detect_file_scope(query, vs, scope, file_id)
    
    if scope != original_scope or file_id != original_file_id:
        print(f"[RAG Pipeline] Auto-adjusted scope from {original_scope}/{original_file_id} to {scope}/{file_id}")
    
    where: Dict[str, str] = {}
    if scope == "folder" and folder_id:
        where["folder_id"] = folder_id
    if scope == "file" and file_id:
        where["file_id"] = file_id

    # STEP 1: Similarity Search on ChromaDB
    print(f"[RAG Pipeline] Step 1: Performing similarity search on ChromaDB...")
    print(f"[RAG Pipeline] Scope: {scope}, File ID: {file_id}, Folder ID: {folder_id}")
    print(f"[RAG Pipeline] Where filter: {where}")
    
    q_lower = query.lower()
    if ("summarize" in q_lower or "summary" in q_lower) and (scope == "file" and file_id):
        print(f"[RAG Pipeline] Using summary search for file {file_id}")
        pairs = _search_docs_with_scores(vs, "summary", where, k=max(k, 30))
        rewritten_query = query
    else:
        rewritten_query = _rewrite_query(query)
        kw = " ".join(_keywords(query))
        boosted = f"{rewritten_query} {kw}".strip()
        print(f"[RAG Pipeline] Using boosted search: '{boosted}' with filter: {where}")
        pairs = _search_docs_with_scores(vs, boosted, where, k=k)

    docs = [d for d, _ in pairs]
    print(f"[RAG Pipeline] Found {len(docs)} relevant text chunks from vector search")
    
    if not docs:
        ctx = ""
        citations = []
    else:
        ctx = _context_block(docs, settings.MAX_CONTEXT_CHARS)
        citations = []
        for d in docs:
            m = d.metadata or {}
            citations.append({
                "file_id": m.get("file_id"),
                "chunk_no": m.get("chunk_no"),
                "page": m.get("page"),
                "source": m.get("source"),
            })

    # Initialize GraphRAG info
    graphrag_info = {
        "enabled": _GR_ENABLED,
        "functions_loaded": bool(get_graphrag_functions()),
        "entities_extracted": [],
        "facts_found": False,
        "facts_length": 0,
        "graph_facts_count": 0,
        "error": None
    }

    # STEPS 2-4: GraphRAG Pipeline (Query Analysis + Graph Retrieval + Context Enrichment)
    graphrag_funcs = get_graphrag_functions()
    if _GR_ENABLED and graphrag_funcs:
        try:
            # STEP 2: Query Analysis - Extract key entities from the query
            print("[RAG Pipeline] Step 2: Analyzing query to extract key entities...")
            entities = graphrag_funcs['extract_query_entities'](query)
            graphrag_info["entities_extracted"] = entities
            print(f"[RAG Pipeline] Extracted entities: {entities}")
            
            if entities:
                # STEP 3: Graph Retrieval - Query Neo4j for facts related to entities
                print("[RAG Pipeline] Step 3: Querying Neo4j knowledge graph...")
                graph_facts = graphrag_funcs['get_subgraph_facts'](
                    entities,
                    file_id=file_id if scope == "file" else None,
                    folder_id=folder_id if scope == "folder" else None,
                    max_facts=min(settings.MAX_CONTEXT_CHARS // 150, 120),  # Dynamic fact limit
                    min_confidence=0.6  # Only high-confidence facts for better context quality
                )
                
                if graph_facts:
                    # STEP 4: Context Enrichment - Place graph facts before raw text
                    print("[RAG Pipeline] Step 4: Enriching context with graph facts...")
                    graphrag_info["facts_found"] = True
                    graphrag_info["facts_length"] = len(graph_facts)
                    graphrag_info["graph_facts_count"] = len(graph_facts.split('\n'))
                    
                    # Place structured graph facts at the beginning, followed by unstructured text
                    ctx = f"# Graph Facts (Structured Knowledge)\n{graph_facts}\n\n# Raw Text Chunks (Unstructured)\n{ctx}"
                    print(f"[RAG Pipeline] Added {graphrag_info['graph_facts_count']} graph facts to context")
                else:
                    print("[RAG Pipeline] No relevant graph facts found for extracted entities")
            else:
                print("[RAG Pipeline] No entities extracted from query")
                
        except Exception as e:
            print(f"[RAG Pipeline] GraphRAG error: {e}")
            graphrag_info["error"] = str(e)
    else:
        if not _GR_ENABLED:
            print("[RAG Pipeline] GraphRAG disabled")
        else:
            print("[RAG Pipeline] GraphRAG functions not loaded")

    # STEP 5: Final Answer Generation - Build the complete prompt
    print("[RAG Pipeline] Step 5: Building final prompt for LLM...")
    prompt = f"{SYSTEM_PROMPT}\n\n# Question\n{query}\n\n# Context\n{ctx}\n\n# Answer"
    
    print(f"[RAG Pipeline] Complete! Context length: {len(ctx)} chars, Citations: {len(citations)}")
    return prompt, citations, rewritten_query, graphrag_info


async def stream_gemini(prompt: str):
    """
    Stream tokens from Google Gemini API using the response model and API key.
    """
    import google.generativeai as genai
    
    genai.configure(api_key=settings.RESPONSE_API_KEY)  # Use separate response API key
    model = genai.GenerativeModel(settings.RESPONSE_MODEL)  # Use separate response model
    
    try:
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
            ),
            stream=True
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"Error: {str(e)}"

def clear_cache_collection():
    import chromadb
    host, port = _http_host_port()
    client = chromadb.HttpClient(host=host, port=port)
    try:
        client.delete_collection(f"{settings.COLLECTION}_cache")
        print(f"[Chroma] Cleared cache collection: {settings.COLLECTION}_cache")
        return True
    except Exception as e:
        print(f"[Chroma] Failed to clear cache collection: {e}")
        return False

def clear_main_collection():
    """Clear the main Chroma collection containing all documents."""
    import chromadb
    host, port = _http_host_port()
    client = chromadb.HttpClient(host=host, port=port)
    try:
        client.delete_collection(settings.COLLECTION)
        print(f"[Chroma] Cleared main collection: {settings.COLLECTION}")
        return True
    except Exception as e:
        print(f"[Chroma] Failed to clear main collection: {e}")
        return False

def clear_all_collections():
    """Clear both main and cache Chroma collections."""
    main_success = clear_main_collection()
    cache_success = clear_cache_collection()
    return main_success and cache_success

def get_chroma_stats():
    """Get statistics about Chroma collections."""
    import chromadb
    host, port = _http_host_port()
    client = chromadb.HttpClient(host=host, port=port)
    
    stats = {
        "main_collection": {
            "name": settings.COLLECTION,
            "count": 0,
            "exists": False
        },
        "cache_collection": {
            "name": f"{settings.COLLECTION}_cache",
            "count": 0,
            "exists": False
        }
    }
    
    try:
        # Check main collection
        try:
            main_collection = client.get_collection(settings.COLLECTION)
            stats["main_collection"]["exists"] = True
            stats["main_collection"]["count"] = main_collection.count()
        except Exception:
            pass
        
        # Check cache collection
        try:
            cache_collection = client.get_collection(f"{settings.COLLECTION}_cache")
            stats["cache_collection"]["exists"] = True
            stats["cache_collection"]["count"] = cache_collection.count()
        except Exception:
            pass
            
    except Exception as e:
        stats["error"] = str(e)
    
    return stats

def get_graph_stats():
    """Get GraphRAG statistics with confidence metrics."""
    graphrag_funcs = get_graphrag_functions()
    if not graphrag_funcs:
        return {"error": "GraphRAG not available"}
    
    try:
        return graphrag_funcs['get_graph_stats']()
    except Exception as e:
        return {"error": f"graph stats: {e}"}

def clear_graph():
    """Clear the GraphRAG knowledge graph."""
    graphrag_funcs = get_graphrag_functions()
    if not graphrag_funcs:
        return False
    
    try:
        return graphrag_funcs['clear_graph']()
    except Exception as e:
        print(f"Failed to clear graph: {e}")
        return False

def cleanup_low_confidence_entities(confidence_threshold: float = None):
    """Clean up low-confidence entities from the graph."""
    graphrag_funcs = get_graphrag_functions()
    if not graphrag_funcs:
        return {"error": "GraphRAG not available"}
    
    try:
        return graphrag_funcs['cleanup_low_confidence_entities'](confidence_threshold)
    except Exception as e:
        return {"error": str(e)}

def update_confidence_thresholds(entity_threshold: float = None, relation_threshold: float = None):
    """Update confidence thresholds for entity and relation filtering."""
    graphrag_funcs = get_graphrag_functions()
    if not graphrag_funcs:
        return {"error": "GraphRAG not available"}
    
    try:
        return graphrag_funcs['update_confidence_thresholds'](entity_threshold, relation_threshold)
    except Exception as e:
        return {"error": str(e)}

def get_graphrag_config():
    """Get current GraphRAG configuration."""
    graphrag_funcs = get_graphrag_functions()
    if not graphrag_funcs:
        return {"error": "GraphRAG not available"}
    
    try:
        return {
            "entity_confidence_threshold": graphrag_funcs['get_entity_threshold'](),
            "relation_confidence_threshold": graphrag_funcs['get_relation_threshold'](),
            "embedding_validation_enabled": graphrag_funcs['get_validation_enabled'](),
            "graphrag_enabled": True
        }
    except Exception as e:
        return {"error": str(e)}