import asyncio
import time
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import hashlib
from pathlib import Path
from collections import Counter
from dotenv import load_dotenv
from graphiti_core import Graphiti
from graphiti_core.llm_client.gemini_client import GeminiClient, LLMConfig
from graphiti_core.embedder.gemini import GeminiEmbedder, GeminiEmbedderConfig
from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient

# Import your existing modules
from documentProcessor import DocumentProcessor, DocumentType, DocumentMetadata, FolderScope

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

class EpisodeType(Enum):
    """Episode types for knowledge graph entries"""
    text = "text"
    document = "document" 
    message = "message"
    event = "event"

class FolderScopedGraphBuilder:
    """Enhanced class with robust error handling"""
    
    def __init__(self, neo4j_config: Dict[str, Any], google_api_key: str,gemini_model: str = "gemini-1.5-pro",embedding_model: str = "text-embedding-004",
                 reranker_model: str = "gemini-reranker-27b") -> None:
        
        self.neo4j_config = neo4j_config
        self.google_api_key = google_api_key
        self.gemini_model = gemini_model
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.document_processor = DocumentProcessor()
        self.graphiti_client = None
        self.processed_documents: Dict[str, DocumentMetadata] = {}
        self.current_folder_scope: Optional[FolderScope] = None
        self.max_retries = 3
        self.base_retry_delay = 1.0
        
    async def initialize(self) -> None:
        """Initialize Graphiti client with Gemini configuration"""
        try:
            llm_client = GeminiClient(
                config=LLMConfig(
                    api_key=self.google_api_key,
                    model=self.gemini_model
                )
            )
            embedder = GeminiEmbedder(
                config=GeminiEmbedderConfig(
                    api_key=self.google_api_key,
                    embedding_model=self.embedding_model
                )
            )
            cross_encoder = GeminiRerankerClient(
                config=LLMConfig(
                    api_key=self.google_api_key,
                    model=self.reranker_model
                )
            )
            self.graphiti_client = Graphiti(
                uri=self.neo4j_config['uri'],
                user=self.neo4j_config['user'],
                password=self.neo4j_config['password'],
                llm_client=llm_client,
                embedder=embedder,
                cross_encoder=cross_encoder
            )
            if hasattr(self.graphiti_client, 'build_indices_and_constraints'):
                await self.graphiti_client.build_indices_and_constraints()
            elif hasattr(self.graphiti_client, 'build_indices'):
                await self.graphiti_client.build_indices()

            #logger.info(f"Graphiti client initialized successfully with Gemini model: {self.gemini_model}")

        except Exception as e:
            #logger.error(f"Failed to initialize Graphiti client: {str(e)}")
            raise

    def set_folder_scope(self, folder_path: str, allowed_extensions: Optional[set] = None, recursive_scan: bool = True, max_file_size: int = 50 * 1024 * 1024) -> None:
        """Set the folder scope for document processing"""
        
        if not os.path.exists(folder_path):
            raise ValueError(f"Folder path does not exist: {folder_path}")
        
        folder_id = self._generate_folder_id(folder_path)
        
        if allowed_extensions is None:
            allowed_extensions = {'.pdf', '.docx', '.doc', '.txt', '.md', '.json', '.csv', '.xlsx', '.pptx', '.jpg', '.jpeg', '.png'}
        
        self.current_folder_scope = FolderScope(
            folder_path=os.path.abspath(folder_path),
            folder_id=folder_id,
            allowed_extensions=allowed_extensions,
            max_file_size=max_file_size,
            recursive_scan=recursive_scan
        )
        
        # logger.info(f"Folder scope set to: {folder_path} (ID: {folder_id})")
    
    def _generate_folder_id(self, folder_path: str) -> str:
        """Generate unique folder identifier"""
        abs_path = os.path.abspath(folder_path)
        return hashlib.sha256(abs_path.encode('utf-8')).hexdigest()[:16]
    
    def _generate_document_hash(self, file_path: str) -> str:
        """Generate document hash for duplicate detection"""
        try:
            with open(file_path, 'rb') as f:
                content = f.read()
                return hashlib.sha256(content).hexdigest()
        except Exception as e:
            #logger.error(f"Error generating hash for {file_path}: {str(e)}")
            return hashlib.sha256(file_path.encode()).hexdigest()
    
    def scan_folder(self) -> List[DocumentMetadata]:
        """Scan folder for documents and create metadata"""
        
        if not self.current_folder_scope:
            raise ValueError("Folder scope not set. Call set_folder_scope() first.")
        
        documents = []
        folder_path = Path(self.current_folder_scope.folder_path)
        
        pattern = "**/*" if self.current_folder_scope.recursive_scan else "*"
        
        for file_path in folder_path.glob(pattern):
            if not file_path.is_file():
                continue
                
            if file_path.suffix.lower() not in self.current_folder_scope.allowed_extensions:
                continue
            
            if file_path.stat().st_size > self.current_folder_scope.max_file_size:
                #logger.warning(f"Skipping large file: {file_path} "f"({file_path.stat().st_size} bytes)")
                continue
                
            doc_type = self.document_processor.supported_types.get(
                file_path.suffix.lower()
            )
            if not doc_type:
                continue
            
            doc_metadata = DocumentMetadata(file_path=str(file_path),file_name=file_path.name,
                folder_path=self.current_folder_scope.folder_path,folder_id=self.current_folder_scope.folder_id,
                document_hash=self._generate_document_hash(str(file_path)),file_size=file_path.stat().st_size,
                modification_time=file_path.stat().st_mtime,document_type=doc_type,creation_timestamp=datetime.now())
            
            documents.append(doc_metadata)
        
        logger.info(f"Found {len(documents)} documents in {self.current_folder_scope.folder_path}")
        return documents
    
    async def add_episode_with_retry(self, episode_data: Dict[str, Any]) -> bool:
        """Add episode with exponential backoff retry logic using centralized config"""
        
        for attempt in range(self.max_retries):
            try:
                # Add episode to knowledge graph
                await self.graphiti_client.add_episode(
                    name=episode_data['name'],
                    episode_body=episode_data['content'],
                    reference_time=datetime.now(),
                    source_description=episode_data['source_description']
                )
                logger.info(f"Successfully added episode: {episode_data['name']}")
                return True
                
            except Exception as e:
                error_msg = str(e).lower()
                
                # Simple retry for any error (no rate limit specific handling)
                #logger.warning(f"Error on attempt {attempt + 1}/{self.max_retries}: {str(e)}")
                
                if attempt < self.max_retries - 1:
                    wait_time = self.base_retry_delay * (2 ** attempt)
                    await asyncio.sleep(wait_time)
                    continue
                else:
                    #logger.error(f"Non-recoverable error adding episode {episode_data['name']}: {str(e)}")
                    return False

        #logger.error(f"Failed to add episode {episode_data['name']} after {self.max_retries} attempts")
        return False
    
    async def process_documents(self, 
                              documents: List[DocumentMetadata], 
                              chunk_size: int = 1000, 
                              overlap_size: int = 100,
                              batch_size: int = 5) -> Dict[str, Any]:
        """Process documents with rate limiting and batch processing"""
        
        if not self.graphiti_client:
            raise ValueError("Graphiti client not initialized. Call initialize() first.")
        
        processing_stats = {
            'total_documents': len(documents),
            'processed_successfully': 0,
            'processing_errors': 0,
            'total_episodes': 0,
            'episodes_added_successfully': 0,
            'episodes_failed': 0,
            'processing_time': 0,
            'errors': [],
            'entity_extraction_stats': {},
            'relationship_stats': {}
        }
        
        start_time = datetime.now()
        
        for doc_idx, doc_metadata in enumerate(documents):
            try:
                logger.info(f"Processing document {doc_idx + 1}/{len(documents)}: {doc_metadata.file_name}")
                
                content = self.document_processor.extract_content(
                    doc_metadata.file_path, 
                    doc_metadata.document_type
                )
                
                if not content.strip():
                    logger.warning(f"Empty content extracted from {doc_metadata.file_name}")
                    continue
                
                episodes = self._create_enhanced_episodes_from_content(
                    content, doc_metadata, chunk_size, overlap_size
                )
                
                logger.info(f"Created {len(episodes)} episodes from {doc_metadata.file_name}")
                processing_stats['total_episodes'] += len(episodes)
                
                successful_episodes = 0
                for batch_start in range(0, len(episodes), batch_size):
                    batch_episodes = episodes[batch_start:batch_start + batch_size]
                    
                    batch_results = await self._process_episode_batch(batch_episodes)
                    successful_episodes += sum(1 for r in batch_results if r)
                
                processing_stats['episodes_added_successfully'] += successful_episodes
                processing_stats['episodes_failed'] += len(episodes) - successful_episodes
                
                doc_type = doc_metadata.document_type.value
                if doc_type not in processing_stats['entity_extraction_stats']:
                    processing_stats['entity_extraction_stats'][doc_type] = {'total_chunks': 0, 'estimated_entities': 0}
                
                for episode in episodes:
                    processing_stats['entity_extraction_stats'][doc_type]['total_chunks'] += 1
                    if 'estimated_entities' in episode:
                        processing_stats['entity_extraction_stats'][doc_type]['estimated_entities'] += episode['estimated_entities']
                
                processing_stats['processed_successfully'] += 1
                doc_metadata.processing_status = "completed"
                self.processed_documents[doc_metadata.document_hash] = doc_metadata
                
                logger.info(f"Document {doc_metadata.file_name} processed: "
                           f"{successful_episodes}/{len(episodes)} episodes added successfully")
                
            except Exception as e:
                error_msg = f"Error processing {doc_metadata.file_name}: {str(e)}"
                logger.error(error_msg)
                processing_stats['processing_errors'] += 1
                processing_stats['errors'].append({'document': doc_metadata.file_name, 'error': str(e)})
                doc_metadata.processing_status = f"error: {str(e)}"
        
        processing_stats['processing_time'] = (datetime.now() - start_time).total_seconds()
        
        logger.info(f"Processing completed. Successfully processed: {processing_stats['processed_successfully']}, "
                   f"Errors: {processing_stats['processing_errors']}, "
                   f"Episodes successfully added: {processing_stats['episodes_added_successfully']}/{processing_stats['total_episodes']}")
        
        return processing_stats
    
    async def _process_episode_batch(self, episodes: List[Dict[str, Any]]) -> List[bool]:
        """Process a batch of episodes sequentially"""
        
        results = []
        for episode in episodes:
            success = await self.add_episode_with_retry(episode)
            results.append(success)
        
        return results
    
    def _create_enhanced_episodes_from_content(self, 
                                             content: str, 
                                             doc_metadata: DocumentMetadata,
                                             chunk_size: int, 
                                             overlap_size: int) -> List[Dict[str, Any]]:
        """Create enhanced Graphiti episodes with GraphRAG-specific metadata"""
        
        episodes = []
        chunks = self._split_text_into_chunks(content, chunk_size, overlap_size)
        
        for i, chunk in enumerate(chunks):
            if not chunk.strip():
                continue
                
            episode_name = f"{doc_metadata.folder_id}_{doc_metadata.file_name}_chunk_{i+1}"
            
            episode_metadata = {
                'folder_path': doc_metadata.folder_path,
                'folder_id': doc_metadata.folder_id,
                'document_source': doc_metadata.file_name,
                'document_hash': doc_metadata.document_hash,
                'document_type': doc_metadata.document_type.value,
                'chunk_index': i,
                'total_chunks': len(chunks),
                'creation_timestamp': doc_metadata.creation_timestamp.isoformat(),
                'file_modification_time': doc_metadata.modification_time,
                'content_length': len(chunk),
                'file_size': doc_metadata.file_size,
                'isolated_scope': True,
                'cross_folder_links_allowed': False,
                'graphrag_ready': True,
                'content_type': self._classify_content_type(chunk),
                'estimated_entities': self._estimate_entity_count(chunk),
                'structural_importance': self._calculate_structural_importance(chunk, i, len(chunks))
            }
            
            enhanced_content = self._enhance_content_for_graphrag(chunk, doc_metadata, i)
            
            episode = {
                'name': episode_name,
                'content': enhanced_content,
                'source_description': f"Folder: {doc_metadata.folder_path} | "
                                    f"Document: {doc_metadata.file_name} | "
                                    f"Chunk: {i+1}/{len(chunks)}",
                'metadata': episode_metadata,
                'estimated_entities': episode_metadata['estimated_entities']
            }
            
            episodes.append(episode)
        
        return episodes
    
    def _classify_content_type(self, content: str) -> str:
        """Classify content type for better GraphRAG processing"""
        content_lower = content.lower()
        if any(word in content_lower for word in ['theorem', 'proof', 'equation', 'formula']): return 'mathematical'
        elif any(word in content_lower for word in ['algorithm', 'function', 'class', 'method']): return 'technical'
        elif any(word in content_lower for word in ['introduction', 'abstract', 'summary']): return 'overview'
        elif any(word in content_lower for word in ['conclusion', 'result', 'finding']): return 'conclusion'
        elif any(word in content_lower for word in ['definition', 'concept', 'terminology']): return 'definitional'
        else: return 'general'
    
    def _estimate_entity_count(self, content: str) -> int:
        """Estimate potential entity count for GraphRAG planning"""
        words = content.split()
        potential_entities = sum(1 for word in words if word and word[0].isupper() and len(word) > 2)
        return min(potential_entities, len(words) // 10)
    
    def _calculate_structural_importance(self, content: str, chunk_index: int, total_chunks: int) -> float:
        """Calculate structural importance for GraphRAG weighting"""
        base_importance = 1.2 if chunk_index == 0 or chunk_index == total_chunks - 1 else 1.0
        importance_keywords = ['key', 'important', 'main', 'primary', 'central', 'core', 'fundamental']
        keyword_boost = sum(1 for keyword in importance_keywords if keyword in content.lower())
        base_importance *= (1 + keyword_boost * 0.1)
        return min(base_importance, 2.0)
    
    def _enhance_content_for_graphrag(self, content: str, doc_metadata: DocumentMetadata, chunk_index: int) -> str:
        """Enhance content with GraphRAG-specific context information"""
        context_prefix = f"[Document: {doc_metadata.file_name}] "
        if chunk_index == 0: context_prefix += "[Beginning of document] "
        content_type = self._classify_content_type(content)
        if content_type != 'general': context_prefix += f"[Content type: {content_type}] "
        return context_prefix + content
    
    def _split_text_into_chunks(self, text: str, chunk_size: int, overlap_size: int) -> List[str]:
        """Split text into overlapping chunks with intelligent boundary detection"""
        if len(text) <= chunk_size: return [text]
        chunks, start = [], 0
        while start < len(text):
            end = start + chunk_size
            if end >= len(text):
                chunks.append(text[start:]); break
            chunk = text[start:end]
            breakpoints = [chunk.rfind(bp) for bp in ['. ', '\n\n', '\n', '? ', '! ']]
            valid_breakpoints = [bp for bp in breakpoints if bp > start + chunk_size * 0.6]
            if valid_breakpoints:
                best_break = max(valid_breakpoints)
                actual_end = start + best_break + 1
                chunks.append(text[start:actual_end])
                start = max(actual_end - overlap_size, start + 1)
            else:
                chunks.append(chunk)
                start = max(end - overlap_size, start + 1)
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    async def get_processed_files(self, folder_id: str) -> set:
        """Get list of already processed file hashes from Neo4j"""
        try:
            if not hasattr(self.graphiti_client, '_driver') or not self.graphiti_client._driver:
                return set()
            
            async with self.graphiti_client._driver.session() as session:
                result = await session.run("""
                    MATCH (e:Episode)
                    WHERE e.metadata.folder_id = $folder_id
                    RETURN DISTINCT e.metadata.document_hash as hash
                """, folder_id=folder_id)
                
                hashes = set()
                async for record in result:
                    if record['hash']:
                        hashes.add(record['hash'])
                
                return hashes
        except Exception:
            return set()
    
    async def filter_new_documents(self, documents: List[DocumentMetadata]) -> List[DocumentMetadata]:
        """Filter out already processed documents"""
        if not self.current_folder_scope:
            return documents
        
        processed_hashes = await self.get_processed_files(self.current_folder_scope.folder_id)
        new_documents = [doc for doc in documents if doc.document_hash not in processed_hashes]
        
        if len(new_documents) < len(documents):
            skipped_count = len(documents) - len(new_documents)
        
        return new_documents
    
    async def process_folder(self, folder_path: str, chunk_size: int = 1500, overlap_size: int = 150,
                           allowed_extensions: Optional[set] = None, batch_size: int = 3) -> Dict[str, Any]:
        """Complete pipeline: scan folder, process documents, create knowledge graph"""
        try:
            self.set_folder_scope(folder_path=folder_path, allowed_extensions=allowed_extensions)
            all_documents = self.scan_folder()
            if not all_documents:
                return {'status': 'no_documents_found', 'folder_path': folder_path}
            
            documents = await self.filter_new_documents(all_documents)
            if not documents:
                return {
                    'status': 'completed', 'folder_path': folder_path, 'folder_id': self.current_folder_scope.folder_id,
                    'processing_stats': {'processed_successfully': 0, 'total_episodes': 0, 'episodes_added_successfully': 0},
                    'processed_documents': 0, 'graphrag_ready': True, 'success_rate': 1.0,
                    'message': 'All documents already processed'
                }
            
            processing_stats = await self.process_documents(documents, chunk_size, overlap_size, batch_size)
            
            if processing_stats['episodes_added_successfully'] > 0:
                await self._create_folder_summary_episode(documents, processing_stats)
            
            return {
                'status': 'completed', 'folder_path': folder_path, 'folder_id': self.current_folder_scope.folder_id,
                'processing_stats': processing_stats, 'processed_documents': [asdict(doc) for doc in documents],
                'graphrag_ready': processing_stats['episodes_added_successfully'] > 0,
                'success_rate': processing_stats['episodes_added_successfully'] / max(processing_stats['total_episodes'], 1)
            }
        except Exception as e:
            logger.error(f"Error processing folder {folder_path}: {str(e)}")
            return {'status': 'error', 'folder_path': folder_path, 'error': str(e)}
    
    async def _create_folder_summary_episode(self, documents: List[DocumentMetadata], processing_stats: Dict[str, Any]) -> None:
        """Create a folder-level summary episode for global GraphRAG queries"""
        if not self.current_folder_scope: return
        try:
            doc_types = Counter(doc.document_type.value for doc in documents)
            total_size = sum(doc.file_size for doc in documents)
            summary_content = f"Folder Summary: {self.current_folder_scope.folder_path}\n\n" \
                              f"This folder contains {len(documents)} documents ({total_size / (1024*1024):.2f} MB) " \
                              f"with types: {json.dumps(doc_types, indent=2)}.\n" \
                              f"Processed {processing_stats['processed_successfully']} documents, creating " \
                              f"{processing_stats['total_episodes']} chunks, of which " \
                              f"{processing_stats['episodes_added_successfully']} were added to the graph."
            
            summary_episode = {'name': f"FOLDER_SUMMARY_{self.current_folder_scope.folder_id}", 'content': summary_content,
                               'source_description': f"Folder Summary: {self.current_folder_scope.folder_path}"}
            
            if await self.add_episode_with_retry(summary_episode):
                logger.info(f"Created folder summary episode for {self.current_folder_scope.folder_path}")
            else:
                logger.error("Failed to create folder summary episode")
        except Exception as e:
            logger.error(f"Error creating folder summary episode: {str(e)}")
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.graphiti_client:
            try:
                await self.graphiti_client.close()
                logger.info("Graphiti client closed successfully")
            except Exception as e:
                logger.error(f"Error closing Graphiti client: {str(e)}")

# async def main():
#     # Configuration
#     neo4j_config = {
#         'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
#         'user': os.getenv('NEO4J_USER', 'neo4j'),
#         'password': os.getenv('NEO4J_PASSWORD', 'password')
#     }
#     google_api_key = os.getenv('GOOGLE_API_KEY')
#     folder_path = os.getenv('FOLDER_PATH', './documents')
#     # Initialize builder
#     builder = FolderScopedGraphBuilder(
#         neo4j_config=neo4j_config,
#         google_api_key=google_api_key
#     )
# if __name__ == "__main__":
#     asyncio.run(main())