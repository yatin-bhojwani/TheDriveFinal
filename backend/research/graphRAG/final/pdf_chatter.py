import asyncio
import os
from dotenv import load_dotenv
from typing import Dict, Any
from dataclasses import dataclass
import google.generativeai as genai
from dataExtraction import DirectGraphRAGExtractor, create_enhanced_graphrag_pipeline
from documentProcessor import SearchMode

load_dotenv()

@dataclass
class PDFChatterConfig:
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    google_api_key: str
    gemini_model: str = "gemini-2.5-pro"
    max_communities: int = 5
    max_context_length: int = 8000

class GeminiCaller:
    def __init__(self, api_key: str, model_name: str = "gemini-2.5-pro"):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.api_key = api_key

    async def generate_response(self, query: str, context: str) -> str:
        prompt = f"""
# You are an intelligent document assistant. Answer the user's question based on the provided context from multiple documents.

# CONTEXT FROM DOCUMENTS:
# {context}

# USER QUESTION: {query}

# INSTRUCTIONS:
# 1. Provide a comprehensive answer based on the context
# 2. If information spans multiple documents, synthesize it coherently
# 3. Cite which documents/sources contain relevant information when possible
# 4. If the context doesn't contain enough information, say so clearly
# 5. Be concise but thorough

# ANSWER:
# """
        response = await asyncio.to_thread(
            self.model.generate_content, 
            prompt
        )
        return response.text if response.text else "I couldn't generate a response."

class PDFChatter:
    def __init__(self, config: PDFChatterConfig):
        self.config = config
        self.neo4j_config = {
            'uri': os.getenv('NEO4J_URI', 'bolt://localhost:7687'),
            'user': os.getenv('NEO4J_USER', 'neo4j'),
            'password': os.getenv('NEO4J_PASSWORD', 'password')
        }
        self.graph_builder = None
        self.graphrag_extractor = None
        self.gemini_caller = GeminiCaller(os.getenv('GOOGLE_API_KEY'), "gemini-2.5-pro")
        self.folder_id = None
        self.is_initialized = False

    async def initialize(self):
        self.graph_builder, self.graphrag_extractor = await create_enhanced_graphrag_pipeline(
            self.neo4j_config, 
            os.getenv('GOOGLE_API_KEY')
        )
        self.is_initialized = True

    async def process_folder(self, folder_path: str, force_reprocess: bool = False) -> Dict[str, Any]:
        """Process folder with intelligent file tracking to avoid reprocessing existing documents"""
        if not self.is_initialized:
            await self.initialize()
            
        # Generate folder ID to check for existing processing
        self.folder_id = self.graph_builder._generate_folder_id(folder_path)
        
        # Check if folder was already processed (unless force reprocess)
        if not force_reprocess:
            existing_stats = await self._check_existing_folder_processing(self.folder_id)
            if existing_stats['already_processed']:
                print(f"✅ Folder already processed. Found {existing_stats['documents_count']} documents in database.")
                print(f"📊 Entities: {existing_stats['entities']}, Relationships: {existing_stats['relationships']}")
                print("💡 Use force_reprocess=True to rebuild the entire knowledge graph.")
                
                return {
                    'status': 'already_processed',
                    'folder_id': self.folder_id,
                    'documents_processed': existing_stats['documents_count'],
                    'total_entities': existing_stats['entities'],
                    'total_relationships': existing_stats['relationships'],
                    'communities_detected': existing_stats['communities'],
                    'graphrag_ready': existing_stats['graphrag_ready'],
                    'recommended_modes': existing_stats['recommended_modes']
                }
        
        # Process folder (either first time or forced reprocess)
        build_result = await self.graph_builder.process_folder(
            folder_path=folder_path,
            chunk_size=1200,
            overlap_size=120,
            batch_size=3
        )
        self.folder_id = build_result.get('folder_id')
        await self.graphrag_extractor.wait_for_entity_extraction(max_wait_time=120)
        stats = await self.graphrag_extractor.get_folder_graph_statistics(self.folder_id)
        communities = []
        if stats.get('community_detection_viable', False):
            communities = await self.graphrag_extractor.detect_communities(self.folder_id)
        return {
            'status': 'success',
            'folder_id': self.folder_id,
            'documents_processed': build_result.get('documents_processed', 0),
            'total_entities': stats.get('total_entities', 0),
            'total_relationships': stats.get('total_relationships', 0),
            'communities_detected': len(communities),
            'graphrag_ready': stats.get('graphrag_readiness') == 'READY',
            'recommended_modes': stats.get('recommended_search_modes', [])
        }

    async def _check_existing_folder_processing(self, folder_id: str) -> Dict[str, Any]:
        """Check if folder has already been processed by querying Neo4j database"""
        try:
            driver = self.graphrag_extractor.driver
            
            async with driver.session() as session:
                result = await session.run("""
                    MATCH (e:Episode)
                    WHERE e.metadata.folder_id = $folder_id
                    RETURN count(e) as episode_count
                """, folder_id=folder_id)
                
                record = await result.single()
                episode_count = record['episode_count'] if record else 0
                
                if episode_count == 0:
                    return {
                        'already_processed': False,
                        'documents_count': 0,
                        'entities': 0,
                        'relationships': 0,
                        'communities': 0,
                        'graphrag_ready': False,
                        'recommended_modes': []
                    }
                
                stats = await self.graphrag_extractor.get_folder_graph_statistics(folder_id)
                communities = []
                if stats.get('community_detection_viable', False):
                    communities = await self.graphrag_extractor.detect_communities(folder_id)
                
                return {
                    'already_processed': True,
                    'documents_count': episode_count,
                    'entities': stats.get('total_entities', 0),
                    'relationships': stats.get('total_relationships', 0),
                    'communities': len(communities),
                    'graphrag_ready': stats.get('graphrag_readiness') == 'READY',
                    'recommended_modes': stats.get('recommended_search_modes', [])
                }
                
        except Exception as e:
            return {
                'already_processed': False,
                'documents_count': 0,
                'entities': 0,
                'relationships': 0,
                'communities': 0,
                'graphrag_ready': False,
                'recommended_modes': []
            }

    async def ask_question(self, question: str, search_mode: str = "hybrid") -> Dict[str, Any]:
        if not self.is_initialized:
            raise RuntimeError("PDF Chatter not initialized. Call initialize() first.")
        if not self.folder_id:
            raise RuntimeError("No folder processed. Call process_folder() first.")
        if search_mode.lower() == "global":
            search_result = await self.graphrag_extractor.global_search(
                question, 
                self.folder_id, 
                max_communities=self.config.max_communities
            )
        elif search_mode.lower() == "local":
            search_result = await self.graphrag_extractor.local_search(
                question, 
                self.folder_id
            )
        else:
            search_result = await self.graphrag_extractor.hybrid_search(
                question, 
                self.folder_id
            )
        context = self._extract_context_for_gemini(search_result)
        ai_response = await self.gemini_caller.generate_response(question, context)
        return {
            'question': question,
            'answer': ai_response,
            'search_mode': search_result.get('search_mode', search_mode),
            'confidence': search_result.get('confidence', 0.0),
            'communities_used': search_result.get('communities_used', 0),
            'entities_found': search_result.get('entities_found', 0),
            'processing_time': search_result.get('processing_time', 0.0),
            'context_length': len(context),
            'graphrag_details': {
                'selected_communities': search_result.get('selected_communities', []),
                'global_confidence': search_result.get('global_confidence'),
                'local_confidence': search_result.get('local_confidence')
            }
        }

    def _extract_context_for_gemini(self, search_result: Dict[str, Any]) -> str:
        context_parts = []
        graphrag_answer = search_result.get('answer', '')
        if graphrag_answer:
            context_parts.append(graphrag_answer)
        search_mode = search_result.get('search_mode', '')
        if search_mode == SearchMode.GLOBAL.value:
            communities_used = search_result.get('communities_used', 0)
            if communities_used > 0:
                context_parts.append(f"Information synthesized from {communities_used} related concept clusters")
        elif search_mode == SearchMode.LOCAL.value:
            entities_found = search_result.get('entities_found', 0)
            if entities_found > 0:
                context_parts.append(f"Found {entities_found} specific references in the documents")
        full_context = "\n".join(context_parts)
        if len(full_context) > self.config.max_context_length:
            full_context = full_context[:self.config.max_context_length] + "\n\n[Context truncated...]"
        return full_context or "No relevant context found in the documents."

    async def cleanup(self):
        if self.graph_builder:
            await self.graph_builder.cleanup()
        if self.graphrag_extractor:
            await self.graphrag_extractor.close()

async def create_pdf_chatter(neo4j_uri: str, neo4j_user: str, neo4j_password: str, 
                           google_api_key: str, gemini_model: str = "gemini-2.5-pro") -> PDFChatter:
    config = PDFChatterConfig(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        google_api_key=google_api_key,
        gemini_model=gemini_model
    )
    chatter = PDFChatter(config)
    await chatter.initialize()
    return chatter

async def example_usage():
    """Demonstrates PDF chatter with intelligent file tracking"""
    config = PDFChatterConfig(
        neo4j_uri="bolt://localhost:7687",
        neo4j_user=os.getenv("NEO4J_USER"),
        neo4j_password=os.getenv("NEO4J_PASSWORD"),
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        gemini_model="gemini-2.5-pro"
    )
    
    chatter = PDFChatter(config)
    await chatter.initialize()
    
    folder_path = os.getenv("FOLDER_PATH")
    if not folder_path:
        print("❌ Please set FOLDER_PATH environment variable")
        return
    
    print(f"🔍 Processing folder: {folder_path}")
    
    # First time processing - will build knowledge graph
    process_result = await chatter.process_folder(folder_path)
    
    if process_result['status'] in ['success', 'already_processed']:
        print(f"✅ Ready for questions!")
        print(f"📊 Documents: {process_result['documents_processed']}")
        print(f"📊 Entities: {process_result['total_entities']}")
        print(f"📊 Relationships: {process_result['total_relationships']}")
        print(f"📊 Communities: {process_result['communities_detected']}")
        
        # Interactive question-answering
        questions = [
            "What are the main topics discussed in these documents?",
            "Can you summarize the key findings?",
            "What methodologies are mentioned?"
        ]
        
        for question in questions:
            print(f"\n❓ Q: {question}")
            result = await chatter.ask_question(question, search_mode="hybrid")
            print(f"💬 A: {result['answer']}")
            print(f"📈 Confidence: {result['confidence']:.2f}")
            print(f"🔍 Mode: {result['search_mode']}")
    else:
        print(f"❌ Failed to process folder: {process_result}")

    await chatter.cleanup()

if __name__ == "__main__":
    asyncio.run(example_usage())
