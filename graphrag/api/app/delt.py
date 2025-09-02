# api/app/deletion_service.py

import os
from typing import List, Dict, Optional, Set
import chromadb
from pathlib import Path

from . import settings


def _get_chroma_client():
    """Get Chroma client instance."""
    # Parse CHROMA_URL to get host and port
    chroma_url = settings.CHROMA_URL
    if chroma_url.startswith("http://"):
        url_parts = chroma_url[7:].split(":")
        host = url_parts[0]
        port = int(url_parts[1]) if len(url_parts) > 1 else 8000
    else:
        host = "localhost"
        port = 8000
    
    return chromadb.HttpClient(host=host, port=port)


def _get_graphrag_db():
    """Get GraphRAG database instance."""
    try:
        from .graphrag import get_graph_db
        return get_graph_db()
    except ImportError:
        return None


def delete_file_from_vector_db(file_id: str) -> Dict:
    """
    Delete all vector embeddings for a specific file from Chroma.
    
    Args:
        file_id: The file identifier to delete
        
    Returns:
        Dict with success status and details
    """
    result = {
        "success": False,
        "file_id": file_id,
        "deleted_chunks": 0,
        "deleted_from_cache": 0,
        "errors": []
    }
    
    try:
        client = _get_chroma_client()
        
        # Delete from main collection
        try:
            main_collection = client.get_collection(settings.COLLECTION)
            
            # Get all documents for this file
            results = main_collection.get(
                where={"file_id": file_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Delete the documents
                main_collection.delete(
                    where={"file_id": file_id}
                )
                result["deleted_chunks"] = len(results["ids"])
                print(f"[VectorDB] Deleted {len(results['ids'])} chunks for file {file_id} from main collection")
        except Exception as e:
            result["errors"].append(f"Main collection error: {str(e)}")
            print(f"[VectorDB] Error deleting from main collection: {e}")
        
        # Delete from cache collection
        try:
            cache_collection = client.get_collection(f"{settings.COLLECTION}_cache")
            
            # Cache might use different metadata structure, so try both approaches
            cache_results = cache_collection.get(
                where={"file_id": file_id},
                include=["metadatas"]
            )
            
            if cache_results["ids"]:
                cache_collection.delete(
                    where={"file_id": file_id}
                )
                result["deleted_from_cache"] = len(cache_results["ids"])
                print(f"[VectorDB] Deleted {len(cache_results['ids'])} items for file {file_id} from cache collection")
        except Exception as e:
            # Cache collection might not exist or have different structure - this is OK
            if "does not exist" in str(e):
                print(f"[VectorDB] Cache collection does not exist - skipping cache deletion")
            else:
                result["errors"].append(f"Cache collection error: {str(e)}")
                print(f"[VectorDB] Error deleting from cache collection: {e}")
        
        result["success"] = result["deleted_chunks"] > 0 or result["deleted_from_cache"] > 0
        
    except Exception as e:
        result["errors"].append(f"Vector DB error: {str(e)}")
        print(f"[VectorDB] Failed to delete file {file_id}: {e}")
    
    return result


def cleanup_orphaned_entities() -> Dict:
    """
    Clean up entities that have no mention relationships from any chunks.
    This should be called after deleting files to clean up orphaned entities.
    """
    result = {
        "success": False,
        "deleted_entities": 0,
        "errors": []
    }
    
    graphrag_db = _get_graphrag_db()
    if not graphrag_db or not graphrag_db.is_connected():
        result["errors"].append("GraphRAG not available or not connected")
        return result
    
    try:
        with graphrag_db.driver.session() as session:
            # Delete all entities that have no mention relationships
            cleanup_query = """
            MATCH (e:Entity)
            WHERE NOT EXISTS((:Chunk)-[:MENTIONS]->(e))
            DELETE e
            RETURN count(e) as deleted_entities
            """
            cleanup_result = session.run(cleanup_query).single()
            result["deleted_entities"] = cleanup_result["deleted_entities"] if cleanup_result else 0
            result["success"] = True
            
            print(f"[GraphRAG] Cleaned up {result['deleted_entities']} orphaned entities")
            
    except Exception as e:
        result["errors"].append(f"GraphRAG error: {str(e)}")
        print(f"[GraphRAG] Failed to cleanup orphaned entities: {e}")
    
    return result


def delete_file_from_knowledge_graph(file_id: str) -> Dict:
    """
    Delete all entities and relations for a specific file from Neo4j knowledge graph.
    
    Args:
        file_id: The file identifier to delete
        
    Returns:
        Dict with success status and details
    """
    result = {
        "success": False,
        "file_id": file_id,
        "deleted_entities": 0,
        "deleted_relations": 0,
        "deleted_chunks": 0,
        "deleted_documents": 0,
        "errors": []
    }
    
    graphrag_db = _get_graphrag_db()
    if not graphrag_db or not graphrag_db.is_connected():
        result["errors"].append("GraphRAG not available or not connected")
        return result
    
    try:
        with graphrag_db.driver.session() as session:
            # Step 1: Delete relations that are linked to this file
            delete_relations_query = """
            MATCH ()-[r:RELATES]->()
            WHERE r.file_id = $file_id
            DELETE r
            RETURN count(r) as deleted_relations
            """
            relations_result = session.run(delete_relations_query, file_id=file_id).single()
            result["deleted_relations"] = relations_result["deleted_relations"] if relations_result else 0
            
            # Step 2: Delete mention relationships for this file's chunks first
            delete_mentions_query = """
            MATCH (c:Chunk {file_id: $file_id})-[m:MENTIONS]->(e:Entity)
            DELETE m
            RETURN count(m) as deleted_mentions
            """
            mentions_result = session.run(delete_mentions_query, file_id=file_id).single()
            deleted_mentions = mentions_result["deleted_mentions"] if mentions_result else 0
            
            # Step 3: Delete HAS_CHUNK relationships for this file's document
            delete_has_chunk_query = """
            MATCH (d:Document {file_id: $file_id})-[h:HAS_CHUNK]->(c:Chunk {file_id: $file_id})
            DELETE h
            RETURN count(h) as deleted_has_chunk
            """
            session.run(delete_has_chunk_query, file_id=file_id)
            
            # Step 4: Now delete chunks for this file (no relationships should remain)
            delete_chunks_query = """
            MATCH (c:Chunk {file_id: $file_id})
            DELETE c
            RETURN count(c) as deleted_chunks
            """
            chunks_result = session.run(delete_chunks_query, file_id=file_id).single()
            result["deleted_chunks"] = chunks_result["deleted_chunks"] if chunks_result else 0
            
            # Step 5: Delete document node if it has no more chunks
            delete_document_query = """
            MATCH (d:Document {file_id: $file_id})
            WHERE NOT EXISTS((d)-[:HAS_CHUNK]->())
            DELETE d
            RETURN count(d) as deleted_documents
            """
            doc_result = session.run(delete_document_query, file_id=file_id).single()
            result["deleted_documents"] = doc_result["deleted_documents"] if doc_result else 0
            
            result["success"] = True
            print(f"[GraphRAG] Deleted {result['deleted_relations']} relations, {result['deleted_chunks']} chunks, and {result['deleted_documents']} documents for file {file_id}")
        
        # Step 6: Clean up orphaned entities (do this outside the session to avoid conflicts)
        cleanup_result = cleanup_orphaned_entities()
        result["deleted_entities"] = cleanup_result["deleted_entities"]
        if cleanup_result["errors"]:
            result["errors"].extend(cleanup_result["errors"])
                
    except Exception as e:
        result["errors"].append(f"GraphRAG error: {str(e)}")
        print(f"[GraphRAG] Failed to delete file {file_id}: {e}")
    
    return result


def delete_folder_from_vector_db(folder_id: str) -> Dict:
    """
    Delete all vector embeddings for files in a specific folder from Chroma.
    
    Args:
        folder_id: The folder identifier to delete
        
    Returns:
        Dict with success status and details
    """
    result = {
        "success": False,
        "folder_id": folder_id,
        "affected_files": [],
        "total_deleted_chunks": 0,
        "total_deleted_from_cache": 0,
        "errors": []
    }
    
    try:
        client = _get_chroma_client()
        
        # Get all files in this folder from main collection
        try:
            main_collection = client.get_collection(settings.COLLECTION)
            
            # Get all documents for this folder
            results = main_collection.get(
                where={"folder_id": folder_id},
                include=["metadatas"]
            )
            
            if results["ids"]:
                # Extract unique file_ids
                file_ids = set()
                for metadata in results["metadatas"]:
                    if metadata and "file_id" in metadata:
                        file_ids.add(metadata["file_id"])
                
                result["affected_files"] = list(file_ids)
                
                # Delete all documents in this folder
                main_collection.delete(
                    where={"folder_id": folder_id}
                )
                result["total_deleted_chunks"] = len(results["ids"])
                print(f"[VectorDB] Deleted {len(results['ids'])} chunks for folder {folder_id} from main collection")
        except Exception as e:
            result["errors"].append(f"Main collection error: {str(e)}")
            print(f"[VectorDB] Error deleting from main collection: {e}")
        
        # Delete from cache collection
        try:
            cache_collection = client.get_collection(f"{settings.COLLECTION}_cache")
            
            cache_results = cache_collection.get(
                where={"folder_id": folder_id},
                include=["metadatas"]
            )
            
            if cache_results["ids"]:
                cache_collection.delete(
                    where={"folder_id": folder_id}
                )
                result["total_deleted_from_cache"] = len(cache_results["ids"])
                print(f"[VectorDB] Deleted {len(cache_results['ids'])} items for folder {folder_id} from cache collection")
        except Exception as e:
            # Cache collection might not exist or have different structure - this is OK
            if "does not exist" in str(e):
                print(f"[VectorDB] Cache collection does not exist - skipping cache deletion")
            else:
                result["errors"].append(f"Cache collection error: {str(e)}")
                print(f"[VectorDB] Error deleting from cache collection: {e}")
        
        result["success"] = result["total_deleted_chunks"] > 0 or result["total_deleted_from_cache"] > 0
        
    except Exception as e:
        result["errors"].append(f"Vector DB error: {str(e)}")
        print(f"[VectorDB] Failed to delete folder {folder_id}: {e}")
    
    return result


def delete_folder_from_knowledge_graph_with_files(folder_id: str, file_ids: List[str]) -> Dict:
    """
    Delete all entities and relations for files in a specific folder from Neo4j knowledge graph.
    Uses a pre-determined list of file_ids instead of querying the vector database.
    
    Args:
        folder_id: The folder identifier to delete
        file_ids: List of file_ids in this folder
        
    Returns:
        Dict with success status and details
    """
    result = {
        "success": False,
        "folder_id": folder_id,
        "affected_files": file_ids,
        "total_deleted_entities": 0,
        "total_deleted_relations": 0,
        "total_deleted_chunks": 0,
        "total_deleted_documents": 0,
        "errors": []
    }
    
    graphrag_db = _get_graphrag_db()
    if not graphrag_db or not graphrag_db.is_connected():
        result["errors"].append("GraphRAG not available or not connected")
        return result
    
    if not file_ids:
        print(f"[GraphRAG] No file_ids provided for folder {folder_id}")
        result["success"] = True
        return result
    
    try:
        with graphrag_db.driver.session() as session:
            total_relations = 0
            total_entities = 0
            total_chunks = 0
            total_documents = 0
            
            for file_id in file_ids:
                # Step 1: Delete relations for this file
                delete_file_relations_query = """
                MATCH ()-[r:RELATES]->()
                WHERE r.file_id = $file_id
                DELETE r
                RETURN count(r) as deleted_relations
                """
                rel_result = session.run(delete_file_relations_query, file_id=file_id).single()
                total_relations += rel_result["deleted_relations"] if rel_result else 0
                
                # Step 3: Delete all relationships for chunks of this file
                delete_chunk_relationships_query = """
                MATCH (c:Chunk {file_id: $file_id})
                OPTIONAL MATCH (c)-[r]-()
                DELETE r
                RETURN count(r) as deleted_chunk_rels
                """
                session.run(delete_chunk_relationships_query, file_id=file_id)
                
                # Step 4: Now delete chunks for this file (safe because all relationships are gone)
                delete_chunks_query = """
                MATCH (c:Chunk {file_id: $file_id})
                DELETE c
                RETURN count(c) as deleted_chunks
                """
                chunks_result = session.run(delete_chunks_query, file_id=file_id).single()
                total_chunks += chunks_result["deleted_chunks"] if chunks_result else 0
                
                # Step 5: Delete document if no more chunks
                delete_document_query = """
                MATCH (d:Document {file_id: $file_id})
                WHERE NOT EXISTS((d)-[:HAS_CHUNK]->(:Chunk))
                DELETE d
                RETURN count(d) as deleted_documents
                """
                doc_result = session.run(delete_document_query, file_id=file_id).single()
                total_documents += doc_result["deleted_documents"] if doc_result else 0
            
            result["total_deleted_entities"] = total_entities
            result["total_deleted_relations"] = total_relations
            result["total_deleted_chunks"] = total_chunks
            result["total_deleted_documents"] = total_documents
            result["success"] = True
            
            print(f"[GraphRAG] Deleted {total_relations} relations, {total_chunks} chunks, and {total_documents} documents for folder {folder_id}")
        
        # Clean up orphaned entities after all deletions
        cleanup_result = cleanup_orphaned_entities()
        result["total_deleted_entities"] = cleanup_result["deleted_entities"]
        if cleanup_result["errors"]:
            result["errors"].extend(cleanup_result["errors"])
                    
    except Exception as e:
        result["errors"].append(f"GraphRAG error: {str(e)}")
        print(f"[GraphRAG] Failed to delete folder {folder_id}: {e}")
    
    return result


def delete_folder_from_knowledge_graph(folder_id: str) -> Dict:
    """
    Delete all entities and relations for files in a specific folder from Neo4j knowledge graph.
    
    Args:
        folder_id: The folder identifier to delete
        
    Returns:
        Dict with success status and details
    """
    result = {
        "success": False,
        "folder_id": folder_id,
        "affected_files": [],
        "total_deleted_entities": 0,
        "total_deleted_relations": 0,
        "total_deleted_chunks": 0,
        "total_deleted_documents": 0,
        "errors": []
    }
    
    graphrag_db = _get_graphrag_db()
    if not graphrag_db or not graphrag_db.is_connected():
        result["errors"].append("GraphRAG not available or not connected")
        return result
    
    try:
        with graphrag_db.driver.session() as session:
            # Step 1: Get all file_ids that have chunks with this folder_id (from vector DB metadata)
            # We need to get this from the chunks that were created during ingestion
            files_query = """
            MATCH (c:Chunk)
            WHERE c.file_id IS NOT NULL
            RETURN DISTINCT c.file_id as file_id
            """
            all_files_result = session.run(files_query)
            
            # We need to check vector DB for folder associations since folder_id isn't stored in Neo4j chunks
            # Let's first get relations with file_id property
            files_with_relations_query = """
            MATCH ()-[r:RELATES]->()
            WHERE r.file_id IS NOT NULL
            RETURN DISTINCT r.file_id as file_id
            """
            
            # For now, let's work with what we have and delete based on chunk file_id pattern
            # This is a limitation - we should store folder_id in Neo4j chunks during ingestion
            
            # Step 2: Delete relations where file_id pattern might match folder structure
            # This is imperfect but better than nothing
            delete_relations_query = """
            MATCH ()-[r:RELATES]->()
            WHERE r.file_id IS NOT NULL
            DELETE r
            RETURN count(r) as deleted_relations, collect(DISTINCT r.file_id) as affected_files
            """
            
            # For safety, let's be more specific and only work if we can identify files
            # We'll implement a conservative approach here
            
            # Get chunks and try to match them to the folder from vector DB
            client = _get_chroma_client()
            try:
                main_collection = client.get_collection(settings.COLLECTION)
                # Get all file_ids in this folder from vector DB
                vector_results = main_collection.get(
                    where={"folder_id": folder_id},
                    include=["metadatas"]
                )
                
                if vector_results["ids"]:
                    # Extract unique file_ids from vector DB
                    file_ids_in_folder = set()
                    for metadata in vector_results["metadatas"]:
                        if metadata and "file_id" in metadata:
                            file_ids_in_folder.add(metadata["file_id"])
                    
                    result["affected_files"] = list(file_ids_in_folder)
                    
                    # Now delete from Neo4j based on these file_ids
                    total_relations = 0
                    total_entities = 0
                    total_chunks = 0
                    total_documents = 0
                    
                    for file_id in file_ids_in_folder:
                        # Delete relations for this file
                        delete_file_relations_query = """
                        MATCH ()-[r:RELATES]->()
                        WHERE r.file_id = $file_id
                        DELETE r
                        RETURN count(r) as deleted_relations
                        """
                        rel_result = session.run(delete_file_relations_query, file_id=file_id).single()
                        total_relations += rel_result["deleted_relations"] if rel_result else 0
                        
                        # Find entities mentioned by chunks of this file
                        entities_query = """
                        MATCH (c:Chunk {file_id: $file_id})-[:MENTIONS]->(e:Entity)
                        RETURN DISTINCT e.name as entity_name
                        """
                        entities_to_check = []
                        entities_result = session.run(entities_query, file_id=file_id)
                        for record in entities_result:
                            entities_to_check.append(record["entity_name"])
                        
                        # Delete mention relationships for this file's chunks
                        delete_mentions_query = """
                        MATCH (c:Chunk {file_id: $file_id})-[m:MENTIONS]->(e:Entity)
                        DELETE m
                        RETURN count(m) as deleted_mentions
                        """
                        session.run(delete_mentions_query, file_id=file_id)
                        
                        # Delete orphaned entities
                        delete_orphaned_entities_query = """
                        MATCH (e:Entity)
                        WHERE NOT EXISTS((:Chunk)-[:MENTIONS]->(e))
                        DELETE e
                        RETURN count(e) as deleted_entities
                        """
                        orphan_result = session.run(delete_orphaned_entities_query).single()
                        total_entities += orphan_result["deleted_entities"] if orphan_result else 0
                        
                        # Delete chunks for this file
                        delete_chunks_query = """
                        MATCH (c:Chunk {file_id: $file_id})
                        DELETE c
                        RETURN count(c) as deleted_chunks
                        """
                        chunks_result = session.run(delete_chunks_query, file_id=file_id).single()
                        total_chunks += chunks_result["deleted_chunks"] if chunks_result else 0
                        
                        # Delete document if no more chunks
                        delete_document_query = """
                        MATCH (d:Document {file_id: $file_id})
                        WHERE NOT EXISTS((d)-[:HAS_CHUNK]->(:Chunk))
                        DELETE d
                        RETURN count(d) as deleted_documents
                        """
                        doc_result = session.run(delete_document_query, file_id=file_id).single()
                        total_documents += doc_result["deleted_documents"] if doc_result else 0
                    
                    result["total_deleted_entities"] = total_entities
                    result["total_deleted_relations"] = total_relations
                    result["total_deleted_chunks"] = total_chunks
                    result["total_deleted_documents"] = total_documents
                    result["success"] = True
                    
                    print(f"[GraphRAG] Deleted {total_entities} entities, {total_relations} relations, {total_chunks} chunks, and {total_documents} documents for folder {folder_id}")
                else:
                    print(f"[GraphRAG] No files found in folder {folder_id}")
                    result["success"] = True
                    
            except Exception as ve:
                result["errors"].append(f"Vector DB error while finding folder files: {str(ve)}")
                print(f"[GraphRAG] Could not get folder files from vector DB: {ve}")
                
    except Exception as e:
        result["errors"].append(f"GraphRAG error: {str(e)}")
        print(f"[GraphRAG] Failed to delete folder {folder_id}: {e}")
    
    return result


def delete_file_completely(file_id: str) -> Dict:
    """
    Delete a file completely from both vector database and knowledge graph.
    
    Args:
        file_id: The file identifier to delete
        
    Returns:
        Dict with comprehensive deletion results
    """
    result = {
        "file_id": file_id,
        "vector_db": {"success": False},
        "knowledge_graph": {"success": False},
        "overall_success": False
    }
    
    # Delete from vector database
    vector_result = delete_file_from_vector_db(file_id)
    result["vector_db"] = vector_result
    
    # Delete from knowledge graph
    graph_result = delete_file_from_knowledge_graph(file_id)
    result["knowledge_graph"] = graph_result
    
    # Overall success if at least one deletion succeeded or nothing was found to delete
    result["overall_success"] = vector_result["success"] or graph_result["success"]
    
    return result


def delete_folder_completely(folder_id: str) -> Dict:
    """
    Delete a folder completely from both vector database and knowledge graph.
    
    Args:
        folder_id: The folder identifier to delete
        
    Returns:
        Dict with comprehensive deletion results
    """
    result = {
        "folder_id": folder_id,
        "vector_db": {"success": False},
        "knowledge_graph": {"success": False},
        "overall_success": False,
        "affected_files": []
    }
    
    # Step 1: First, get the list of file_ids in this folder before any deletion
    try:
        client = _get_chroma_client()
        main_collection = client.get_collection(settings.COLLECTION)
        vector_results = main_collection.get(
            where={"folder_id": folder_id},
            include=["metadatas"]
        )
        
        if vector_results["ids"]:
            file_ids_in_folder = set()
            for metadata in vector_results["metadatas"]:
                if metadata and "file_id" in metadata:
                    file_ids_in_folder.add(metadata["file_id"])
            result["affected_files"] = list(file_ids_in_folder)
            print(f"[Deletion] Found {len(file_ids_in_folder)} files in folder {folder_id}: {list(file_ids_in_folder)}")
        else:
            print(f"[Deletion] No files found in folder {folder_id}")
            
    except Exception as e:
        print(f"[Deletion] Error getting file list for folder {folder_id}: {e}")
    
    # Step 2: Delete from vector database
    vector_result = delete_folder_from_vector_db(folder_id)
    result["vector_db"] = vector_result
    
    # Step 3: Delete from knowledge graph (pass the file_ids we found)
    graph_result = delete_folder_from_knowledge_graph_with_files(folder_id, result["affected_files"])
    result["knowledge_graph"] = graph_result
    
    # Overall success if at least one deletion succeeded
    result["overall_success"] = vector_result["success"] or graph_result["success"]
    
    return result


def get_file_deletion_preview(file_id: str) -> Dict:
    """
    Preview what would be deleted for a specific file without actually deleting.
    
    Args:
        file_id: The file identifier to preview
        
    Returns:
        Dict with preview information
    """
    preview = {
        "file_id": file_id,
        "vector_db": {
            "chunks_to_delete": 0,
            "cache_items_to_delete": 0,
            "exists": False
        },
        "knowledge_graph": {
            "entities_to_delete": 0,
            "relations_to_delete": 0,
            "chunks_to_delete": 0,
            "documents_to_delete": 0,
            "exists": False
        },
        "errors": []
    }
    
    # Check vector database
    try:
        client = _get_chroma_client()
        
        # Check main collection
        try:
            main_collection = client.get_collection(settings.COLLECTION)
            results = main_collection.get(
                where={"file_id": file_id},
                include=["metadatas"]
            )
            preview["vector_db"]["chunks_to_delete"] = len(results["ids"])
            preview["vector_db"]["exists"] = len(results["ids"]) > 0
        except Exception as e:
            preview["errors"].append(f"Vector DB main collection error: {str(e)}")
        
        # Check cache collection
        try:
            cache_collection = client.get_collection(f"{settings.COLLECTION}_cache")
            cache_results = cache_collection.get(
                where={"file_id": file_id},
                include=["metadatas"]
            )
            preview["vector_db"]["cache_items_to_delete"] = len(cache_results["ids"])
        except Exception as e:
            # Cache collection might not exist - this is OK for preview
            if "does not exist" not in str(e):
                preview["errors"].append(f"Vector DB cache collection error: {str(e)}")
            
    except Exception as e:
        preview["errors"].append(f"Vector DB error: {str(e)}")
    
    # Check knowledge graph
    graphrag_db = _get_graphrag_db()
    if graphrag_db and graphrag_db.is_connected():
        try:
            with graphrag_db.driver.session() as session:
                # Count relations for this file
                relations_query = """
                MATCH ()-[r:RELATES]->()
                WHERE r.file_id = $file_id
                RETURN count(r) as relations
                """
                relations_result = session.run(relations_query, file_id=file_id).single()
                relations_count = relations_result["relations"] if relations_result else 0
                
                # Count entities mentioned by chunks of this file
                entities_query = """
                MATCH (c:Chunk {file_id: $file_id})-[:MENTIONS]->(e:Entity)
                RETURN count(DISTINCT e) as entities
                """
                entities_result = session.run(entities_query, file_id=file_id).single()
                entities_count = entities_result["entities"] if entities_result else 0
                
                # Count chunks for this file
                chunks_query = """
                MATCH (c:Chunk {file_id: $file_id})
                RETURN count(c) as chunks
                """
                chunks_result = session.run(chunks_query, file_id=file_id).single()
                chunks_count = chunks_result["chunks"] if chunks_result else 0
                
                # Count documents for this file
                docs_query = """
                MATCH (d:Document {file_id: $file_id})
                RETURN count(d) as documents
                """
                docs_result = session.run(docs_query, file_id=file_id).single()
                docs_count = docs_result["documents"] if docs_result else 0
                
                preview["knowledge_graph"]["entities_to_delete"] = entities_count
                preview["knowledge_graph"]["relations_to_delete"] = relations_count
                preview["knowledge_graph"]["chunks_to_delete"] = chunks_count
                preview["knowledge_graph"]["documents_to_delete"] = docs_count
                preview["knowledge_graph"]["exists"] = entities_count > 0 or relations_count > 0 or chunks_count > 0
                    
        except Exception as e:
            preview["errors"].append(f"GraphRAG error: {str(e)}")
    else:
        preview["errors"].append("GraphRAG not available")
    
    return preview


def get_folder_deletion_preview(folder_id: str) -> Dict:
    """
    Preview what would be deleted for a specific folder without actually deleting.
    
    Args:
        folder_id: The folder identifier to preview
        
    Returns:
        Dict with preview information
    """
    preview = {
        "folder_id": folder_id,
        "affected_files": [],
        "vector_db": {
            "chunks_to_delete": 0,
            "cache_items_to_delete": 0,
            "exists": False
        },
        "knowledge_graph": {
            "entities_to_delete": 0,
            "relations_to_delete": 0,
            "chunks_to_delete": 0,
            "documents_to_delete": 0,
            "exists": False
        },
        "errors": []
    }
    
    # Check vector database
    try:
        client = _get_chroma_client()
        
        # Check main collection
        try:
            main_collection = client.get_collection(settings.COLLECTION)
            results = main_collection.get(
                where={"folder_id": folder_id},
                include=["metadatas"]
            )
            preview["vector_db"]["chunks_to_delete"] = len(results["ids"])
            preview["vector_db"]["exists"] = len(results["ids"]) > 0
            
            # Extract file IDs
            file_ids = set()
            for metadata in results["metadatas"]:
                if metadata and "file_id" in metadata:
                    file_ids.add(metadata["file_id"])
            preview["affected_files"] = list(file_ids)
            
        except Exception as e:
            preview["errors"].append(f"Vector DB main collection error: {str(e)}")
        
        # Check cache collection
        try:
            cache_collection = client.get_collection(f"{settings.COLLECTION}_cache")
            cache_results = cache_collection.get(
                where={"folder_id": folder_id},
                include=["metadatas"]
            )
            preview["vector_db"]["cache_items_to_delete"] = len(cache_results["ids"])
        except Exception as e:
            # Cache collection might not exist - this is OK for preview
            if "does not exist" not in str(e):
                preview["errors"].append(f"Vector DB cache collection error: {str(e)}")
            
    except Exception as e:
        preview["errors"].append(f"Vector DB error: {str(e)}")
    
    # Check knowledge graph
    graphrag_db = _get_graphrag_db()
    if graphrag_db and graphrag_db.is_connected():
        try:
            with graphrag_db.driver.session() as session:
                # For each file in the folder, count what would be deleted
                total_entities = 0
                total_relations = 0
                total_chunks = 0
                total_documents = 0
                
                for file_id in preview["affected_files"]:
                    # Count relations for this file
                    relations_query = """
                    MATCH ()-[r:RELATES]->()
                    WHERE r.file_id = $file_id
                    RETURN count(r) as relations
                    """
                    relations_result = session.run(relations_query, file_id=file_id).single()
                    total_relations += relations_result["relations"] if relations_result else 0
                    
                    # Count entities mentioned by chunks of this file
                    entities_query = """
                    MATCH (c:Chunk {file_id: $file_id})-[:MENTIONS]->(e:Entity)
                    RETURN count(DISTINCT e) as entities
                    """
                    entities_result = session.run(entities_query, file_id=file_id).single()
                    total_entities += entities_result["entities"] if entities_result else 0
                    
                    # Count chunks for this file
                    chunks_query = """
                    MATCH (c:Chunk {file_id: $file_id})
                    RETURN count(c) as chunks
                    """
                    chunks_result = session.run(chunks_query, file_id=file_id).single()
                    total_chunks += chunks_result["chunks"] if chunks_result else 0
                    
                    # Count documents for this file
                    docs_query = """
                    MATCH (d:Document {file_id: $file_id})
                    RETURN count(d) as documents
                    """
                    docs_result = session.run(docs_query, file_id=file_id).single()
                    total_documents += docs_result["documents"] if docs_result else 0
                
                preview["knowledge_graph"]["entities_to_delete"] = total_entities
                preview["knowledge_graph"]["relations_to_delete"] = total_relations
                preview["knowledge_graph"]["chunks_to_delete"] = total_chunks
                preview["knowledge_graph"]["documents_to_delete"] = total_documents
                preview["knowledge_graph"]["exists"] = total_entities > 0 or total_relations > 0 or total_chunks > 0
                    
        except Exception as e:
            preview["errors"].append(f"GraphRAG error: {str(e)}")
    else:
        preview["errors"].append("GraphRAG not available")
    
    return preview
