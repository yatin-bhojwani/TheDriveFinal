# api/app/bulk_deletion_service.py

from typing import List, Dict, Set
import chromadb
from . import settings
from .delt import _get_chroma_client, _get_graphrag_db, delete_file_completely


def delete_multiple_files(file_ids: List[str]) -> Dict:
    """
    Delete multiple files at once.
    
    Args:
        file_ids: List of file identifiers to delete
        
    Returns:
        Dict with results for each file
    """
    results = {
        "total_files": len(file_ids),
        "successful_deletions": 0,
        "failed_deletions": 0,
        "results": {}
    }
    
    for file_id in file_ids:
        try:
            result = delete_file_completely(file_id)
            results["results"][file_id] = result
            
            if result["overall_success"]:
                results["successful_deletions"] += 1
            else:
                results["failed_deletions"] += 1
                
        except Exception as e:
            results["results"][file_id] = {
                "file_id": file_id,
                "overall_success": False,
                "error": str(e)
            }
            results["failed_deletions"] += 1
    
    return results


def get_orphaned_data_preview() -> Dict:
    """
    Find data that might be orphaned (exists in DB but file may have been deleted from filesystem).
    This is useful for cleaning up after external file deletions.
    
    Returns:
        Dict with orphaned data information
    """
    orphaned_data = {
        "vector_db": {
            "unique_file_ids": set(),
            "unique_folder_ids": set(),
            "total_chunks": 0
        },
        "knowledge_graph": {
            "unique_file_ids": set(),
            "unique_folder_ids": set(),
            "total_entities": 0,
            "total_relations": 0
        },
        "errors": []
    }
    
    # Check vector database
    try:
        client = _get_chroma_client()
        
        try:
            main_collection = client.get_collection(settings.COLLECTION)
            results = main_collection.get(include=["metadatas"])
            
            orphaned_data["vector_db"]["total_chunks"] = len(results["ids"])
            
            for metadata in results["metadatas"]:
                if metadata:
                    if "file_id" in metadata:
                        orphaned_data["vector_db"]["unique_file_ids"].add(metadata["file_id"])
                    if "folder_id" in metadata:
                        orphaned_data["vector_db"]["unique_folder_ids"].add(metadata["folder_id"])
                        
        except Exception as e:
            orphaned_data["errors"].append(f"Vector DB error: {str(e)}")
            
    except Exception as e:
        orphaned_data["errors"].append(f"Vector DB connection error: {str(e)}")
    
    # Check knowledge graph
    graphrag_db = _get_graphrag_db()
    if graphrag_db and graphrag_db.is_connected():
        try:
            with graphrag_db.driver.session() as session:
                # Get all entities and relations
                query = """
                MATCH (e:Entity)
                OPTIONAL MATCH (e)-[r:RELATED_TO]-()
                RETURN DISTINCT e.file_id as file_id, e.folder_id as folder_id, count(r) as relation_count
                """
                result = session.run(query)
                
                total_entities = 0
                total_relations = 0
                
                for record in result:
                    total_entities += 1
                    total_relations += record["relation_count"] if record["relation_count"] else 0
                    
                    if record["file_id"]:
                        orphaned_data["knowledge_graph"]["unique_file_ids"].add(record["file_id"])
                    if record["folder_id"]:
                        orphaned_data["knowledge_graph"]["unique_folder_ids"].add(record["folder_id"])
                
                orphaned_data["knowledge_graph"]["total_entities"] = total_entities
                orphaned_data["knowledge_graph"]["total_relations"] = total_relations // 2  # Relations are bidirectional
                
        except Exception as e:
            orphaned_data["errors"].append(f"GraphRAG error: {str(e)}")
    else:
        orphaned_data["errors"].append("GraphRAG not available")
    
    # Convert sets to lists for JSON serialization
    orphaned_data["vector_db"]["unique_file_ids"] = list(orphaned_data["vector_db"]["unique_file_ids"])
    orphaned_data["vector_db"]["unique_folder_ids"] = list(orphaned_data["vector_db"]["unique_folder_ids"])
    orphaned_data["knowledge_graph"]["unique_file_ids"] = list(orphaned_data["knowledge_graph"]["unique_file_ids"])
    orphaned_data["knowledge_graph"]["unique_folder_ids"] = list(orphaned_data["knowledge_graph"]["unique_folder_ids"])
    
    return orphaned_data
