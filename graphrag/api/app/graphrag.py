# api/app/graphrag.py

import os
import json
import time
import re
from typing import List, Dict, Optional, Tuple
from neo4j import GraphDatabase

# Neo4j Configuration
NEO4J_URL = os.getenv("NEO4J_URL", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password123")

MAX_FACTS = 120
MAX_HOPS = 2

# Confidence thresholds for filtering entities and relations
ENTITY_CONFIDENCE_THRESHOLD = 0.65
RELATION_CONFIDENCE_THRESHOLD = 0.65
ENABLE_EMBEDDING_VALIDATION = True  # Re-check entities against chunk text using embeddings


def get_llm():
    """Get LLM instance - avoiding circular import."""
    from langchain_google_genai import ChatGoogleGenerativeAI
    model_name = os.getenv("MODEL_NAME", "gemini-1.5-pro")
    api_key = os.getenv("GEMINI_API_KEY")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.2
    )


class Neo4jGraphRAG:
    def __init__(self):
        self.driver = None
        self._connect()

    def _connect(self):
        """Initialize Neo4j connection."""
        try:
            self.driver = GraphDatabase.driver(NEO4J_URL, auth=(NEO4J_USER, NEO4J_PASSWORD))
            # Test connection
            with self.driver.session() as session:
                session.run("RETURN 1")
            print(f"[GraphRAG] Connected to Neo4j at {NEO4J_URL}")
            self._create_constraints()
        except Exception as e:
            print(f"[GraphRAG] Failed to connect to Neo4j: {e}")
            self.driver = None

    def _create_constraints(self):
        """Create necessary constraints and indexes."""
        with self.driver.session() as session:
            try:
                session.run("CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE")
                session.run("CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)")
                session.run("CREATE INDEX document_file_idx IF NOT EXISTS FOR (d:Document) ON (d.file_id)")
                session.run("CREATE INDEX chunk_file_idx IF NOT EXISTS FOR (c:Chunk) ON (c.file_id)")
                print("[GraphRAG] Neo4j constraints and indexes created")
            except Exception as e:
                print(f"[GraphRAG] Warning: Could not create constraints/indexes: {e}")

    def close(self):
        """Close Neo4j connection."""
        if self.driver:
            self.driver.close()

    def is_connected(self):
        """Check if Neo4j is connected."""
        return self.driver is not None


# Global instance - lazy initialization
_graph_db = None

def get_graph_db():
    """Get graph database instance with lazy initialization."""
    global _graph_db
    if _graph_db is None:
        _graph_db = Neo4jGraphRAG()
    return _graph_db


# ========== Enhanced Entity and Relation Extraction ==========

ENTITY_EXTRACTION_PROMPT = """
You are a highly-specialized entity extraction model. Your task is to extract all salient entities from the provided text with confidence scores.

**Instructions:**
1.  **Strict Adherence:** Analyze the entire text or chunks and extract every important entity. An entity is a named or conceptual item.
2.  **Canonical Form:** Present the entity name in its most standard, canonical form (e.g., "AI" should be "Artificial Intelligence").
3.  **Type Specificity:** Assign the most specific entity type from a predefined set. Your available types are: **PERSON, ORGANIZATION, LOCATION, PRODUCT, TECHNOLOGY, EVENT, CONCEPT, TIME, NUMBER, MISCELLANEOUS**. If a more suitable, specific type exists, use it and add it to the list.
4.  **Attribute Richness:** For each entity, identify all relevant attributes and properties. These can include descriptions, dates, affiliations, or specifications. List these as a structured JSON object.
5.  **Confidence Score:** For each entity, provide a confidence score (0.0-1.0) based on:
     - How clearly the entity is mentioned in the text (0.9-1.0 for explicit mentions)
     - How relevant the entity is to the main topic (0.7-0.9 for highly relevant)
     - How much supporting context exists (0.5-0.7 for implied or context-dependent)
     - Avoid hallucinated or weakly supported entities (below 0.5)
6.  **Format:** Your output must be a JSON array of objects. Each object must have four keys: `name` (string), `type` (string), `attributes` (JSON object), and `confidence` (float between 0.0 and 1.0).
7.  **Constraint:** Return all significant entities. If the text has few entities, return fewer; if it has many, return more. Do not arbitrarily limit your response.
8.  **Entity Types:** At times a particular file may be an assignment, so entity types may sound confusing. Instead of defining a strict set of types, focus on capturing the essence of the entity like what topic it is about, then mapping it to the entity.

Return ONLY a JSON array of objects with 'name', 'type', 'attributes', and 'confidence' fields.

Text: {text}
"""

RELATION_EXTRACTION_PROMPT = """
You are a highly-specialized relationship extraction model. Your task is to identify and extract the most meaningful relationships between entities in the provided text with confidence scores.

**Instructions:**
1.  **Entity-First:** All relationships must be defined as a connection between two or more distinct entities.
2.  **Standard Predicates:** Use a descriptive, canonical predicate (e.g., `CREATED`, `WORKS_AT`, `LOCATED_IN`, `IS_A`). Use past tense for completed actions and present tense for ongoing or static relationships. Be as specific as possible.
3.  **Contextual Detail:** For each relationship, provide a concise `context` string that explains why the relationship exists, citing key phrases or sentences from the text. This is crucial for verifying the relationship's validity.
4.  **Implicit Relations:** Identify and extract relationships that are not explicitly stated but are strongly implied by the text.
5.  **Confidence Score:** For each relationship, provide a confidence score (0.0-1.0) based on:
     - How explicitly the relationship is stated (0.9-1.0 for direct statements)
     - How strong the textual evidence is (0.7-0.9 for clear implications)
     - How much context supports the relationship (0.5-0.7 for inferred relations)
     - Avoid speculative or weakly supported relationships (below 0.5)
6.  **Format:** Your output must be a JSON array of objects. Each object must have five keys: `subject` (string), `predicate` (string), `object` (string), `context` (string), and `confidence` (float between 0.0 and 1.0).
7.  **Constraint:** Return all significant relationships. Do not arbitrarily limit your response.

**Output Format:**
- You **MUST** return a valid JSON array of objects.
- Each object represents a relationship and must contain the keys: `subject`, `predicate`, `object`, `context`, and `confidence`.
- Do not include any text or explanations outside of the JSON array.

**Constraint:** Focus on the most important and clearly stated relationships. 
Text: {text}
"""

def _safe_extract_json(response: str, expected_type: type = list):
    """
    Safely extracts a JSON object (list or dict) from a string,
    even if it's wrapped in text or markdown.
    """
    if not response:
        return expected_type()

    start_bracket = response.find('[')
    start_curly = response.find('{')
    
    if start_bracket == -1 and start_curly == -1:
        return expected_type()

    if start_bracket == -1:
        start_pos = start_curly
    elif start_curly == -1:
        start_pos = start_bracket
    else:
        start_pos = min(start_bracket, start_curly)

    if response[start_pos] == '[':
        end_pos = response.rfind(']')
    else:
        end_pos = response.rfind('}')
        
    if end_pos == -1:
        return expected_type()

    json_str = response[start_pos : end_pos + 1]
    
    try:
        data = json.loads(json_str)
        if isinstance(data, expected_type):
            return data
        return expected_type()
    except json.JSONDecodeError:
        return expected_type()


def _validate_entity_in_text(entity_name: str, text: str) -> bool:
    """
    Validate that an entity is actually mentioned in the text using simple keyword matching.
    This helps reduce hallucinated entities.
    """
    if not entity_name or not text:
        return False
    
    # Convert to lowercase for case-insensitive matching
    entity_lower = entity_name.lower()
    text_lower = text.lower()
    
    # Check for exact match
    if entity_lower in text_lower:
        return True
    
    # Check for partial matches (split entity name by common separators)
    entity_words = re.split(r'[\s\-_]+', entity_lower)
    entity_words = [word.strip() for word in entity_words if len(word.strip()) > 2]
    
    if not entity_words:
        return False
    
    # At least 70% of entity words should appear in the text
    matches = sum(1 for word in entity_words if word in text_lower)
    match_ratio = matches / len(entity_words)
    
    return match_ratio >= 0.7


def _deduplicate_entities(entities: List[Dict]) -> List[Dict]:
    """
    Deduplicate entities based on name similarity and merge attributes.
    """
    if not entities:
        return []
    
    deduplicated = []
    processed_names = set()
    
    for entity in entities:
        name = entity["name"]
        name_lower = name.lower()
        
        # Check for exact duplicates
        if name_lower in processed_names:
            continue
        
        # Check for similar names (simple approach)
        is_duplicate = False
        for existing in deduplicated:
            existing_name_lower = existing["name"].lower()
            
            # If names are very similar (>80% character overlap)
            if _calculate_similarity(name_lower, existing_name_lower) > 0.8:
                # Merge attributes and keep the one with higher confidence
                if entity.get("confidence", 0) > existing.get("confidence", 0):
                    # Update existing with better entity but merge attributes
                    merged_attributes = existing.get("attributes", {})
                    merged_attributes.update(entity.get("attributes", {}))
                    existing.update(entity)
                    existing["attributes"] = merged_attributes
                else:
                    # Just merge attributes into existing
                    existing_attributes = existing.get("attributes", {})
                    existing_attributes.update(entity.get("attributes", {}))
                    existing["attributes"] = existing_attributes
                
                is_duplicate = True
                break
        
        if not is_duplicate:
            deduplicated.append(entity)
            processed_names.add(name_lower)
    
    return deduplicated


def _calculate_similarity(str1: str, str2: str) -> float:
    """
    Calculate simple character-based similarity between two strings.
    """
    if not str1 or not str2:
        return 0.0
    
    # Simple Jaccard similarity using character bigrams
    def get_bigrams(s):
        return set(s[i:i+2] for i in range(len(s)-1))
    
    bigrams1 = get_bigrams(str1)
    bigrams2 = get_bigrams(str2)
    
    if not bigrams1 and not bigrams2:
        return 1.0
    if not bigrams1 or not bigrams2:
        return 0.0
    
    intersection = len(bigrams1 & bigrams2)
    union = len(bigrams1 | bigrams2)
    
    return intersection / union if union > 0 else 0.0


def extract_entities(text: str) -> List[Dict[str, str]]:
    """Extract entities with types, attributes, and confidence scores using LLM."""
    try:
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text.strip()[:1500])
        llm = get_llm()
        response = llm.invoke(prompt).content.strip()
        
        entities = _safe_extract_json(response, list)
        
        valid_entities = []
        for entity in entities:
            if isinstance(entity, dict) and all(k in entity for k in ["name", "type"]):
                name = entity["name"].strip()[:200]
                entity_type = entity["type"].strip().upper()[:50]
                attributes = entity.get("attributes", {})
                confidence = entity.get("confidence", 0.5)  # Default confidence if not provided
                
                # Ensure confidence is a valid float between 0 and 1
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.5
                
                # Filter by confidence threshold
                if name and entity_type and confidence >= ENTITY_CONFIDENCE_THRESHOLD:
                    # Validate entity against chunk text using simple keyword matching
                    if ENABLE_EMBEDDING_VALIDATION:
                        if not _validate_entity_in_text(name, text):
                            confidence *= 0.7  # Reduce confidence if validation fails
                            if confidence < ENTITY_CONFIDENCE_THRESHOLD:
                                continue
                    
                    valid_entities.append({
                        "name": name,
                        "type": entity_type,
                        "attributes": attributes if isinstance(attributes, dict) else {},
                        "confidence": confidence
                    })
        
        # Sort by confidence and return top entities
        valid_entities.sort(key=lambda x: x["confidence"], reverse=True)
        return valid_entities[:10]
        
    except Exception as e:
        print(f"[GraphRAG] Entity extraction failed: {e}")
        return []

def extract_relations(text: str, entities: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Extract relationships between entities with confidence scores using LLM."""
    try:
        entity_names = [e["name"] for e in entities]
        enhanced_prompt = RELATION_EXTRACTION_PROMPT.format(text=text.strip()[:1500])
        if entity_names:
            enhanced_prompt += f"\n\nKnown entities in this text: {', '.join(entity_names[:10])}"
        
        llm = get_llm()
        response = llm.invoke(enhanced_prompt).content.strip()
        
        relations = _safe_extract_json(response, list)
        
        valid_relations = []
        for relation in relations:
            if isinstance(relation, dict) and all(k in relation for k in ["subject", "predicate", "object"]):
                subject = relation["subject"].strip()[:200]
                predicate = relation["predicate"].strip()[:100]
                object_name = relation["object"].strip()[:200]
                context = relation.get("context", "").strip()[:300]
                confidence = relation.get("confidence", 0.5)  # Default confidence if not provided
                
                # Ensure confidence is a valid float between 0 and 1
                try:
                    confidence = float(confidence)
                    confidence = max(0.0, min(1.0, confidence))
                except (ValueError, TypeError):
                    confidence = 0.5
                
                # Filter by confidence threshold
                if subject and predicate and object_name and confidence >= RELATION_CONFIDENCE_THRESHOLD:
                    # Validate that both subject and object entities exist in the extracted entities or text
                    if ENABLE_EMBEDDING_VALIDATION:
                        subject_valid = _validate_entity_in_text(subject, text) or any(e["name"].lower() == subject.lower() for e in entities)
                        object_valid = _validate_entity_in_text(object_name, text) or any(e["name"].lower() == object_name.lower() for e in entities)
                        
                        if not (subject_valid and object_valid):
                            confidence *= 0.6  # Reduce confidence if validation fails
                            if confidence < RELATION_CONFIDENCE_THRESHOLD:
                                continue
                    
                    valid_relations.append({
                        "subject": subject,
                        "predicate": predicate,
                        "object": object_name,
                        "context": context,
                        "confidence": confidence
                    })
        
        # Sort by confidence and return top relations
        valid_relations.sort(key=lambda x: x["confidence"], reverse=True)
        return valid_relations[:15]
        
    except Exception as e:
        print(f"[GraphRAG] Relation extraction failed: {e}")
        return []


# ========== Neo4j Graph Operations ==========

def upsert_entities_and_relations(
    file_id: str,
    page: Optional[int],
    chunk_no: int,
    source: str,
    text: str
) -> None:
    """
    Extract entities and relationships from text with confidence filtering and store them in Neo4j.
    """
    if not get_graph_db().is_connected():
        print("[GraphRAG] Neo4j not connected, skipping graph update")
        return
    
    try:
        # Extract entities and relations with confidence scores
        raw_entities = extract_entities(text)
        
        # Deduplicate entities to reduce noise
        entities = _deduplicate_entities(raw_entities)
        
        # Extract relations based on filtered entities
        relations = extract_relations(text, entities)
        
        # Log extraction results with confidence info
        avg_entity_conf = sum(e.get("confidence", 0) for e in entities) / len(entities) if entities else 0
        avg_relation_conf = sum(r.get("confidence", 0) for r in relations) / len(relations) if relations else 0
        
        print(f"[GraphRAG] Extracted {len(entities)} entities (avg conf: {avg_entity_conf:.2f}) and {len(relations)} relations (avg conf: {avg_relation_conf:.2f}) for file={file_id}, chunk={chunk_no}")
        
        if not entities and not relations:
            return
        
        with get_graph_db().driver.session() as session:
            # Create document and chunk nodes
            session.run("""
                MERGE (d:Document {file_id: $file_id})
                SET d.source = $source, d.updated_at = timestamp()
                MERGE (c:Chunk {file_id: $file_id, chunk_no: $chunk_no})
                SET c.page = $page, c.text = $text, c.source = $source, c.updated_at = timestamp()
                MERGE (d)-[:HAS_CHUNK]->(c)
            """, file_id=file_id, chunk_no=chunk_no, page=page, text=text[:1000], source=source)
            
            # Insert entities with confidence scores
            for entity in entities:
                # Convert attributes dict to a JSON string to prevent Neo4j TypeError
                attributes_str = json.dumps(entity.get("attributes", {}))
                confidence = entity.get("confidence", 0.5)
                
                session.run("""
                    MERGE (e:Entity {name: $name})
                    SET e.type = $type, e.attributes = $attributes, e.confidence = $confidence, e.updated_at = timestamp()
                    WITH e
                    MATCH (c:Chunk {file_id: $file_id, chunk_no: $chunk_no})
                    MERGE (c)-[:MENTIONS]->(e)
                """, name=entity["name"], type=entity["type"], 
                    attributes=attributes_str, confidence=confidence,
                    file_id=file_id, chunk_no=chunk_no)
            
            # Insert relations with confidence scores
            for relation in relations:
                confidence = relation.get("confidence", 0.5)
                
                session.run("""
                    MERGE (s:Entity {name: $subject})
                    MERGE (o:Entity {name: $object})
                    MERGE (s)-[r:RELATES {type: $predicate}]->(o)
                    SET r.context = $context, r.confidence = $confidence, r.file_id = $file_id, r.chunk_no = $chunk_no, 
                        r.page = $page, r.source = $source, r.updated_at = timestamp()
                """, subject=relation["subject"], object=relation["object"], 
                    predicate=relation["predicate"], context=relation["context"], confidence=confidence,
                    file_id=file_id, chunk_no=chunk_no, page=page, source=source)
        
    except Exception as e:
        print(f"[GraphRAG] Failed to upsert entities and relations: {e}")


QUERY_ENTITY_PROMPT = """
Identify the main entities mentioned in this question that would be relevant for searching a knowledge graph.
Return ONLY a JSON array of entity names (strings). Focus on proper nouns, important concepts, and key terms.

Question: {question}
"""

def extract_query_entities(query: str) -> List[str]:
    """Extract important entities from the query to anchor graph lookup."""
    try:
        prompt = QUERY_ENTITY_PROMPT.format(question=query)
        llm = get_llm()
        response = llm.invoke(prompt).content.strip()
        
        entities = _safe_extract_json(response, list)
        
        if isinstance(entities, list):
            return [str(e).strip() for e in entities if str(e).strip()][:5]
        
        return []
        
    except Exception as e:
        print(f"[GraphRAG] Query entity extraction failed: {e}")
        return []


def get_subgraph_facts(
    entities: List[str],
    file_id: Optional[str] = None,
    folder_id: Optional[str] = None,
    max_facts: int = MAX_FACTS,
    min_confidence: float = 0.6
) -> str:
    """
    Build a textual block of 'facts' from the Neo4j graph, filtered by file_id and confidence.
    Uses intelligent matching: exact names, then types, then partial matches.
    Prioritizes high-confidence entities and relations to reduce noise.
    """
    if not entities or not get_graph_db().is_connected():
        return ""
    
    try:
        with get_graph_db().driver.session() as session:
            facts = []
            processed_entities = set()
            
            for query_entity in entities:
                query_entity_lower = query_entity.lower()
                
                # Strategy 1: Exact name match (case-insensitive) with confidence filtering
                entity_result = session.run("""
                    MATCH (e:Entity)
                    WHERE toLower(e.name) = $entity_name 
                    AND (e.confidence IS NULL OR e.confidence >= $min_confidence)
                    OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                    WHERE ($file_id IS NULL OR c.file_id = $file_id)
                    RETURN e.name as name, e.type as type, e.attributes as attributes, 
                           e.confidence as confidence,
                           collect(DISTINCT c.file_id) as mentioned_in_files
                    ORDER BY e.confidence DESC
                    LIMIT 5
                """, entity_name=query_entity_lower, file_id=file_id, min_confidence=min_confidence)
                
                found_entities = list(entity_result)
                
                # Strategy 2: If no exact match, try type-based matching
                if not found_entities:
                    # Map common query terms to types
                    type_mapping = {
                        'sets': 'SET', 'set': 'SET', 
                        'functions': 'FUNCTION', 'function': 'FUNCTION',
                        'concepts': 'CONCEPT', 'concept': 'CONCEPT',
                        'locations': 'LOCATION', 'location': 'LOCATION'
                    }
                    
                    target_type = type_mapping.get(query_entity_lower)
                    if target_type:
                        entity_result = session.run("""
                            MATCH (e:Entity)
                            WHERE e.type = $entity_type
                            AND (e.confidence IS NULL OR e.confidence >= $min_confidence)
                            OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                            WHERE ($file_id IS NULL OR c.file_id = $file_id)
                            RETURN e.name as name, e.type as type, e.attributes as attributes,
                                   e.confidence as confidence,
                                   collect(DISTINCT c.file_id) as mentioned_in_files
                            ORDER BY e.confidence DESC
                            LIMIT 5
                        """, entity_type=target_type, file_id=file_id, min_confidence=min_confidence)
                        
                        found_entities = list(entity_result)
                
                # Strategy 3: If still no match, try partial/fuzzy matching
                if not found_entities:
                    # Extract key words from query entity
                    query_words = [w.strip().upper() for w in query_entity.replace(',', ' ').split() if len(w.strip()) > 1]
                    
                    for word in query_words:
                        entity_result = session.run("""
                            MATCH (e:Entity)
                            WHERE (toUpper(e.name) CONTAINS $word OR toUpper(e.type) CONTAINS $word)
                            AND (e.confidence IS NULL OR e.confidence >= $min_confidence)
                            OPTIONAL MATCH (c:Chunk)-[:MENTIONS]->(e)
                            WHERE ($file_id IS NULL OR c.file_id = $file_id)
                            RETURN e.name as name, e.type as type, e.attributes as attributes,
                                   e.confidence as confidence,
                                   collect(DISTINCT c.file_id) as mentioned_in_files
                            ORDER BY e.confidence DESC
                            LIMIT 3
                        """, word=word, file_id=file_id, min_confidence=min_confidence)
                        
                        partial_entities = list(entity_result)
                        if partial_entities:
                            found_entities = partial_entities
                            break
                
                # Process found entities
                for entity_record in found_entities:
                    name = entity_record["name"]
                    if name in processed_entities:
                        continue
                    processed_entities.add(name)
                    
                    entity_type = entity_record["type"] or "ENTITY"
                    confidence = entity_record["confidence"] or 0.0
                    
                    # Parse attributes
                    try:
                        attributes = json.loads(entity_record["attributes"] or "{}")
                    except (json.JSONDecodeError, TypeError):
                        attributes = {}
                    
                    # Add entity fact with confidence
                    attr_str = ""
                    if attributes:
                        attr_list = [f"{k}:{v}" for k, v in attributes.items() if v][:3]
                        if attr_list:
                            attr_str = f" [{', '.join(attr_list)}]"
                    
                    confidence_str = f" (conf: {confidence:.2f})" if confidence > 0 else ""
                    facts.append(f"Entity: {name} (Type: {entity_type}){attr_str}{confidence_str}")
                    
                    # Get high-confidence relationships for this entity
                    rel_result = session.run("""
                        MATCH (s:Entity {name: $entity_name})-[r:RELATES]->(o:Entity)
                        WHERE ($file_id IS NULL OR r.file_id = $file_id)
                        AND (r.confidence IS NULL OR r.confidence >= $min_confidence)
                        AND (o.confidence IS NULL OR o.confidence >= $min_confidence)
                        RETURN s.name as subject, r.type as predicate, o.name as object, 
                               r.context as context, r.confidence as rel_confidence, 
                               o.confidence as obj_confidence,
                               r.file_id as file_id, r.chunk_no as chunk_no
                        ORDER BY r.confidence DESC, o.confidence DESC
                        LIMIT $limit
                    """, entity_name=name, file_id=file_id, min_confidence=min_confidence,
                        limit=max_facts // max(len(entities), 1))
                    
                    for record in rel_result:
                        rel_conf = record['rel_confidence'] or 0.0
                        obj_conf = record['obj_confidence'] or 0.0
                        
                        fact = f"{record['subject']} --[{record['predicate']}]--> {record['object']}"
                        if record['context']:
                            fact += f" (Context: {record['context'][:100]})"
                        fact += f" (file:{record['file_id']}, chunk:{record['chunk_no']}"
                        if rel_conf > 0 or obj_conf > 0:
                            fact += f", rel_conf:{rel_conf:.2f}, obj_conf:{obj_conf:.2f}"
                        fact += ")"
                        facts.append(fact)
                        
                        if len(facts) >= max_facts:
                            break
                
                if len(facts) >= max_facts:
                    break
            
            return "\n".join(facts[:max_facts])
            
    except Exception as e:
        print(f"[GraphRAG] Failed to get subgraph facts: {e}")
        return ""


def get_graph_stats() -> Dict:
    """Get statistics about the knowledge graph including confidence metrics."""
    if not get_graph_db().is_connected():
        return {"error": "Neo4j not connected"}
    
    try:
        with get_graph_db().driver.session() as session:
            # Entity and relationship counts with confidence stats
            result = session.run("""
                MATCH (e:Entity) 
                OPTIONAL MATCH (e)-[r:RELATES]->()
                RETURN count(DISTINCT e) as entities, 
                       count(r) as relationships,
                       collect(DISTINCT e.type) as entity_types,
                       avg(e.confidence) as avg_entity_confidence,
                       avg(r.confidence) as avg_relation_confidence,
                       count(CASE WHEN e.confidence >= $threshold THEN 1 END) as high_conf_entities,
                       count(CASE WHEN r.confidence >= $threshold THEN 1 END) as high_conf_relations
            """, threshold=ENTITY_CONFIDENCE_THRESHOLD)
            
            record = result.single()
            
            # Document and chunk counts
            file_result = session.run("""
                MATCH (d:Document)
                OPTIONAL MATCH (d)-[:HAS_CHUNK]->(c:Chunk)
                RETURN count(DISTINCT d) as documents, count(c) as chunks
            """)
            
            file_record = file_result.single()
            
            return {
                "entities": record["entities"] or 0,
                "relationships": record["relationships"] or 0,
                "entity_types": [t for t in (record["entity_types"] or []) if t],
                "documents": file_record["documents"] or 0,
                "chunks": file_record["chunks"] or 0,
                "avg_entity_confidence": round(record["avg_entity_confidence"] or 0, 3),
                "avg_relation_confidence": round(record["avg_relation_confidence"] or 0, 3),
                "high_confidence_entities": record["high_conf_entities"] or 0,
                "high_confidence_relations": record["high_conf_relations"] or 0,
                "confidence_threshold": ENTITY_CONFIDENCE_THRESHOLD
            }
            
    except Exception as e:
        print(f"[GraphRAG] Failed to get graph stats: {e}")
        return {"error": str(e)}

def cleanup_low_confidence_entities(confidence_threshold: float = None) -> Dict:
    """
    Remove entities and relations below the confidence threshold to clean up the graph.
    """
    if not get_graph_db().is_connected():
        return {"error": "Neo4j not connected"}
    
    if confidence_threshold is None:
        confidence_threshold = ENTITY_CONFIDENCE_THRESHOLD
    
    try:
        with get_graph_db().driver.session() as session:
            # Count entities and relations before cleanup
            before_stats = session.run("""
                MATCH (e:Entity) 
                OPTIONAL MATCH (e)-[r:RELATES]->()
                RETURN count(DISTINCT e) as entities, count(r) as relationships
            """).single()
            
            # Delete low-confidence relations first
            relation_result = session.run("""
                MATCH ()-[r:RELATES]->()
                WHERE r.confidence IS NOT NULL AND r.confidence < $threshold
                DELETE r
                RETURN count(r) as deleted_relations
            """, threshold=confidence_threshold)
            
            deleted_relations = relation_result.single()["deleted_relations"]
            
            # Delete low-confidence entities (that are now orphaned or below threshold)
            entity_result = session.run("""
                MATCH (e:Entity)
                WHERE e.confidence IS NOT NULL AND e.confidence < $threshold
                OPTIONAL MATCH (e)-[r]-()
                DELETE r, e
                RETURN count(DISTINCT e) as deleted_entities
            """, threshold=confidence_threshold)
            
            deleted_entities = entity_result.single()["deleted_entities"]
            
            # Count entities and relations after cleanup
            after_stats = session.run("""
                MATCH (e:Entity) 
                OPTIONAL MATCH (e)-[r:RELATES]->()
                RETURN count(DISTINCT e) as entities, count(r) as relationships
            """).single()
            
            return {
                "deleted_entities": deleted_entities,
                "deleted_relations": deleted_relations,
                "before": {
                    "entities": before_stats["entities"],
                    "relationships": before_stats["relationships"]
                },
                "after": {
                    "entities": after_stats["entities"],
                    "relationships": after_stats["relationships"]
                },
                "threshold_used": confidence_threshold
            }
            
    except Exception as e:
        print(f"[GraphRAG] Failed to cleanup low confidence entities: {e}")
        return {"error": str(e)}


def update_confidence_thresholds(entity_threshold: float = None, relation_threshold: float = None) -> Dict:
    """
    Update the global confidence thresholds for entity and relation filtering.
    """
    global ENTITY_CONFIDENCE_THRESHOLD, RELATION_CONFIDENCE_THRESHOLD
    
    old_entity_threshold = ENTITY_CONFIDENCE_THRESHOLD
    old_relation_threshold = RELATION_CONFIDENCE_THRESHOLD
    
    if entity_threshold is not None:
        if 0.0 <= entity_threshold <= 1.0:
            ENTITY_CONFIDENCE_THRESHOLD = entity_threshold
        else:
            return {"error": "Entity threshold must be between 0.0 and 1.0"}
    
    if relation_threshold is not None:
        if 0.0 <= relation_threshold <= 1.0:
            RELATION_CONFIDENCE_THRESHOLD = relation_threshold
        else:
            return {"error": "Relation threshold must be between 0.0 and 1.0"}
    
    return {
        "old_thresholds": {
            "entity": old_entity_threshold,
            "relation": old_relation_threshold
        },
        "new_thresholds": {
            "entity": ENTITY_CONFIDENCE_THRESHOLD,
            "relation": RELATION_CONFIDENCE_THRESHOLD
        }
    }


def clear_graph():
    """Clear all data from the Neo4j graph."""
    if not get_graph_db().is_connected():
        return False
    
    try:
        with get_graph_db().driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("[GraphRAG] Graph cleared")
        return True
    except Exception as e:
        print(f"[GraphRAG] Failed to clear graph: {e}")
        return False