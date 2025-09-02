import asyncio
import time
import json
import os
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from enum import Enum
import networkx as nx
from collections import defaultdict, Counter
import neo4j
from neo4j import AsyncGraphDatabase, AsyncDriver
from graphBuilder import FolderScopedGraphBuilder
from documentProcessor import SearchMode

logger = logging.getLogger(__name__)

try:
    import community as community_louvain
    HAS_COMMUNITY_DETECTION = True
except ImportError:
    logger.warning("python-louvain not available. Install with: pip install python-louvain")
    HAS_COMMUNITY_DETECTION = False

@dataclass
class Community:
    """Represents a detected community for GraphRAG"""
    community_id: str
    entities: List[str]
    summary: str
    level: int
    size: int
    parent_community: Optional[str] = None
    folder_id: Optional[str] = None

@dataclass 
class GraphRAGContext:
    """Complete context for GraphRAG query answering"""
    entities: List[Dict[str, Any]]
    relationships: List[Dict[str, Any]]
    communities: List[Community]
    source_chunks: List[Dict[str, Any]]
    folder_context: Dict[str, Any]

class CommunityStorage:
    """Persistent storage system for detected communities"""
    
    def __init__(self, driver):
        self.driver = driver
    
    async def save_communities(self, communities: List[Community], folder_id: str = None) -> bool:
        """Save communities to Neo4j for persistence"""
        
        try:
            async with self.driver.session() as session:
                # Clear existing communities for this folder
                if folder_id:
                    await session.run("""
                        MATCH (c:Community {folder_id: $folder_id})
                        DETACH DELETE c
                    """, folder_id=folder_id)
                
                # Save new communities
                for community in communities:
                    community_data = {
                        'community_id': community.community_id,
                        'folder_id': folder_id,
                        'entities': community.entities,
                        'summary': community.summary,
                        'level': community.level,
                        'size': community.size,
                        'created_at': datetime.now().isoformat()
                    }
                    
                    await session.run("""
                        CREATE (c:Community)
                        SET c = $props
                    """, props=community_data)
                
                logger.info(f"Saved {len(communities)} communities to Neo4j")
                return True
                
        except Exception as e:
            logger.error(f"Error saving communities: {e}")
            return False
    
    async def load_communities(self, folder_id: str = None) -> List[Community]:
        """Load communities from Neo4j"""
        
        try:
            async with self.driver.session() as session:
                query = """
                MATCH (c:Community)
                WHERE $folder_id IS NULL OR c.folder_id = $folder_id
                RETURN c
                ORDER BY c.size DESC
                """
                
                result = await session.run(query, folder_id=folder_id)
                records = await result.data()
                
                communities = []
                for record in records:
                    props = record['c']
                    community = Community(
                        community_id=props['community_id'],
                        entities=props['entities'],
                        summary=props['summary'],
                        level=props['level'],
                        size=props['size'],
                        folder_id=props.get('folder_id')
                    )
                    communities.append(community)
                
                logger.info(f"Loaded {len(communities)} communities from Neo4j")
                return communities
                
        except Exception as e:
            logger.error(f"Error loading communities: {e}")
            return []
    
    async def get_communities_by_query(self, query: str, folder_id: str = None) -> List[Community]:
        """Get communities relevant to a specific query"""
        
        communities = await self.load_communities(folder_id)
        if not communities:
            return []
        
        # Score communities by relevance
        query_lower = query.lower()
        scored_communities = []
        
        for community in communities:
            score = 0.0
            
            # Score based on summary
            summary_words = community.summary.lower().split()
            query_words = query_lower.split()
            overlap = len(set(summary_words) & set(query_words))
            score += overlap * 0.5
            
            # Score based on entity names
            for entity in community.entities:
                if any(word in entity.lower() for word in query_lower.split()):
                    score += 1.0
            
            if score > 0:
                scored_communities.append((community, score))
        
        # Return sorted by relevance
        scored_communities.sort(key=lambda x: x[1], reverse=True)
        return [comm for comm, score in scored_communities]

class DirectGraphRAGExtractor:
    """Enhanced GraphRAG extractor with community detection and hierarchical search"""
    
    def __init__(self, neo4j_config: Dict[str, Any], graph_builder: FolderScopedGraphBuilder):
        self.neo4j_config = neo4j_config
        self.graph_builder = graph_builder
        self.driver: Optional[AsyncDriver] = None
        self.cached_communities: Dict[str, List[Community]] = {}
        
        # Add persistent storage
        self.community_storage = None

    async def initialize(self):
        """Initialize direct connection to Neo4j and community storage"""
        self.driver = AsyncGraphDatabase.driver(
            self.neo4j_config['uri'],
            auth=(self.neo4j_config['user'], self.neo4j_config['password'])
        )
        
        # Test connection
        async with self.driver.session() as session:
            await session.run("RETURN 1")
            logger.info("Direct Neo4j connection established")
        
        # Initialize community storage
        self.community_storage = CommunityStorage(self.driver)
        logger.info("Community storage initialized")

    async def wait_for_entity_extraction(self, max_wait_time: int = 300, check_interval: int = 10) -> bool:
        """Wait for Graphiti to complete entity extraction from episodes"""
        
        logger.info("Waiting for entity extraction to complete...")
        start_time = time.time()
        
        while time.time() - start_time < max_wait_time:
            stats = await self._get_current_graph_stats()
            
            if stats['total_entities'] > 0 and stats['total_relationships'] > 0:
                logger.info(f"Entity extraction completed: {stats['total_entities']} entities, "
                           f"{stats['total_relationships']} relationships found")
                return True
            
            logger.info(f"Still processing... Found {stats['total_entities']} entities so far")
            await asyncio.sleep(check_interval)
        
        logger.warning(f"Entity extraction didn't complete within {max_wait_time} seconds")
        return False

    async def _get_current_graph_stats(self) -> Dict[str, Any]:
        """Get current graph statistics with better error handling"""
        
        # Try multiple possible node labels that Graphiti might use
        stats_queries = [
            # Standard Graphiti labels
            """
            CALL {
                MATCH (e:Entity) RETURN count(e) as entities
                UNION ALL
                MATCH (n:Node) RETURN count(n) as entities
                UNION ALL  
                MATCH (n) WHERE n:Entity OR n:Node RETURN count(n) as entities
            }
            WITH max(entities) as total_entities
            
            CALL {
                MATCH ()-[r]-() WHERE type(r) <> 'EXTRACTED_FROM' RETURN count(r) as rels
                UNION ALL
                MATCH ()-[r:RELATED_TO|:MENTIONS|:PART_OF|:SIMILAR_TO]-() RETURN count(r) as rels
            }
            WITH total_entities, max(rels) as total_relationships
            
            MATCH (ep:Episode) 
            RETURN 
                total_entities,
                total_relationships,
                count(ep) as total_episodes
            """,
            
            # Fallback: get all nodes and relationships
            """
            MATCH (n) 
            WITH count(n) as all_nodes
            
            MATCH ()-[r]-()
            WITH all_nodes, count(r) as all_relationships
            
            MATCH (ep:Episode)
            RETURN 
                all_nodes as total_entities,
                all_relationships as total_relationships, 
                count(ep) as total_episodes
            """,
            
            # Minimal fallback
            """
            MATCH (n)
            OPTIONAL MATCH ()-[r]-()
            OPTIONAL MATCH (ep:Episode)
            RETURN 
                count(DISTINCT n) as total_entities,
                count(DISTINCT r) as total_relationships,
                count(DISTINCT ep) as total_episodes
            """
        ]
        
        for i, query in enumerate(stats_queries):
            try:
                async with self.driver.session() as session:
                    result = await session.run(query)
                    record = await result.single()
                    
                    if record:
                        return {
                            'total_entities': record['total_entities'] or 0,
                            'total_relationships': record['total_relationships'] or 0,
                            'total_episodes': record['total_episodes'] or 0,
                            'query_used': i + 1
                        }
            except Exception as e:
                logger.debug(f"Stats query {i+1} failed: {str(e)}")
                continue
        
        # Final fallback
        return {
            'total_entities': 0,
            'total_relationships': 0, 
            'total_episodes': 0,
            'query_used': 'fallback'
        }

    async def discover_node_structure(self) -> Dict[str, Any]:
        """Discover the actual node structure in the graph"""
        
        discovery_query = """
        // Get all node labels
        CALL db.labels() YIELD label
        WITH collect(label) as all_labels
        
        // Get all relationship types
        CALL db.relationshipTypes() YIELD relationshipType
        WITH all_labels, collect(relationshipType) as all_rel_types
        
        // Sample nodes of each type
        UNWIND all_labels as label
        CALL {
            WITH label
            CALL apoc.cypher.run("MATCH (n:" + label + ") RETURN n LIMIT 3", {}) 
            YIELD value
            RETURN value.n as sample_node, label
        }
        
        RETURN 
            all_labels,
            all_rel_types,
            collect({label: label, sample: properties(sample_node)}) as node_samples
        """
        
        try:
            async with self.driver.session() as session:
                result = await session.run(discovery_query)
                record = await result.single()
                
                return {
                    'node_labels': record['all_labels'],
                    'relationship_types': record['all_rel_types'], 
                    'node_samples': record['node_samples']
                }
        except Exception as e:
            # Fallback discovery without APOC
            logger.info("Using fallback discovery method")
            return await self._fallback_discovery()

    async def _fallback_discovery(self) -> Dict[str, Any]:
        """Fallback discovery method without APOC procedures"""
        
        async with self.driver.session() as session:
            # Get sample of all nodes
            result = await session.run("""
                MATCH (n) 
                RETURN labels(n) as node_labels, properties(n) as properties
                LIMIT 20
            """)
            records = await result.data()
            
            # Get sample relationships
            rel_result = await session.run("""
                MATCH ()-[r]-()
                RETURN DISTINCT type(r) as rel_type
                LIMIT 20  
            """)
            rel_records = await rel_result.data()
            
            # Process results
            all_labels = set()
            node_samples = []
            
            for record in records:
                labels = record['node_labels']
                if labels:
                    all_labels.update(labels)
                    node_samples.append({
                        'labels': labels,
                        'sample_properties': list(record['properties'].keys()) if record['properties'] else []
                    })
            
            rel_types = [r['rel_type'] for r in rel_records]
            
            return {
                'node_labels': list(all_labels),
                'relationship_types': rel_types,
                'node_samples': node_samples[:10]
            }


    async def detect_communities(self, folder_id: str = None, 
                           min_community_size: int = 3,
                           max_communities: int = 20,
                           force_redetect: bool = False) -> List[Community]:
        """Detect communities with persistent storage"""
        
        if not HAS_COMMUNITY_DETECTION:
            logger.warning("Community detection not available. Install python-louvain: pip install python-louvain")
            return []
        
        # Try to load existing communities first (unless force redetect)
        if not force_redetect and self.community_storage:
            existing_communities = await self.community_storage.load_communities(folder_id)
            if existing_communities:
                logger.info(f"Loaded {len(existing_communities)} existing communities from storage")
                # Update cache
                cache_key = folder_id or 'global'
                self.cached_communities[cache_key] = existing_communities
                return existing_communities
        
        # Wait for entity extraction and get flexible entity graph
        await self.wait_for_entity_extraction(max_wait_time=60)
        entities, relationships = await self._get_entity_relationship_graph_flexible(folder_id)
        
        if len(entities) < min_community_size:
            logger.info(f"Not enough entities for community detection: {len(entities)} found, need {min_community_size}")
            return []
        
        if len(relationships) == 0:
            logger.info("No relationships found, creating single community from all entities")
            # Create one community with all entities
            summary = f"Single community containing all {len(entities)} extracted entities"
            communities = [Community(
                community_id=f"single_comm_{folder_id or 'global'}",
                entities=[e['name'] for e in entities],
                summary=summary,
                level=0,
                size=len(entities),
                folder_id=folder_id
            )]
        else:
            # Perform community detection
            communities = await self._perform_community_detection(
                entities, relationships, folder_id, min_community_size, max_communities
            )
        
        # Save to persistent storage
        if self.community_storage and communities:
            await self.community_storage.save_communities(communities, folder_id)
        
        # Update cache
        cache_key = folder_id or 'global'
        self.cached_communities[cache_key] = communities
        
        logger.info(f"Successfully detected and stored {len(communities)} communities")
        return communities

    async def _perform_community_detection(self, entities: List[Dict], relationships: List[Dict],
                                        folder_id: str, min_community_size: int, 
                                        max_communities: int) -> List[Community]:
        """Perform the actual community detection algorithm"""
        
        logger.info(f"Building community graph from {len(entities)} entities and {len(relationships)} relationships")
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add entities as nodes
        entity_to_id = {entity['name']: i for i, entity in enumerate(entities)}
        for entity in entities:
            G.add_node(entity_to_id[entity['name']], 
                    name=entity['name'], 
                    description=entity.get('description', ''))
        
        # Add relationships as edges
        edges_added = 0
        for rel in relationships:
            source_id = entity_to_id.get(rel['source_entity'])
            target_id = entity_to_id.get(rel['target_entity'])
            if source_id is not None and target_id is not None and source_id != target_id:
                weight = rel.get('weight', 1.0)
                G.add_edge(source_id, target_id, weight=weight)
                edges_added += 1
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {edges_added} edges")
        
        # Detect communities using Louvain algorithm
        try:
            partition = community_louvain.best_partition(G)
            logger.info(f"Louvain partitioning completed: {len(set(partition.values()))} communities")
        except Exception as e:
            logger.warning(f"Louvain algorithm failed: {e}, using connected components")
            partition = {}
            for i, component in enumerate(nx.connected_components(G)):
                for node in component:
                    partition[node] = i
        
        # Convert to Community objects
        communities = []
        community_groups = defaultdict(list)
        
        for node_id, comm_id in partition.items():
            entity_name = G.nodes[node_id]['name']
            community_groups[comm_id].append(entity_name)
        
        for comm_id, entity_names in community_groups.items():
            if len(entity_names) >= min_community_size:
                summary = await self._generate_community_summary(entity_names, entities)
                
                community = Community(
                    community_id=f"comm_{comm_id}_{folder_id or 'global'}",
                    entities=entity_names,
                    summary=summary,
                    level=0,
                    size=len(entity_names),
                    folder_id=folder_id
                )
                communities.append(community)
        
        return communities[:max_communities]

    async def _get_entity_relationship_graph_flexible(self, folder_id: str = None) -> Tuple[List[Dict], List[Dict]]:
        """Extract entities and relationships with flexible node label detection"""
        
        # First discover what node types exist
        structure = await self.discover_node_structure()
        node_labels = structure['node_labels']
        
        logger.info(f"Available node labels: {node_labels}")
        
        entities = []
        relationships = []
        
        # Try different entity node patterns
        entity_patterns = [
            "Entity",  # Standard Graphiti
            "Node",    # Alternative
            "Concept", # Possible alternative
        ]
        
        for pattern in entity_patterns:
            if pattern in node_labels:
                logger.info(f"Using node label: {pattern}")
                
                # Get entities
                entity_query = f"""
                MATCH (e:{pattern})
                RETURN 
                    e.name as name,
                    coalesce(e.entity_type, e.type, 'unknown') as type,
                    coalesce(e.description, e.summary, '') as description,
                    id(e) as neo4j_id
                ORDER BY e.name
                LIMIT 1000
                """
                
                # Get relationships
                relationship_query = f"""
                MATCH (a:{pattern})-[r]-(b:{pattern})
                WHERE a <> b
                RETURN DISTINCT
                    a.name as source_entity,
                    b.name as target_entity,
                    type(r) as relationship_type,
                    coalesce(r.weight, 1.0) as weight
                LIMIT 1000
                """
                
                try:
                    async with self.driver.session() as session:
                        # Get entities
                        entity_result = await session.run(entity_query)
                        entities = await entity_result.data()
                        
                        # Get relationships
                        rel_result = await session.run(relationship_query)
                        relationships = await rel_result.data()
                        
                        if entities:
                            logger.info(f"Found {len(entities)} entities and {len(relationships)} relationships using label '{pattern}'")
                            break
                            
                except Exception as e:
                    logger.debug(f"Failed to query with label '{pattern}': {e}")
                    continue
        
        # If no entities found with standard labels, try a more general approach
        if not entities:
            logger.info("No entities found with standard labels, trying general approach...")
            entities, relationships = await self._extract_entities_from_episodes()
        
        return entities, relationships

    async def _extract_entities_from_episodes(self) -> Tuple[List[Dict], List[Dict]]:
        """Extract entity-like information directly from episodes when entities aren't extracted yet"""
        
        episode_query = """
        MATCH (ep:Episode)
        RETURN 
            ep.name as episode_name,
            ep.episode_body as content,
            ep.source_description as source
        ORDER BY ep.reference_time DESC
        LIMIT 100
        """
        
        async with self.driver.session() as session:
            result = await session.run(episode_query)
            episodes = await result.data()
        
        # Extract potential entities from episode content
        entities = []
        relationships = []
        entity_mentions = defaultdict(list)
        
        for episode in episodes:
            content = episode.get('content', '')
            if not content:
                continue
                
            # Simple entity extraction (capitalize words, length > 2)
            words = content.split()
            potential_entities = []
            
            for word in words:
                cleaned_word = word.strip('.,!?();:"\'')
                if (cleaned_word.istitle() and 
                    len(cleaned_word) > 2 and 
                    cleaned_word.isalpha()):
                    potential_entities.append(cleaned_word)
            
            # Count entity mentions
            for entity in potential_entities:
                entity_mentions[entity].append(episode['episode_name'])
        
        # Convert to entity format
        entity_id = 0
        for entity_name, mentions in entity_mentions.items():
            if len(mentions) >= 2:  # Entity must appear in at least 2 episodes
                entities.append({
                    'name': entity_name,
                    'type': 'extracted',
                    'description': f"Mentioned in {len(mentions)} documents",
                    'neo4j_id': entity_id,
                    'mention_count': len(mentions)
                })
                entity_id += 1
        
        # Create co-occurrence relationships
        for episode in episodes:
            content = episode.get('content', '')
            episode_entities = []
            
            for entity in entities:
                if entity['name'].lower() in content.lower():
                    episode_entities.append(entity['name'])
            
            # Create relationships between co-occurring entities
            for i, entity1 in enumerate(episode_entities):
                for entity2 in episode_entities[i+1:]:
                    relationships.append({
                        'source_entity': entity1,
                        'target_entity': entity2,
                        'relationship_type': 'CO_OCCURS',
                        'weight': 1.0
                    })
        
        logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships from episodes")
        return entities, relationships

    async def get_folder_graph_statistics(self, folder_id: str = None) -> Dict[str, Any]:
        """Get comprehensive GraphRAG statistics with better error handling"""
        
        # Wait a bit for processing to complete
        await self.wait_for_entity_extraction(max_wait_time=60)
        
        # Get current stats
        stats = await self._get_current_graph_stats()
        
        # Discover graph structure
        structure = await self.discover_node_structure()
        
        # Calculate readiness
        total_entities = stats['total_entities']
        total_relationships = stats['total_relationships']
        connectivity_ratio = (total_relationships / max(total_entities, 1)) if total_entities > 0 else 0.0
        
        # Assess GraphRAG readiness
        graphrag_readiness = 'READY'
        recommendations = []
        
        if total_entities < 5:
            graphrag_readiness = 'INSUFFICIENT_ENTITIES'
            recommendations.append("Add more documents or wait for entity extraction to complete")
        elif total_relationships < 3:
            graphrag_readiness = 'INSUFFICIENT_RELATIONSHIPS' 
            recommendations.append("Entity extraction may still be in progress")
        elif connectivity_ratio < 0.05:
            graphrag_readiness = 'LOW_CONNECTIVITY'
            recommendations.append("Consider adding more diverse content")
        
        return {
            'folder_id': folder_id,
            'graphrag_readiness': graphrag_readiness,
            'total_entities': total_entities,
            'total_episodes': stats['total_episodes'],
            'total_relationships': total_relationships,
            'connectivity_ratio': connectivity_ratio,
            'community_detection_viable': total_entities >= 5 and connectivity_ratio >= 0.05,
            'recommended_search_modes': self._recommend_search_modes(total_entities, connectivity_ratio),
            'graph_structure': structure,
            'processing_recommendations': recommendations,
            'stats_query_used': stats.get('query_used', 'unknown')
        }

    async def detect_communities(self, folder_id: str = None, 
                               min_community_size: int = 3,
                               max_communities: int = 20) -> List[Community]:
        """Detect communities with improved entity discovery"""
        
        if not HAS_COMMUNITY_DETECTION:
            logger.warning("Community detection not available. Install python-louvain: pip install python-louvain")
            return []
        
        # Wait for entity extraction and get flexible entity graph
        await self.wait_for_entity_extraction(max_wait_time=60)
        entities, relationships = await self._get_entity_relationship_graph_flexible(folder_id)
        
        if len(entities) < min_community_size:
            logger.info(f"Not enough entities for community detection: {len(entities)} found, need {min_community_size}")
            return []
        
        if len(relationships) == 0:
            logger.info("No relationships found, creating single community from all entities")
            # Create one community with all entities
            summary = f"Single community containing all {len(entities)} extracted entities"
            return [Community(
                community_id=f"single_comm_{folder_id or 'global'}",
                entities=[e['name'] for e in entities],
                summary=summary,
                level=0,
                size=len(entities),
                folder_id=folder_id
            )]
        
        logger.info(f"Building community graph from {len(entities)} entities and {len(relationships)} relationships")
        
        # Build NetworkX graph
        G = nx.Graph()
        
        # Add entities as nodes
        entity_to_id = {entity['name']: i for i, entity in enumerate(entities)}
        for entity in entities:
            G.add_node(entity_to_id[entity['name']], 
                      name=entity['name'], 
                      description=entity.get('description', ''))
        
        # Add relationships as edges
        edges_added = 0
        for rel in relationships:
            source_id = entity_to_id.get(rel['source_entity'])
            target_id = entity_to_id.get(rel['target_entity'])
            if source_id is not None and target_id is not None and source_id != target_id:
                weight = rel.get('weight', 1.0)
                G.add_edge(source_id, target_id, weight=weight)
                edges_added += 1
        
        logger.info(f"Graph built: {G.number_of_nodes()} nodes, {edges_added} edges")
        
        if G.number_of_edges() == 0:
            logger.warning("No valid edges in graph, using connected components")
            # Create communities from connected components (isolated nodes)
            communities = []
            for i, component in enumerate(nx.connected_components(G)):
                if len(component) >= min_community_size:
                    entity_names = [G.nodes[node_id]['name'] for node_id in component]
                    summary = f"Connected component with {len(entity_names)} entities"
                    
                    communities.append(Community(
                        community_id=f"comp_{i}_{folder_id or 'global'}",
                        entities=entity_names,
                        summary=summary,
                        level=0,
                        size=len(entity_names),
                        folder_id=folder_id
                    ))
            return communities
        
        # Detect communities using Louvain algorithm
        try:
            partition = community_louvain.best_partition(G)
            logger.info(f"Louvain partitioning completed: {len(set(partition.values()))} communities")
        except Exception as e:
            logger.warning(f"Louvain algorithm failed: {e}, using connected components")
            partition = {}
            for i, component in enumerate(nx.connected_components(G)):
                for node in component:
                    partition[node] = i
        
        # Convert to Community objects
        communities = []
        community_groups = defaultdict(list)
        
        for node_id, comm_id in partition.items():
            entity_name = G.nodes[node_id]['name']
            community_groups[comm_id].append(entity_name)
        
        for comm_id, entity_names in community_groups.items():
            if len(entity_names) >= min_community_size:
                summary = await self._generate_community_summary(entity_names, entities)
                
                community = Community(
                    community_id=f"comm_{comm_id}_{folder_id or 'global'}",
                    entities=entity_names,
                    summary=summary,
                    level=0,
                    size=len(entity_names),
                    folder_id=folder_id
                )
                communities.append(community)
        
        # Cache communities
        cache_key = folder_id or 'global'
        self.cached_communities[cache_key] = communities
        
        logger.info(f"Successfully detected {len(communities)} communities with sizes: {[c.size for c in communities]}")
        
        return communities[:max_communities]

    async def _generate_community_summary(self, entity_names: List[str], 
                                        all_entities: List[Dict]) -> str:
        """Generate summary for a community of entities"""
        
        # Get entity descriptions
        entity_descriptions = {}
        for entity in all_entities:
            if entity['name'] in entity_names:
                entity_descriptions[entity['name']] = entity.get('description', '')
        
        # Simple summary generation
        if len(entity_names) <= 3:
            return f"Small community containing: {', '.join(entity_names)}"
        else:
            # Identify common themes from descriptions
            all_text = ' '.join(entity_descriptions.values()).lower()
            if all_text.strip():
                common_words = Counter(word for word in all_text.split() 
                                     if len(word) > 3 and word.isalpha())
                top_themes = [word for word, count in common_words.most_common(3) 
                             if count > 1]
                
                theme_text = ', '.join(top_themes) if top_themes else 'various topics'
                return f"Community of {len(entity_names)} entities related to: {theme_text}"
            else:
                return f"Community of {len(entity_names)} entities: {', '.join(entity_names[:5])}"

    # ... [Keep all the other methods from your original code: global_search, local_search, 
    #      hybrid_search, etc. - they don't need changes]

    async def get_all_communities(self, folder_id: str = None) -> List[Community]:
        """Get all communities for a folder with fallback to detection"""
        
        # Try cache first
        cache_key = folder_id or 'global'
        if cache_key in self.cached_communities:
            return self.cached_communities[cache_key]
        
        # Try persistent storage
        if self.community_storage:
            communities = await self.community_storage.load_communities(folder_id)
            if communities:
                self.cached_communities[cache_key] = communities
                return communities
        
        # Fall back to detection
        logger.info("No stored communities found, detecting new ones...")
        return await self.detect_communities(folder_id)

    async def get_communities_by_topic(self, topic_keywords: List[str], 
                                    folder_id: str = None) -> List[Community]:
        """Get communities that match specific topic keywords"""
        
        all_communities = await self.get_all_communities(folder_id)
        
        matching_communities = []
        for community in all_communities:
            # Check if any topic keywords appear in summary or entity names
            text_to_search = f"{community.summary} {' '.join(community.entities)}".lower()
            
            relevance_score = 0
            for keyword in topic_keywords:
                if keyword.lower() in text_to_search:
                    relevance_score += 1
            
            if relevance_score > 0:
                matching_communities.append((community, relevance_score))
        
        # Sort by relevance and return
        matching_communities.sort(key=lambda x: x[1], reverse=True)
        return [comm for comm, score in matching_communities]

    async def get_community_statistics(self, folder_id: str = None) -> Dict[str, Any]:
        """Get detailed statistics about detected communities"""
        
        communities = await self.get_all_communities(folder_id)
        
        if not communities:
            return {
                'total_communities': 0,
                'message': 'No communities detected yet'
            }
        
        # Calculate statistics
        sizes = [c.size for c in communities]
        entity_distribution = {}
        
        for community in communities:
            for entity in community.entities:
                if entity not in entity_distribution:
                    entity_distribution[entity] = []
                entity_distribution[entity].append(community.community_id)
        
        # Find entities that appear in multiple communities
        cross_community_entities = {
            entity: comms for entity, comms in entity_distribution.items() 
            if len(comms) > 1
        }
        
        return {
            'total_communities': len(communities),
            'size_distribution': {
                'min': min(sizes),
                'max': max(sizes),
                'average': sum(sizes) / len(sizes),
                'sizes': sizes
            },
            'total_entities_covered': len(entity_distribution),
            'cross_community_entities': len(cross_community_entities),
            'community_summaries': [
                {
                    'id': c.community_id,
                    'size': c.size,
                    'summary': c.summary
                } for c in communities
            ],
            'overlapping_entities': cross_community_entities
        }

    async def global_search(self, user_query: str, folder_id: str = None,
                      max_communities: int = 5) -> Dict[str, Any]:
        """GraphRAG Global Search: Use communities for high-level understanding"""
        
        start_time = time.time()
        
        # Get communities using the new access method
        communities = await self.get_all_communities(folder_id)
        
        if not communities:
            logger.info("No communities found, falling back to local search")
            return await self.local_search(user_query, folder_id)
        
        # Score communities by relevance to query
        community_scores = []
        query_lower = user_query.lower()
        
        for community in communities:
            score = 0.0
            
            # Score based on entity names
            for entity in community.entities:
                if any(word in entity.lower() for word in query_lower.split()):
                    score += 1.0
            
            # Score based on community summary
            summary_words = community.summary.lower().split()
            query_words = query_lower.split()
            overlap = len(set(summary_words) & set(query_words))
            score += overlap * 0.5
            
            community_scores.append((community, score))
        
        # Select top communities
        relevant_communities = sorted(community_scores, key=lambda x: x[1], reverse=True)
        selected_communities = [comm for comm, score in relevant_communities[:max_communities] if score > 0]
        
        if not selected_communities:
            logger.info("No relevant communities found, falling back to local search")
            return await self.local_search(user_query, folder_id)
        
        # Generate global answer from community summaries
        global_context = []
        for community in selected_communities:
            global_context.append(f"Community: {community.summary}")
        
        answer = self._synthesize_global_answer(user_query, global_context, selected_communities)
        
        return {
            'query': user_query,
            'search_mode': SearchMode.GLOBAL.value,
            'answer': answer,
            'communities_used': len(selected_communities),
            'total_entities_covered': sum(c.size for c in selected_communities),
            'processing_time': time.time() - start_time,
            'confidence': min(len(selected_communities) / max_communities, 1.0),
            'selected_communities': [c.community_id for c in selected_communities]
        }

    async def local_search(self, user_query: str, folder_id: str = None,
                     max_entities: int = 10) -> Dict[str, Any]:
        """GraphRAG Local Search: Entity-focused detailed search"""
        
        start_time = time.time()
        
        # Use Graphiti's search to find relevant content
        if not self.graph_builder.graphiti_client:
            raise RuntimeError("Graphiti client not available")
        
        try:
            # Search using Graphiti - remove the 'limit' parameter
            search_results = await self.graph_builder.graphiti_client.search(
                query=user_query
            )
            
            # Limit results manually if needed
            if search_results and len(search_results) > max_entities:
                search_results = search_results[:max_entities]
            
            if not search_results:
                return {
                    'query': user_query,
                    'search_mode': SearchMode.LOCAL.value,
                    'answer': "No relevant information found in the knowledge graph.",
                    'confidence': 0.0,
                    'entities_found': 0,
                    'processing_time': time.time() - start_time
                }
            
            # Extract and process results
            answer_parts = []
            entities_found = 0
            
            for i, result in enumerate(search_results[:5]):
                if hasattr(result, 'episode') and result.episode:
                    episode = result.episode
                    content = getattr(episode, 'episode_body', '') or getattr(episode, 'content', '')
                    source = getattr(episode, 'source_description', 'Unknown source')
                    
                    if content:
                        answer_parts.append(f"{i+1}. From {source}:")
                        answer_parts.append(f"   {content[:300]}...")
                        entities_found += 1
                
                elif isinstance(result, dict):
                    content = result.get('content', result.get('episode_body', ''))
                    source = result.get('source_description', result.get('source', 'Unknown source'))
                    
                    if content:
                        answer_parts.append(f"{i+1}. From {source}:")
                        answer_parts.append(f"   {content[:300]}...")
                        entities_found += 1
            
            if not answer_parts:
                answer = "Search completed but no detailed content found."
            else:
                answer = f"Found {entities_found} relevant sections:\n\n" + "\n".join(answer_parts)
            
            confidence = min(entities_found / max_entities, 1.0)
            
        except Exception as e:
            logger.error(f"Error in Graphiti search: {e}")
            answer = f"Search error: {str(e)}"
            confidence = 0.0
            entities_found = 0
        
        return {
            'query': user_query,
            'search_mode': SearchMode.LOCAL.value,
            'answer': answer,
            'entities_found': entities_found,
            'confidence': confidence,
            'processing_time': time.time() - start_time
        }

    async def hybrid_search(self, user_query: str, folder_id: str = None) -> Dict[str, Any]:
        """GraphRAG Hybrid Search: Combine global and local approaches"""
        
        start_time = time.time()
        
        # Run both searches
        global_result = await self.global_search(user_query, folder_id, max_communities=3)
        local_result = await self.local_search(user_query, folder_id, max_entities=8)
        
        # Combine results intelligently
        combined_answer = self._combine_global_local_answers(
            user_query, global_result, local_result
        )
        
        return {
            'query': user_query,
            'search_mode': SearchMode.HYBRID.value,
            'answer': combined_answer,
            'global_confidence': global_result.get('confidence', 0.0),
            'local_confidence': local_result.get('confidence', 0.0),
            'combined_confidence': (global_result.get('confidence', 0.0) + 
                                  local_result.get('confidence', 0.0)) / 2,
            'processing_time': time.time() - start_time,
            'communities_used': global_result.get('communities_used', 0),
            'entities_found': local_result.get('entities_found', 0)
        }

    def _combine_global_local_answers(self, user_query: str,
                                    global_result: Dict, 
                                    local_result: Dict) -> str:
        """Combine global and local search results intelligently"""
        
        global_answer = global_result.get('answer', '')
        local_answer = local_result.get('answer', '')
        
        if not global_answer and not local_answer:
            return "No relevant information found using either global or local search."
        
        combined_parts = []
        
        # Decide which approach provides better coverage
        global_conf = global_result.get('confidence', 0.0)
        local_conf = local_result.get('confidence', 0.0)
        
        if global_conf > 0.3:
            combined_parts.append("=== OVERVIEW ===")
            combined_parts.append(global_answer)
        
        if local_conf > 0.3:
            combined_parts.append("\n=== DETAILED INFORMATION ===")
            combined_parts.append(local_answer)
        
        if not combined_parts:
            # Fallback: use whichever has content
            if global_answer:
                return global_answer
            else:
                return local_answer
        
        return "\n".join(combined_parts)

    def _synthesize_global_answer(self, user_query: str, 
                                global_context: List[str], 
                                communities: List[Community]) -> str:
        """Synthesize answer using global community-level understanding"""
        
        if not global_context:
            return "No relevant communities found for this query."
        
        # Build comprehensive answer from community summaries
        intro = f"Based on the overall knowledge structure, here's what I found about '{user_query}':\n\n"
        
        main_insights = []
        for i, context in enumerate(global_context[:3]):
            main_insights.append(f"{i+1}. {context}")
        
        community_details = []
        for community in communities[:3]:
            if community.size > 5:  # Focus on larger communities
                community_details.append(
                    f"A cluster of {community.size} related concepts including: "
                    f"{', '.join(community.entities[:5])}"
                )
        
        answer_parts = [intro]
        if main_insights:
            answer_parts.append("Main themes identified:")
            answer_parts.extend(main_insights)
        
        if community_details:
            answer_parts.append("\nRelated concept clusters:")
            answer_parts.extend(community_details)
        
        return "\n".join(answer_parts)

    def _calculate_local_confidence(self, local_context: Dict[str, Any]) -> float:
        """Calculate confidence for local search results"""
        entity_score = min(len(local_context['entities']) / 5, 1.0)
        episode_score = min(len(local_context['episodes']) / 3, 1.0)
        relationship_score = min(len(local_context['relationships']) / 10, 1.0)
        
        return (entity_score + episode_score + relationship_score) / 3

    def _recommend_search_modes(self, entity_count: int, connectivity: float) -> List[str]:
        """Recommend optimal search modes based on graph characteristics"""
        
        recommendations = []
        
        if entity_count >= 20 and connectivity >= 0.2:
            recommendations.append(SearchMode.GLOBAL.value)
        
        if entity_count >= 5:
            recommendations.append(SearchMode.LOCAL.value)
        
        if len(recommendations) > 1:
            recommendations.append(SearchMode.HYBRID.value)
        
        return recommendations or [SearchMode.LOCAL.value]  # Fallback

    def _analyze_query_type(self, user_query: str) -> Dict[str, Any]:
        """Analyze query to determine optimal search strategy"""
        
        query_lower = user_query.lower()
        
        # Broad scope indicators
        broad_indicators = ['overview', 'summary', 'main', 'general', 'overall', 'topics', 'themes']
        broad_score = sum(1 for indicator in broad_indicators if indicator in query_lower)
        
        # Specific scope indicators  
        specific_indicators = ['specific', 'detail', 'explain', 'how', 'why', 'what is']
        specific_score = sum(1 for indicator in specific_indicators if indicator in query_lower)
        
        # Question type analysis
        question_words = ['what', 'how', 'why', 'when', 'where', 'who']
        is_question = any(word in query_lower for word in question_words)
        
        # Determine scope
        if broad_score > specific_score:
            scope = 'broad'
        elif specific_score > broad_score:
            scope = 'specific'
        else:
            scope = 'mixed'
        
        return {
            'scope': scope,
            'is_question': is_question,
            'broad_score': broad_score,
            'specific_score': specific_score,
            'complexity': len(query_lower.split()),
            'recommended_mode': SearchMode.GLOBAL.value if scope == 'broad' else SearchMode.LOCAL.value
        }

    async def answer_query_intelligently(self, user_query: str, 
                                       folder_id: str = None,
                                       preferred_mode: SearchMode = SearchMode.HYBRID) -> Dict[str, Any]:
        """Intelligent query answering with automatic mode selection"""
        
        # Analyze query to determine best approach
        query_analysis = self._analyze_query_type(user_query)
        
        if preferred_mode == SearchMode.HYBRID:
            # Auto-select based on query analysis
            if query_analysis['scope'] == 'broad':
                result = await self.global_search(user_query, folder_id)
            elif query_analysis['scope'] == 'specific':
                result = await self.local_search(user_query, folder_id)
            else:
                result = await self.hybrid_search(user_query, folder_id)
        elif preferred_mode == SearchMode.GLOBAL:
            result = await self.global_search(user_query, folder_id)
        else:
            result = await self.local_search(user_query, folder_id)
        
        # Add query analysis to result
        result['query_analysis'] = query_analysis
        
        return result

    async def close(self):
        """Close connections and cleanup"""
        if self.driver:
            await self.driver.close()
            logger.info("Neo4j driver closed")


# Additional utility functions for GraphRAG pipeline integration

async def create_enhanced_graphrag_pipeline(neo4j_config: Dict[str, Any], 
                                          google_api_key: str) -> Tuple[FolderScopedGraphBuilder, DirectGraphRAGExtractor]:
    """Create and initialize both graph builder and extractor"""
    
    # Initialize graph builder
    graph_builder = FolderScopedGraphBuilder(
        neo4j_config=neo4j_config,
        google_api_key=google_api_key
    )
    await graph_builder.initialize()
    
    # Initialize GraphRAG extractor
    graphrag_extractor = DirectGraphRAGExtractor(neo4j_config, graph_builder)
    await graphrag_extractor.initialize()
    
    return graph_builder, graphrag_extractor

async def complete_graphrag_setup_and_test(folder_path: str, 
                                         neo4j_config: Dict[str, Any],
                                         google_api_key: str,
                                         test_query: str = "What are the main topics?") -> Dict[str, Any]:
    """Complete GraphRAG setup with proper timing and testing"""
    
    graph_builder, graphrag_extractor = await create_enhanced_graphrag_pipeline(
        neo4j_config, google_api_key
    )
    
    try:
        # Build the graph
        logger.info("Building knowledge graph...")
        build_result = await graph_builder.process_folder(
            folder_path=folder_path,
            chunk_size=1200,
            overlap_size=120,
            batch_size=2
        )
        
        # Wait for entity extraction
        logger.info("Waiting for entity extraction to complete...")
        extraction_ready = await graphrag_extractor.wait_for_entity_extraction(max_wait_time=120)
        
        # Get updated statistics
        logger.info("Getting graph statistics...")
        graph_stats = await graphrag_extractor.get_folder_graph_statistics(
            build_result.get('folder_id')
        )
        
        # Detect communities if ready
        communities = []
        if graph_stats['community_detection_viable']:
            logger.info("Detecting communities...")
            communities = await graphrag_extractor.detect_communities(
                build_result.get('folder_id')
            )
        
        # Test query
        logger.info(f"Testing query: '{test_query}'")
        test_result = await graphrag_extractor.answer_query_intelligently(
            test_query, 
            build_result.get('folder_id')
        )
        
        return {
            'status': 'SUCCESS',
            'build_result': build_result,
            'extraction_ready': extraction_ready,
            'graph_statistics': graph_stats,
            'communities_detected': len(communities),
            'community_details': [{'id': c.community_id, 'size': c.size, 'summary': c.summary} 
                                for c in communities],
            'test_query_result': test_result,
            'graphrag_status': graph_stats['graphrag_readiness'],
            'recommended_modes': graph_stats['recommended_search_modes']
        }
        
    finally:
        await graph_builder.cleanup()
        await graphrag_extractor.close()