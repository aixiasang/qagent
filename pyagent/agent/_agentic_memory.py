import json
import re
import uuid
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, AsyncGenerator, Union
from datetime import datetime
from dataclasses import dataclass, field, asdict
import logging
from ..core import Agent,Memory, ChatResponse, ChaterPool, EmbedderPool, Embedder,Chater,ToolKit,Speaker,ChromaVectorStore

from ..prompt import (
    get_agentic_memory_analyze_prompt,
    get_agentic_memory_evolution_prompt,
)

logger = logging.getLogger(__name__)


@dataclass
class AgenticMemoryNote:
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    keywords: List[str] = field(default_factory=list)
    links: List[str] = field(default_factory=list)
    context: str = field(default="General")
    category: str = field(default="Uncategorized")
    tags: List[str] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    last_accessed: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d%H%M"))
    retrieval_count: int = field(default=0)
    evolution_history: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class AgenticMemoryAgent(Agent):
    def __init__(
        self,
        name: str,
        chater: Union[Chater, ChaterPool],
        embedder: Union[Embedder, EmbedderPool],
        memory: Memory,
        tools: Optional[ToolKit] = None,
        system_prompt: Optional[str] = None,
        max_iterations: int = 5,
        tool_timeout: Optional[int] = None,
        enable_logging: bool = False,
        log_file: Optional[str] = None,
        log_level: str = "INFO",
        speaker: Optional[Speaker] = None,
        evo_threshold: int = 100,
        vector_store_path: str = "./agentic_memory_store",
        collection_name: str = "memories",
    ):
        super().__init__(
            name=name,
            chater=chater,
            memory=memory,
            tools=tools,
            system_prompt=system_prompt,
            max_iterations=max_iterations,
            tool_timeout=tool_timeout,
            enable_logging=enable_logging,
            log_file=log_file,
            log_level=log_level,
            speaker=speaker,
        )
        
        self.embedder = embedder
        self.agentic_memories: Dict[str, AgenticMemoryNote] = {}
        self.vector_store = ChromaVectorStore(vector_store_path, collection_name)
        self.evo_counter = 0
        self.evo_threshold = evo_threshold
    
    def _create_enhanced_document(self, note: AgenticMemoryNote) -> str:
        enhanced = note.content
        
        if note.context and note.context != "General":
            enhanced += f" context: {note.context}"
        
        if note.keywords:
            enhanced += f" keywords: {', '.join(note.keywords)}"
        
        if note.tags:
            enhanced += f" tags: {', '.join(note.tags)}"
        
        return enhanced
    
    def _serialize_metadata(self, note: AgenticMemoryNote) -> Dict[str, str]:
        metadata = {}
        for key, value in note.to_dict().items():
            if isinstance(value, list):
                metadata[key] = json.dumps(value)
            elif isinstance(value, dict):
                metadata[key] = json.dumps(value)
            else:
                metadata[key] = str(value)
        return metadata
    
    async def _analyze_content(self, content: str) -> Dict:
        prompt = get_agentic_memory_analyze_prompt(content)
        
        try:
            response = await self.chater.chat(
                messages=[
                    {"role": "system", "content": "You must respond with a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            response_text = response.content
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                return result
            
            return {"keywords": [], "context": "General", "tags": []}
            
        except Exception as e:
            self.logger.error(f"Error analyzing content: {e}")
            return {"keywords": [], "context": "General", "tags": []}
    
    async def _search_similar(self, query: str, k: int = 5) -> List[Tuple[str, float]]:
        try:
            embed_response = await self.embedder.embed([query])
            query_embedding = np.array(embed_response.embedding).flatten()
            
            search_results = await self.vector_store.search(query_embedding, k=k)
            
            results = []
            for result in search_results:
                results.append((result.id, result.score))
            
            return results
        except Exception as e:
            self.logger.error(f"Error in similarity search: {e}")
            return []
    
    async def _decide_evolution(
        self,
        new_note: AgenticMemoryNote,
        neighbors: List[AgenticMemoryNote]
    ) -> Dict:
        if not neighbors:
            return {
                "should_evolve": False,
                "actions": [],
                "suggested_connections": [],
                "tags_to_update": [],
                "new_context_neighborhood": [],
                "new_tags_neighborhood": []
            }
        
        neighbors_text = ""
        for i, neighbor in enumerate(neighbors):
            neighbors_text += f"memory index:{i}\ttalk start time:{neighbor.timestamp}\tmemory content: {neighbor.content}\tmemory context: {neighbor.context}\tmemory keywords: {str(neighbor.keywords)}\tmemory tags: {str(neighbor.tags)}\n"
        
        prompt = get_agentic_memory_evolution_prompt(
            content=new_note.content,
            context=new_note.context,
            keywords=new_note.keywords,
            nearest_neighbors=neighbors_text,
            neighbor_number=len(neighbors)
        )
        
        try:
            response = await self.chater.chat(
                messages=[
                    {"role": "system", "content": "You must respond with a valid JSON object."},
                    {"role": "user", "content": prompt}
                ],
                stream=False
            )
            
            response_text = response.content
            
            json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                result = json.loads(json_str)
                
                required_keys = [
                    "should_evolve", "actions", "suggested_connections",
                    "tags_to_update", "new_context_neighborhood", "new_tags_neighborhood"
                ]
                
                for key in required_keys:
                    if key not in result:
                        result[key] = [] if key != "should_evolve" else False
                
                return result
            
            return {
                "should_evolve": False,
                "actions": [],
                "suggested_connections": [],
                "tags_to_update": [],
                "new_context_neighborhood": [],
                "new_tags_neighborhood": []
            }
            
        except Exception as e:
            self.logger.error(f"Error in evolution decision: {e}")
            return {
                "should_evolve": False,
                "actions": [],
                "suggested_connections": [],
                "tags_to_update": [],
                "new_context_neighborhood": [],
                "new_tags_neighborhood": []
            }
    
    async def _process_memory(self, note: AgenticMemoryNote) -> bool:
        if not self.agentic_memories:
            return False
        
        try:
            neighbor_ids = await self._search_similar(note.content, k=5)
            
            if not neighbor_ids:
                return False
            
            neighbors = [self.agentic_memories[nid] for nid, _ in neighbor_ids if nid in self.agentic_memories]
            
            if not neighbors:
                return False
            
            decision = await self._decide_evolution(note, neighbors)
            
            should_evolve = decision.get("should_evolve", False)
            
            if should_evolve:
                actions = decision.get("actions", [])
                
                for action in actions:
                    if action == "strengthen":
                        suggested_connections = decision.get("suggested_connections", [])
                        new_tags = decision.get("tags_to_update", [])
                        
                        for conn_id in suggested_connections:
                            if conn_id not in note.links:
                                note.links.append(conn_id)
                        
                        if new_tags:
                            note.tags = new_tags
                    
                    elif action == "update_neighbor":
                        new_context_list = decision.get("new_context_neighborhood", [])
                        new_tags_list = decision.get("new_tags_neighborhood", [])
                        
                        for i, neighbor in enumerate(neighbors):
                            if i < len(new_tags_list):
                                neighbor.tags = new_tags_list[i]
                            if i < len(new_context_list):
                                neighbor.context = new_context_list[i]
                            
                            self.agentic_memories[neighbor.id] = neighbor
            
            return should_evolve
            
        except Exception as e:
            self.logger.error(f"Error in process_memory: {e}")
            return False
    
    async def _consolidate_memories(self):
        try:
            await self.vector_store.clear()
            
            for memory in self.agentic_memories.values():
                try:
                    enhanced_doc = self._create_enhanced_document(memory)
                    embed_response = await self.embedder.embed([enhanced_doc])
                    embedding = np.array(embed_response.embedding).flatten()
                    
                    metadata = self._serialize_metadata(memory)
                    
                    await self.vector_store.add(
                        ids=[memory.id],
                        texts=[memory.content],
                        embeddings=np.array([embedding]),
                        metadatas=[metadata]
                    )
                except Exception as e:
                    self.logger.error(f"Failed to consolidate memory {memory.id}: {e}")
        except Exception as e:
            self.logger.error(f"Failed to consolidate memories: {e}")
    
    async def add_memory_note(
        self,
        content: str,
        keywords: Optional[List[str]] = None,
        context: Optional[str] = None,
        tags: Optional[List[str]] = None,
        timestamp: Optional[str] = None,
        **kwargs
    ) -> str:
        note = AgenticMemoryNote(
            content=content,
            keywords=keywords or [],
            context=context or "General",
            tags=tags or [],
            timestamp=timestamp,
            **kwargs
        )
        
        needs_analysis = (
            not note.keywords or
            note.context == "General" or
            not note.tags
        )
        
        if needs_analysis:
            try:
                analysis = await self._analyze_content(content)
                
                if not note.keywords:
                    note.keywords = analysis.get("keywords", [])
                if note.context == "General":
                    note.context = analysis.get("context", "General")
                if not note.tags:
                    note.tags = analysis.get("tags", [])
                    
            except Exception as e:
                self.logger.warning(f"LLM analysis failed: {e}")
        
        should_evolve = await self._process_memory(note)
        
        self.agentic_memories[note.id] = note
        
        try:
            enhanced_doc = self._create_enhanced_document(note)
            embed_response = await self.embedder.embed([enhanced_doc])
            embedding = np.array(embed_response.embedding).flatten()
            
            metadata = self._serialize_metadata(note)
            
            await self.vector_store.add(
                ids=[note.id],
                texts=[note.content],
                embeddings=np.array([embedding]),
                metadatas=[metadata]
            )
        except Exception as e:
            self.logger.error(f"Failed to add memory to vector store: {e}")
        
        if should_evolve:
            self.evo_counter += 1
            if self.evo_counter % self.evo_threshold == 0:
                await self._consolidate_memories()
        
        return note.id
    
    async def search_memory(
        self,
        query: str,
        k: int = 5,
        include_neighbors: bool = True
    ) -> List[Dict[str, Any]]:
        if not self.agentic_memories:
            return []
        
        try:
            results = await self._search_similar(query, k)
            
            memories = []
            seen_ids = set()
            
            for doc_id, score in results:
                if doc_id not in seen_ids and doc_id in self.agentic_memories:
                    memory = self.agentic_memories[doc_id]
                    memories.append({
                        'id': memory.id,
                        'content': memory.content,
                        'context': memory.context,
                        'keywords': memory.keywords,
                        'tags': memory.tags,
                        'timestamp': memory.timestamp,
                        'score': score,
                        'is_neighbor': False
                    })
                    seen_ids.add(doc_id)
            
            if include_neighbors:
                neighbor_count = 0
                for memory_dict in list(memories):
                    if neighbor_count >= k:
                        break
                    
                    memory_obj = self.agentic_memories.get(memory_dict['id'])
                    if memory_obj and memory_obj.links:
                        for link_id in memory_obj.links:
                            if link_id not in seen_ids and link_id in self.agentic_memories and neighbor_count < k:
                                neighbor = self.agentic_memories[link_id]
                                memories.append({
                                    'id': neighbor.id,
                                    'content': neighbor.content,
                                    'context': neighbor.context,
                                    'keywords': neighbor.keywords,
                                    'tags': neighbor.tags,
                                    'timestamp': neighbor.timestamp,
                                    'score': 0.0,
                                    'is_neighbor': True
                                })
                                seen_ids.add(link_id)
                                neighbor_count += 1
            
            return memories[:k]
            
        except Exception as e:
            self.logger.error(f"Error in search_memory: {e}")
            return []
    
    def read_memory(self, memory_id: str) -> Optional[AgenticMemoryNote]:
        return self.agentic_memories.get(memory_id)
    
    async def update_memory(self, memory_id: str, **kwargs) -> bool:
        if memory_id not in self.agentic_memories:
            return False
        
        note = self.agentic_memories[memory_id]
        
        for key, value in kwargs.items():
            if hasattr(note, key):
                setattr(note, key, value)
        
        try:
            await self.vector_store.delete([memory_id])
            
            enhanced_doc = self._create_enhanced_document(note)
            embed_response = await self.embedder.embed([enhanced_doc])
            embedding = np.array(embed_response.embedding).flatten()
            
            metadata = self._serialize_metadata(note)
            
            await self.vector_store.add(
                ids=[note.id],
                texts=[note.content],
                embeddings=np.array([embedding]),
                metadatas=[metadata]
            )
        except Exception as e:
            self.logger.error(f"Failed to update memory in vector store: {e}")
        
        return True
    
    async def delete_memory(self, memory_id: str) -> bool:
        if memory_id in self.agentic_memories:
            try:
                await self.vector_store.delete([memory_id])
            except Exception as e:
                self.logger.error(f"Failed to delete from vector store: {e}")
            
            del self.agentic_memories[memory_id]
            return True
        return False
    
    async def reply_with_memory(
        self,
        user_message: str,
        k: int = 3,
        stream: bool = False,
        auto_speak: bool = True,
    ) -> AsyncGenerator[ChatResponse, None]:
        
        related_memories = await self.search_memory(user_message, k=k)
        
        if related_memories:
            memory_context = "\n\nRelevant memories:\n"
            for mem in related_memories:
                memory_context += f"- [{mem['timestamp']}] {mem['content'][:100]}"
                if mem['keywords']:
                    memory_context += f" (keywords: {', '.join(mem['keywords'][:3])})"
                memory_context += "\n"
            
            enhanced_message = user_message + memory_context
            
            async for response in super().reply(enhanced_message, stream=stream, auto_speak=auto_speak):
                yield response
        else:
            async for response in super().reply(user_message, stream=stream, auto_speak=auto_speak):
                yield response
    
    def get_all_memories(self) -> List[AgenticMemoryNote]:
        return list(self.agentic_memories.values())
    
    def get_memory_stats(self) -> Dict[str, Any]:
        total = len(self.agentic_memories)
        
        total_links = sum(len(m.links) for m in self.agentic_memories.values())
        
        all_tags = []
        for m in self.agentic_memories.values():
            all_tags.extend(m.tags)
        
        unique_tags = len(set(all_tags))
        
        return {
            "total_memories": total,
            "total_links": total_links,
            "unique_tags": unique_tags,
            "evolution_count": self.evo_counter,
            "avg_links_per_memory": total_links / total if total > 0 else 0
        }
