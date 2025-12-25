"""
Dionysus Memory Provider

Integrates MemEvolve with Graphiti (Neo4j knowledge graph) for:
- Persistent graph memory with entity relationships
- Semantic search across agent experiences
- Cross-agent knowledge sharing
- Session continuity

Architecture (Simplified):
- Single backend: Graphiti/Neo4j (no PostgreSQL)
- Direct API calls to Graphiti service
- Entity extraction from trajectories
- Local cache fallback when Graphiti unavailable
"""

import asyncio
import json
import logging
import os
from typing import Any, Dict, List, Optional
from uuid import uuid4

from EvolveLab.base_memory import BaseMemoryProvider
from EvolveLab.memory_types import (
    MemoryItem,
    MemoryItemType,
    MemoryRequest,
    MemoryResponse,
    MemoryStatus,
    MemoryType,
    TrajectoryData,
)

# Import Graphiti client (simplified)
from .graphiti_client import GraphitiClient, GraphitiConfig, GraphitiResponse

logger = logging.getLogger(__name__)


# Memory type mapping for Graphiti entity types
MEMEVOLVE_TO_GRAPHITI_TYPES = {
    "strategic": "Pattern",
    "operational": "Procedure",
    "short-term": "Context",
    "workflow": "Episode",
    "cold-start": "Fact",
}


class DionysusMemoryProvider(BaseMemoryProvider):
    """
    Memory provider that integrates with Graphiti (Neo4j knowledge graph).

    Provides persistent graph memory storage via direct Graphiti API.
    Falls back to local JSON cache when Graphiti is unavailable.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the Dionysus memory provider.

        Args:
            config: Optional configuration dict (overrides from environment).
        """
        super().__init__(memory_type=MemoryType.DIONYSUS, config=config or {})

        # Logger
        self.logger = logging.getLogger(f"{__name__}.Dionysus")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [Graphiti] [%(levelname)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

        # Load Graphiti configuration from environment
        self.graphiti_config = GraphitiConfig.from_environment()

        # Client (initialized lazily)
        self._client: Optional[GraphitiClient] = None

        # Session tracking
        self._session_id: Optional[str] = None
        self._step_count: int = 0

        # Local cache for fallback
        cache_dir = self.config.get("local_cache_dir", "./storage/graphiti/cache")
        self._cache_dir = cache_dir
        self._cache_file = os.path.join(cache_dir, "fallback_cache.json")
        self._cache: Dict[str, Any] = {}

        # Feature flag
        self._enabled = self.graphiti_config.enabled

    def _get_client(self) -> GraphitiClient:
        """Get or create the Graphiti client."""
        if self._client is None:
            self._client = GraphitiClient(self.graphiti_config)
        return self._client

    def _run_async(self, coro):
        """Run an async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If already in async context, create a new task
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    def initialize(self) -> bool:
        """
        Initialize the Dionysus memory provider.

        - Validates configuration
        - Tests connectivity to Graphiti (if enabled)
        - Loads local cache for fallback

        Returns:
            True if initialization successful.
        """
        self.logger.info("Initializing DionysusMemoryProvider (Graphiti backend)...")

        # Validate configuration
        errors = self.graphiti_config.validate()
        if errors:
            if self._enabled:
                self.logger.error(f"Configuration errors: {errors}")
                self.logger.warning("Falling back to local cache only")
                self._enabled = False
            else:
                self.logger.info("Graphiti disabled - local cache only mode")

        # Ensure cache directory exists
        os.makedirs(self._cache_dir, exist_ok=True)

        # Load existing cache
        if os.path.exists(self._cache_file):
            try:
                with open(self._cache_file, "r") as f:
                    self._cache = json.load(f)
                self.logger.info(f"Loaded {len(self._cache)} cached memories")
            except Exception as e:
                self.logger.warning(f"Failed to load cache: {e}")
                self._cache = {}

        # Test connectivity if enabled
        if self._enabled:
            try:
                response = self._run_async(self._get_client().health())
                if response.success:
                    self.logger.info(
                        f"Graphiti connection verified ({response.latency_ms:.0f}ms)"
                    )
                else:
                    self.logger.warning(
                        f"Graphiti health check failed: {response.error}"
                    )
                    self.logger.info("Will use local cache as fallback")
            except Exception as e:
                self.logger.warning(f"Graphiti connectivity test failed: {e}")

        # Generate session ID
        self._session_id = str(uuid4())
        self.logger.info(f"Session ID: {self._session_id}")

        return True

    def provide_memory(self, request: MemoryRequest) -> MemoryResponse:
        """
        Retrieve memories from Graphiti based on request.

        BEGIN phase: Retrieve patterns and facts
        IN phase: Retrieve episodes and context

        Args:
            request: MemoryRequest with query, context, and status

        Returns:
            MemoryResponse with relevant memories
        """
        self._step_count += 1

        # Determine entity types based on phase
        if request.status == MemoryStatus.BEGIN:
            entity_types = ["Pattern", "Fact", "Procedure"]
        else:  # IN phase
            entity_types = ["Episode", "Context"]

        memories: List[MemoryItem] = []

        # Try Graphiti first if enabled
        if self._enabled:
            try:
                response = self._run_async(
                    self._get_client().search(
                        query=request.query,
                        limit=self.config.get("top_k", 5),
                        entity_types=entity_types,
                    )
                )

                if response.success and response.data:
                    memories = self._parse_search_response(response.data)
                    self.logger.debug(
                        f"Retrieved {len(memories)} memories from Graphiti "
                        f"({response.latency_ms:.0f}ms)"
                    )

                    # Update cache with fresh results
                    cache_key = self._make_cache_key(request.query, entity_types)
                    self._cache[cache_key] = {
                        "memories": [self._memory_to_dict(m) for m in memories],
                        "timestamp": self._step_count,
                    }
                    self._save_cache()

                else:
                    self.logger.warning(f"Graphiti search failed: {response.error}")
                    memories = self._get_cached_memories(request.query, entity_types)

            except Exception as e:
                self.logger.warning(f"Graphiti search error: {e}")
                memories = self._get_cached_memories(request.query, entity_types)
        else:
            # Use local cache only
            memories = self._get_cached_memories(request.query, entity_types)

        return MemoryResponse(
            memories=memories,
            memory_type=self.memory_type,
            total_count=len(memories),
            request_id=str(uuid4()),
        )

    def take_in_memory(self, trajectory_data: TrajectoryData) -> tuple[bool, str]:
        """
        Ingest trajectory into Graphiti as an episode.

        Graphiti automatically extracts entities and relationships from
        natural language episode text. This is more robust than manual
        entity extraction.

        Args:
            trajectory_data: TrajectoryData with query, trajectory, result, metadata

        Returns:
            (success, description) tuple
        """
        if not trajectory_data.trajectory:
            return False, "Empty trajectory"

        # Format trajectory as episode text (Graphiti extracts entities automatically)
        episode_content = self._format_trajectory_as_episode(trajectory_data)

        # Build metadata
        metadata = {
            "source": f"memevolve:{self._session_id}",
            "session_id": self._session_id,
            "step_count": self._step_count,
            "success": getattr(trajectory_data, "success", True),
            **(trajectory_data.metadata or {}),
        }

        # Send to Graphiti if enabled
        if self._enabled:
            try:
                response = self._run_async(
                    self._get_client().add_episode(
                        content=episode_content,
                        metadata=metadata,
                    )
                )

                if response.success:
                    data = response.data or {}
                    nodes = data.get("nodes", [])
                    edges = data.get("edges", [])
                    episode_uuid = data.get("episode_uuid", "unknown")

                    summary = f"Episode {episode_uuid[:8]}: {len(nodes)} entities, {len(edges)} relationships"
                    self.logger.info(f"Ingested to Graphiti ({response.latency_ms:.0f}ms): {summary}")

                    # Log extracted entities for debugging
                    if nodes:
                        entity_names = [n.get("name", "?") for n in nodes[:5]]
                        self.logger.debug(f"Extracted entities: {', '.join(entity_names)}")

                    return True, summary
                else:
                    self.logger.warning(f"Graphiti add_episode failed: {response.error}")
                    # Fall through to local cache

            except Exception as e:
                self.logger.warning(f"Graphiti add_episode error: {e}")
                # Fall through to local cache

        # Store in local cache as fallback
        cache_key = f"episode_{uuid4().hex[:8]}"
        self._cache[cache_key] = {
            "type": "episode",
            "content": episode_content,
            "metadata": metadata,
            "pending_sync": True,
        }
        self._save_cache()

        return True, f"Episode cached locally (pending sync: {cache_key})"

    def _format_trajectory_as_episode(self, trajectory_data: TrajectoryData) -> str:
        """
        Format trajectory data as natural language episode text.

        Graphiti's LLM will extract entities and relationships from this text.
        The richer and more descriptive, the better the extraction.

        Args:
            trajectory_data: The trajectory to format

        Returns:
            Natural language episode text
        """
        # Build step narrative
        steps = trajectory_data.trajectory
        if isinstance(steps, list):
            if len(steps) <= 5:
                steps_text = " → ".join(str(s) for s in steps)
            else:
                # Summarize long trajectories
                steps_text = f"{steps[0]} → ... ({len(steps)} steps) → {steps[-1]}"
        else:
            steps_text = str(steps)

        # Build result text
        result = getattr(trajectory_data, "result", None)
        result_text = str(result) if result else "Task completed"

        # Build success indicator
        success = getattr(trajectory_data, "success", True)
        outcome = "succeeded" if success else "failed"

        # Format as natural language episode
        episode = f"""Task: {trajectory_data.query}
Agent {outcome} with the following approach:
Steps taken: {steps_text}
Result: {result_text}"""

        # Add learning if available
        learning = (trajectory_data.metadata or {}).get("learning")
        if learning:
            episode += f"\nLearning: {learning}"

        return episode

    # -------------------------------------------------------------------------
    # Helper Methods
    # -------------------------------------------------------------------------

    def _parse_search_response(self, data: Dict[str, Any]) -> List[MemoryItem]:
        """Parse Graphiti search response into MemoryItems."""
        memories = []
        items = data.get("results", data.get("entities", []))

        for item in items:
            memory = MemoryItem(
                id=item.get("id", str(uuid4())),
                content=item.get("name", item.get("content", "")),
                metadata={
                    "source": "graphiti",
                    "entity_type": item.get("type", "unknown"),
                    "properties": item.get("properties", {}),
                    "score": item.get("score", 0.0),
                },
                score=item.get("score", 0.0),
                type=MemoryItemType.TEXT,
            )
            memories.append(memory)

        return memories

    def _make_cache_key(self, query: str, memory_types: List[str]) -> str:
        """Generate a cache key for a query."""
        import hashlib
        key_str = f"{query}:{','.join(sorted(memory_types))}"
        return hashlib.md5(key_str.encode()).hexdigest()[:16]

    def _get_cached_memories(
        self, query: str, memory_types: List[str]
    ) -> List[MemoryItem]:
        """Retrieve memories from local cache."""
        cache_key = self._make_cache_key(query, memory_types)
        cached = self._cache.get(cache_key, {})

        if not cached:
            self.logger.debug("No cached memories found")
            return []

        memories = []
        for item in cached.get("memories", []):
            memories.append(
                MemoryItem(
                    id=item.get("id", str(uuid4())),
                    content=item.get("content", ""),
                    metadata=item.get("metadata", {"source": "cache"}),
                    score=item.get("score", 0.0),
                    type=MemoryItemType.TEXT,
                )
            )

        self.logger.debug(f"Retrieved {len(memories)} memories from cache")
        return memories

    def _memory_to_dict(self, memory: MemoryItem) -> Dict[str, Any]:
        """Convert MemoryItem to dictionary for caching."""
        return {
            "id": memory.id,
            "content": memory.content,
            "metadata": memory.metadata,
            "score": memory.score,
        }

    def _save_cache(self) -> None:
        """Save cache to disk."""
        try:
            with open(self._cache_file, "w") as f:
                json.dump(self._cache, f, indent=2, default=str)
        except Exception as e:
            self.logger.warning(f"Failed to save cache: {e}")


    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    def get_session_id(self) -> str:
        """Get the current session ID."""
        return self._session_id or ""

    def set_session_id(self, session_id: str) -> None:
        """Set session ID for continuity across devices."""
        self._session_id = session_id
        self.logger.info(f"Session ID set: {session_id}")

    def reset_session(self) -> str:
        """Reset to a new session."""
        self._session_id = str(uuid4())
        self._step_count = 0
        self.logger.info(f"New session: {self._session_id}")
        return self._session_id
