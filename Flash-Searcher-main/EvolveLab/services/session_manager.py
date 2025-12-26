"""
Session Manager for MemEvolve

Manages agent sessions via Graphiti knowledge graph.
Sessions are stored as episodes with automatic entity extraction.

Architecture:
- Uses GraphitiClient (HTTP) to communicate with Graphiti service
- Sessions stored as Episode nodes in Neo4j
- Entity relationships auto-extracted from session content
- Search via Graphiti's hybrid search (semantic + graph)

Usage:
    manager = SessionManager.from_environment()
    await manager.initialize()

    # Start a session
    session_id = await manager.create_session(device_id="device-123")

    # Add activity
    await manager.record_activity(session_id, "User searched for ML papers")

    # End session (triggers summary)
    summary = await manager.end_session(session_id)

    # Query history
    results = await manager.search_sessions("machine learning")
"""

import asyncio
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..providers.graphiti_client import GraphitiClient, GraphitiConfig, GraphitiResponse

logger = logging.getLogger(__name__)


@dataclass
class SessionInfo:
    """Session metadata."""
    session_id: str
    device_id: str
    started_at: datetime
    updated_at: datetime
    activity_count: int = 0
    summary: Optional[str] = None
    entities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SessionActivity:
    """A single activity within a session."""
    activity_id: str
    session_id: str
    content: str
    timestamp: datetime
    activity_type: str = "action"  # action, observation, thought, result
    metadata: Dict[str, Any] = field(default_factory=dict)


class SessionManagerError(Exception):
    """Base exception for session manager errors."""
    pass


class SessionNotFoundError(SessionManagerError):
    """Session not found."""
    pass


class SessionManager:
    """
    Manages agent sessions via Graphiti knowledge graph.

    Sessions are stored as Episodes in Graphiti with:
    - Automatic entity extraction from activities
    - Temporal tracking (valid_at timestamps)
    - Relationship inference between entities

    Local file tracking for active sessions, Graphiti for persistence.
    """

    DEFAULT_SESSION_DIR = ".memevolve/sessions"

    def __init__(self, config: Optional[GraphitiConfig] = None):
        """
        Initialize session manager.

        Args:
            config: Graphiti configuration. Uses environment if not provided.
        """
        self.config = config or GraphitiConfig.from_environment()
        self._client: Optional[GraphitiClient] = None
        self._active_sessions: Dict[str, SessionInfo] = {}

        # Local session file directory
        session_dir = os.getenv("MEMEVOLVE_SESSION_DIR", self.DEFAULT_SESSION_DIR)
        self._session_dir = Path.home() / session_dir

        # Logger
        self.logger = logging.getLogger(f"{__name__}.SessionManager")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [Session] [%(levelname)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    @classmethod
    def from_environment(cls) -> "SessionManager":
        """Create from environment variables."""
        return cls(GraphitiConfig.from_environment())

    def _get_client(self) -> GraphitiClient:
        """Get or create Graphiti client."""
        if self._client is None:
            self._client = GraphitiClient(self.config)
        return self._client

    def _run_async(self, coro):
        """Run async coroutine synchronously."""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as pool:
                    future = pool.submit(asyncio.run, coro)
                    return future.result()
            else:
                return loop.run_until_complete(coro)
        except RuntimeError:
            return asyncio.run(coro)

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    async def initialize(self) -> bool:
        """
        Initialize session manager.

        - Validates configuration
        - Tests Graphiti connectivity
        - Creates local session directory
        - Loads any active sessions from disk

        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing SessionManager...")

        # Validate config
        errors = self.config.validate()
        if errors:
            self.logger.warning(f"Configuration warnings: {errors}")

        # Create session directory
        self._session_dir.mkdir(parents=True, exist_ok=True)

        # Test Graphiti connection
        if self.config.enabled:
            try:
                response = await self._get_client().health()
                if response.success:
                    self.logger.info(
                        f"Graphiti connected ({response.latency_ms:.0f}ms)"
                    )
                else:
                    self.logger.warning(f"Graphiti health check failed: {response.error}")
            except Exception as e:
                self.logger.warning(f"Graphiti connection test failed: {e}")

        # Load active sessions from disk
        await self._load_active_sessions()

        self.logger.info(f"SessionManager ready ({len(self._active_sessions)} active sessions)")
        return True

    async def _load_active_sessions(self) -> None:
        """Load active sessions from local files."""
        if not self._session_dir.exists():
            return

        for session_file in self._session_dir.glob("*.json"):
            try:
                data = json.loads(session_file.read_text())
                session_id = data.get("session_id")
                if session_id:
                    self._active_sessions[session_id] = SessionInfo(
                        session_id=session_id,
                        device_id=data.get("device_id", "unknown"),
                        started_at=datetime.fromisoformat(data["started_at"]),
                        updated_at=datetime.fromisoformat(data["updated_at"]),
                        activity_count=data.get("activity_count", 0),
                        metadata=data.get("metadata", {}),
                    )
                    self.logger.debug(f"Loaded session: {session_id}")
            except Exception as e:
                self.logger.warning(f"Failed to load session {session_file}: {e}")

    def _save_session_to_disk(self, session: SessionInfo) -> None:
        """Save session to local file."""
        session_file = self._session_dir / f"{session.session_id}.json"
        data = {
            "session_id": session.session_id,
            "device_id": session.device_id,
            "started_at": session.started_at.isoformat(),
            "updated_at": session.updated_at.isoformat(),
            "activity_count": session.activity_count,
            "metadata": session.metadata,
        }
        session_file.write_text(json.dumps(data, indent=2))

    def _remove_session_from_disk(self, session_id: str) -> None:
        """Remove session file from disk."""
        session_file = self._session_dir / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

    # -------------------------------------------------------------------------
    # Session Lifecycle
    # -------------------------------------------------------------------------

    async def create_session(
        self,
        device_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new session.

        Args:
            device_id: Device/user identifier
            metadata: Optional session metadata

        Returns:
            Session ID (UUID string)
        """
        session_id = str(uuid4())
        now = datetime.now(timezone.utc)

        session = SessionInfo(
            session_id=session_id,
            device_id=device_id,
            started_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        self._active_sessions[session_id] = session
        self._save_session_to_disk(session)

        self.logger.info(f"Created session: {session_id} (device: {device_id})")

        # Record session start in Graphiti
        if self.config.enabled:
            await self._ingest_session_event(
                session_id=session_id,
                event_type="session_start",
                content=f"Session started for device {device_id}",
                metadata={"device_id": device_id, **(metadata or {})},
            )

        return session_id

    async def get_or_create_session(
        self,
        device_id: str,
        timeout_minutes: int = 30,
    ) -> tuple[str, bool]:
        """
        Get existing session for device or create new one.

        Args:
            device_id: Device identifier
            timeout_minutes: Session timeout in minutes

        Returns:
            (session_id, is_new) tuple
        """
        # Check for existing active session for this device
        now = datetime.now(timezone.utc)

        for session_id, session in list(self._active_sessions.items()):
            if session.device_id == device_id:
                # Check if expired
                elapsed = (now - session.updated_at).total_seconds() / 60
                if elapsed < timeout_minutes:
                    self.logger.debug(f"Resuming session: {session_id}")
                    return session_id, False
                else:
                    # Session expired, end it
                    await self.end_session(session_id)

        # Create new session
        session_id = await self.create_session(device_id)
        return session_id, True

    async def record_activity(
        self,
        session_id: str,
        content: str,
        activity_type: str = "action",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionActivity:
        """
        Record an activity within a session.

        Activities are ingested to Graphiti for entity extraction.

        Args:
            session_id: Session to record activity in
            content: Activity content (natural language)
            activity_type: Type of activity (action, observation, thought, result)
            metadata: Optional activity metadata

        Returns:
            SessionActivity record

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        if session_id not in self._active_sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        session = self._active_sessions[session_id]
        now = datetime.now(timezone.utc)

        activity = SessionActivity(
            activity_id=str(uuid4()),
            session_id=session_id,
            content=content,
            timestamp=now,
            activity_type=activity_type,
            metadata=metadata or {},
        )

        # Update session
        session.activity_count += 1
        session.updated_at = now
        self._save_session_to_disk(session)

        self.logger.debug(
            f"Activity recorded: {activity_type} in {session_id[:8]} "
            f"({session.activity_count} total)"
        )

        # Ingest to Graphiti
        if self.config.enabled:
            await self._ingest_session_event(
                session_id=session_id,
                event_type=activity_type,
                content=content,
                metadata={
                    "activity_id": activity.activity_id,
                    "activity_type": activity_type,
                    **(metadata or {}),
                },
            )

        return activity

    async def end_session(
        self,
        session_id: str,
        generate_summary: bool = True,
    ) -> Optional[str]:
        """
        End a session and optionally generate summary.

        Args:
            session_id: Session to end
            generate_summary: Whether to generate session summary

        Returns:
            Session summary if generated, None otherwise

        Raises:
            SessionNotFoundError: If session doesn't exist
        """
        if session_id not in self._active_sessions:
            raise SessionNotFoundError(f"Session not found: {session_id}")

        session = self._active_sessions[session_id]
        now = datetime.now(timezone.utc)

        duration_minutes = (now - session.started_at).total_seconds() / 60

        self.logger.info(
            f"Ending session: {session_id[:8]} "
            f"(duration: {duration_minutes:.1f}min, activities: {session.activity_count})"
        )

        summary = None
        if generate_summary and self.config.enabled:
            summary = await self._generate_session_summary(session)
            session.summary = summary

        # Record session end in Graphiti
        if self.config.enabled:
            await self._ingest_session_event(
                session_id=session_id,
                event_type="session_end",
                content=f"Session ended. Duration: {duration_minutes:.1f} minutes. "
                        f"Activities: {session.activity_count}. "
                        f"Summary: {summary or 'No summary generated'}",
                metadata={
                    "duration_minutes": duration_minutes,
                    "activity_count": session.activity_count,
                    "summary": summary,
                },
            )

        # Cleanup
        del self._active_sessions[session_id]
        self._remove_session_from_disk(session_id)

        return summary

    # -------------------------------------------------------------------------
    # Query Operations
    # -------------------------------------------------------------------------

    async def search_sessions(
        self,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Search session history.

        Uses Graphiti's hybrid search across session episodes.

        Args:
            query: Search query
            limit: Maximum results

        Returns:
            List of matching session records
        """
        if not self.config.enabled:
            self.logger.warning("Graphiti disabled, cannot search")
            return []

        try:
            response = await self._get_client().search(
                query=query,
                limit=limit,
                entity_types=["Episode", "session_start", "session_end"],
            )

            if response.success and response.data:
                results = response.data.get("results", response.data.get("edges", []))
                self.logger.debug(
                    f"Search '{query}' returned {len(results)} results "
                    f"({response.latency_ms:.0f}ms)"
                )
                return results
            else:
                self.logger.warning(f"Search failed: {response.error}")
                return []

        except Exception as e:
            self.logger.error(f"Search error: {e}")
            return []

    async def get_session(self, session_id: str) -> Optional[SessionInfo]:
        """Get session by ID (active sessions only)."""
        return self._active_sessions.get(session_id)

    def get_active_sessions(self) -> List[SessionInfo]:
        """Get all active sessions."""
        return list(self._active_sessions.values())

    # -------------------------------------------------------------------------
    # Internal Methods
    # -------------------------------------------------------------------------

    async def _ingest_session_event(
        self,
        session_id: str,
        event_type: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[GraphitiResponse]:
        """Ingest session event to Graphiti."""
        try:
            # Format as natural language for entity extraction
            episode_content = f"""
Session Event: {event_type}
Session ID: {session_id}
Timestamp: {datetime.now(timezone.utc).isoformat()}

{content}
"""

            response = await self._get_client().add_episode(
                content=episode_content,
                metadata={
                    "session_id": session_id,
                    "event_type": event_type,
                    "source": "memevolve_session",
                    **(metadata or {}),
                },
            )

            if response.success:
                data = response.data or {}
                nodes = data.get("nodes", [])
                edges = data.get("edges", [])

                if nodes:
                    # Update session entities
                    session = self._active_sessions.get(session_id)
                    if session:
                        new_entities = [n.get("name", "?") for n in nodes[:5]]
                        session.entities.extend(new_entities)
                        self.logger.debug(f"Extracted entities: {new_entities}")

                self.logger.debug(
                    f"Ingested {event_type}: {len(nodes)} entities, "
                    f"{len(edges)} relationships ({response.latency_ms:.0f}ms)"
                )
            else:
                self.logger.warning(f"Ingest failed: {response.error}")

            return response

        except Exception as e:
            self.logger.error(f"Ingest error: {e}")
            return None

    async def _generate_session_summary(self, session: SessionInfo) -> str:
        """
        Generate session summary.

        For now, returns a simple summary. Could use LLM later.
        """
        duration_minutes = (
            datetime.now(timezone.utc) - session.started_at
        ).total_seconds() / 60

        entity_str = ", ".join(session.entities[:10]) if session.entities else "none"

        summary = (
            f"Session for device {session.device_id}. "
            f"Duration: {duration_minutes:.1f} minutes. "
            f"Activities: {session.activity_count}. "
            f"Key entities: {entity_str}."
        )

        return summary

    # -------------------------------------------------------------------------
    # Context Manager
    # -------------------------------------------------------------------------

    async def __aenter__(self) -> "SessionManager":
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.close()


# Convenience functions
def get_session_manager() -> SessionManager:
    """Get a SessionManager instance configured from environment."""
    return SessionManager.from_environment()


async def create_session(device_id: str, **kwargs) -> str:
    """Convenience function to create a session."""
    manager = SessionManager.from_environment()
    await manager.initialize()
    return await manager.create_session(device_id, **kwargs)
