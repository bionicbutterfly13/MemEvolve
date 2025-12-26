"""
EvolveLab Services

Service modules for external integrations:
- neo4j_service: Direct Neo4j database connectivity
- session_manager: Session tracking via Graphiti knowledge graph
"""

from .neo4j_service import Neo4jService, Neo4jConfig, Neo4jError
from .session_manager import (
    SessionManager,
    SessionInfo,
    SessionActivity,
    SessionManagerError,
    SessionNotFoundError,
    get_session_manager,
)

__all__ = [
    # Neo4j
    "Neo4jService",
    "Neo4jConfig",
    "Neo4jError",
    # Sessions
    "SessionManager",
    "SessionInfo",
    "SessionActivity",
    "SessionManagerError",
    "SessionNotFoundError",
    "get_session_manager",
]
