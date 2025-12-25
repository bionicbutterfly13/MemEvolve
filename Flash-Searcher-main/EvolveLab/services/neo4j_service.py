"""
Neo4j Service for EvolveLab.

Direct connection to Neo4j database using the official Python driver.
Provides an alternative to HTTP-based Graphiti calls for DionysusMemoryProvider.

Features:
- Sync and async query execution
- Connection pooling (built into neo4j driver)
- Custom exception handling
- Context manager support
- Node and relationship CRUD operations

Usage:
    # Sync usage with context manager
    with Neo4jService.from_environment() as neo4j:
        nodes = neo4j.search_nodes("Entity", "machine learning", limit=10)

    # Async usage
    async with Neo4jService.from_environment() as neo4j:
        node_id = await neo4j.create_node_async(["Entity"], {"name": "test"})

Environment Variables:
    NEO4J_URI: Bolt connection URI (e.g., bolt://localhost:7687)
    NEO4J_USER: Database username
    NEO4J_PASSWORD: Database password
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from uuid import uuid4

# Neo4j driver imports
try:
    from neo4j import GraphDatabase, AsyncGraphDatabase
    from neo4j.exceptions import (
        Neo4jError as DriverNeo4jError,
        ServiceUnavailable,
        AuthError,
        TransientError,
    )
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None
    AsyncGraphDatabase = None
    DriverNeo4jError = Exception
    ServiceUnavailable = Exception
    AuthError = Exception
    TransientError = Exception

logger = logging.getLogger(__name__)


# =============================================================================
# Custom Exceptions
# =============================================================================

class Neo4jError(Exception):
    """Base exception for Neo4j service errors."""

    def __init__(self, message: str, cause: Optional[Exception] = None):
        super().__init__(message)
        self.message = message
        self.cause = cause

    def __str__(self) -> str:
        if self.cause:
            return f"{self.message}: {self.cause}"
        return self.message


class Neo4jConnectionError(Neo4jError):
    """Raised when connection to Neo4j fails."""
    pass


class Neo4jAuthenticationError(Neo4jError):
    """Raised when authentication fails."""
    pass


class Neo4jQueryError(Neo4jError):
    """Raised when a query fails."""

    def __init__(
        self,
        message: str,
        query: Optional[str] = None,
        params: Optional[Dict] = None,
        cause: Optional[Exception] = None,
    ):
        super().__init__(message, cause)
        self.query = query
        self.params = params


class Neo4jNotAvailableError(Neo4jError):
    """Raised when neo4j Python package is not installed."""
    pass


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Neo4jConfig:
    """Configuration for Neo4j connection."""

    uri: str = ""
    user: str = ""
    password: str = ""
    database: str = "neo4j"
    max_connection_lifetime: int = 3600  # seconds
    max_connection_pool_size: int = 50
    connection_acquisition_timeout: int = 60  # seconds
    encrypted: bool = False
    trust: str = "TRUST_ALL_CERTIFICATES"

    @classmethod
    def from_environment(cls) -> "Neo4jConfig":
        """
        Load configuration from environment variables.

        Environment Variables:
            NEO4J_URI: Bolt connection URI
            NEO4J_USER: Database username
            NEO4J_PASSWORD: Database password
            NEO4J_DATABASE: Database name (default: neo4j)
            NEO4J_POOL_SIZE: Max connection pool size (default: 50)

        Returns:
            Neo4jConfig with values from environment.
        """
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            user=os.getenv("NEO4J_USER", "neo4j"),
            password=os.getenv("NEO4J_PASSWORD", ""),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            max_connection_pool_size=int(os.getenv("NEO4J_POOL_SIZE", "50")),
        )

    def validate(self) -> List[str]:
        """
        Validate configuration.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []
        if not self.uri:
            errors.append("NEO4J_URI is required")
        if not self.user:
            errors.append("NEO4J_USER is required")
        if not self.password:
            errors.append("NEO4J_PASSWORD is required")
        return errors


# =============================================================================
# Neo4j Service
# =============================================================================

class Neo4jService:
    """
    Service class for Neo4j database operations.

    Provides both sync and async methods for:
    - Running raw Cypher queries
    - Creating nodes and relationships
    - Searching nodes by label and query
    - CRUD operations on nodes

    Supports context manager protocol for automatic connection cleanup.
    """

    def __init__(
        self,
        uri: Optional[str] = None,
        user: Optional[str] = None,
        password: Optional[str] = None,
        config: Optional[Neo4jConfig] = None,
    ):
        """
        Initialize Neo4j service.

        Args:
            uri: Neo4j Bolt URI (e.g., bolt://localhost:7687)
            user: Database username
            password: Database password
            config: Neo4jConfig object (overrides uri/user/password if provided)
        """
        if not NEO4J_AVAILABLE:
            raise Neo4jNotAvailableError(
                "neo4j Python package is not installed. "
                "Install with: pip install neo4j"
            )

        # Use config if provided, otherwise build from parameters
        if config:
            self._config = config
        else:
            self._config = Neo4jConfig(
                uri=uri or os.getenv("NEO4J_URI", "bolt://localhost:7687"),
                user=user or os.getenv("NEO4J_USER", "neo4j"),
                password=password or os.getenv("NEO4J_PASSWORD", ""),
            )

        # Validate configuration
        errors = self._config.validate()
        if errors:
            raise Neo4jConnectionError(f"Invalid configuration: {', '.join(errors)}")

        # Drivers (initialized lazily)
        self._driver = None
        self._async_driver = None

        # Connection state
        self._connected = False

        # Logger
        self._logger = logging.getLogger(f"{__name__}.Neo4jService")

    @classmethod
    def from_environment(cls) -> "Neo4jService":
        """
        Create Neo4jService from environment variables.

        Returns:
            Neo4jService configured from environment.
        """
        config = Neo4jConfig.from_environment()
        return cls(config=config)

    # -------------------------------------------------------------------------
    # Connection Management
    # -------------------------------------------------------------------------

    def connect(self) -> bool:
        """
        Establish connection to Neo4j database.

        Creates the driver and verifies connectivity.

        Returns:
            True if connection successful.

        Raises:
            Neo4jConnectionError: If connection fails.
            Neo4jAuthenticationError: If authentication fails.
        """
        if self._connected and self._driver:
            return True

        try:
            self._driver = GraphDatabase.driver(
                self._config.uri,
                auth=(self._config.user, self._config.password),
                max_connection_lifetime=self._config.max_connection_lifetime,
                max_connection_pool_size=self._config.max_connection_pool_size,
                connection_acquisition_timeout=self._config.connection_acquisition_timeout,
            )

            # Verify connectivity
            self._driver.verify_connectivity()
            self._connected = True
            self._logger.info(f"Connected to Neo4j at {self._config.uri}")
            return True

        except AuthError as e:
            raise Neo4jAuthenticationError(
                f"Authentication failed for user '{self._config.user}'",
                cause=e,
            )
        except ServiceUnavailable as e:
            raise Neo4jConnectionError(
                f"Neo4j service unavailable at {self._config.uri}",
                cause=e,
            )
        except Exception as e:
            raise Neo4jConnectionError(
                f"Failed to connect to Neo4j",
                cause=e,
            )

    async def connect_async(self) -> bool:
        """
        Establish async connection to Neo4j database.

        Returns:
            True if connection successful.

        Raises:
            Neo4jConnectionError: If connection fails.
            Neo4jAuthenticationError: If authentication fails.
        """
        if self._async_driver:
            return True

        try:
            self._async_driver = AsyncGraphDatabase.driver(
                self._config.uri,
                auth=(self._config.user, self._config.password),
                max_connection_lifetime=self._config.max_connection_lifetime,
                max_connection_pool_size=self._config.max_connection_pool_size,
                connection_acquisition_timeout=self._config.connection_acquisition_timeout,
            )

            # Verify connectivity
            await self._async_driver.verify_connectivity()
            self._logger.info(f"Async connected to Neo4j at {self._config.uri}")
            return True

        except AuthError as e:
            raise Neo4jAuthenticationError(
                f"Authentication failed for user '{self._config.user}'",
                cause=e,
            )
        except ServiceUnavailable as e:
            raise Neo4jConnectionError(
                f"Neo4j service unavailable at {self._config.uri}",
                cause=e,
            )
        except Exception as e:
            raise Neo4jConnectionError(
                f"Failed to connect to Neo4j",
                cause=e,
            )

    def close(self) -> None:
        """Close the Neo4j connection and release resources."""
        if self._driver:
            try:
                self._driver.close()
                self._logger.info("Neo4j connection closed")
            except Exception as e:
                self._logger.warning(f"Error closing Neo4j connection: {e}")
            finally:
                self._driver = None
                self._connected = False

    async def close_async(self) -> None:
        """Close the async Neo4j connection."""
        if self._async_driver:
            try:
                await self._async_driver.close()
                self._logger.info("Async Neo4j connection closed")
            except Exception as e:
                self._logger.warning(f"Error closing async Neo4j connection: {e}")
            finally:
                self._async_driver = None

    # -------------------------------------------------------------------------
    # Context Manager Support
    # -------------------------------------------------------------------------

    def __enter__(self) -> "Neo4jService":
        """Enter sync context manager."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit sync context manager."""
        self.close()

    async def __aenter__(self) -> "Neo4jService":
        """Enter async context manager."""
        await self.connect_async()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit async context manager."""
        await self.close_async()

    # -------------------------------------------------------------------------
    # Query Execution
    # -------------------------------------------------------------------------

    def run_query(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query and return results.

        Args:
            query: Cypher query string.
            params: Query parameters.

        Returns:
            List of result records as dictionaries.

        Raises:
            Neo4jQueryError: If query execution fails.
        """
        if not self._connected:
            self.connect()

        params = params or {}

        try:
            with self._driver.session(database=self._config.database) as session:
                result = session.run(query, params)
                records = [dict(record) for record in result]
                self._logger.debug(f"Query returned {len(records)} records")
                return records

        except DriverNeo4jError as e:
            raise Neo4jQueryError(
                f"Query execution failed",
                query=query,
                params=params,
                cause=e,
            )

    async def run_query_async(
        self,
        query: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query asynchronously.

        Args:
            query: Cypher query string.
            params: Query parameters.

        Returns:
            List of result records as dictionaries.

        Raises:
            Neo4jQueryError: If query execution fails.
        """
        if not self._async_driver:
            await self.connect_async()

        params = params or {}

        try:
            async with self._async_driver.session(database=self._config.database) as session:
                result = await session.run(query, params)
                records = [dict(record) async for record in result]
                self._logger.debug(f"Async query returned {len(records)} records")
                return records

        except DriverNeo4jError as e:
            raise Neo4jQueryError(
                f"Async query execution failed",
                query=query,
                params=params,
                cause=e,
            )

    # -------------------------------------------------------------------------
    # Node Operations
    # -------------------------------------------------------------------------

    def create_node(
        self,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> str:
        """
        Create a node with the given labels and properties.

        Args:
            labels: List of labels for the node.
            properties: Node properties.

        Returns:
            The node's element ID (Neo4j 5.x) or string ID.

        Raises:
            Neo4jQueryError: If node creation fails.
        """
        # Generate a UUID if not provided
        if "id" not in properties:
            properties["id"] = str(uuid4())

        labels_str = ":".join(labels) if labels else ""
        query = f"""
        CREATE (n:{labels_str} $props)
        RETURN elementId(n) AS node_id, n.id AS id
        """

        results = self.run_query(query, {"props": properties})
        if results:
            return results[0].get("id", results[0].get("node_id", ""))
        return ""

    async def create_node_async(
        self,
        labels: List[str],
        properties: Dict[str, Any],
    ) -> str:
        """
        Create a node asynchronously.

        Args:
            labels: List of labels for the node.
            properties: Node properties.

        Returns:
            The node's ID.

        Raises:
            Neo4jQueryError: If node creation fails.
        """
        if "id" not in properties:
            properties["id"] = str(uuid4())

        labels_str = ":".join(labels) if labels else ""
        query = f"""
        CREATE (n:{labels_str} $props)
        RETURN elementId(n) AS node_id, n.id AS id
        """

        results = await self.run_query_async(query, {"props": properties})
        if results:
            return results[0].get("id", results[0].get("node_id", ""))
        return ""

    def get_node_by_id(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID property.

        Args:
            node_id: The node's id property value.

        Returns:
            Node properties as dict, or None if not found.

        Raises:
            Neo4jQueryError: If query fails.
        """
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) AS labels, elementId(n) AS element_id
        """

        results = self.run_query(query, {"node_id": node_id})
        if results:
            record = results[0]
            node_data = dict(record.get("n", {}))
            node_data["_labels"] = record.get("labels", [])
            node_data["_element_id"] = record.get("element_id", "")
            return node_data
        return None

    async def get_node_by_id_async(self, node_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a node by its ID property asynchronously.

        Args:
            node_id: The node's id property value.

        Returns:
            Node properties as dict, or None if not found.
        """
        query = """
        MATCH (n {id: $node_id})
        RETURN n, labels(n) AS labels, elementId(n) AS element_id
        """

        results = await self.run_query_async(query, {"node_id": node_id})
        if results:
            record = results[0]
            node_data = dict(record.get("n", {}))
            node_data["_labels"] = record.get("labels", [])
            node_data["_element_id"] = record.get("element_id", "")
            return node_data
        return None

    def delete_node(self, node_id: str) -> bool:
        """
        Delete a node by its ID property.

        Also deletes any relationships connected to the node.

        Args:
            node_id: The node's id property value.

        Returns:
            True if node was deleted, False if not found.

        Raises:
            Neo4jQueryError: If deletion fails.
        """
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """

        results = self.run_query(query, {"node_id": node_id})
        if results:
            return results[0].get("deleted_count", 0) > 0
        return False

    async def delete_node_async(self, node_id: str) -> bool:
        """
        Delete a node asynchronously.

        Args:
            node_id: The node's id property value.

        Returns:
            True if node was deleted, False if not found.
        """
        query = """
        MATCH (n {id: $node_id})
        DETACH DELETE n
        RETURN count(n) AS deleted_count
        """

        results = await self.run_query_async(query, {"node_id": node_id})
        if results:
            return results[0].get("deleted_count", 0) > 0
        return False

    # -------------------------------------------------------------------------
    # Relationship Operations
    # -------------------------------------------------------------------------

    def create_relationship(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a relationship between two nodes.

        Args:
            from_id: ID of the source node.
            to_id: ID of the target node.
            rel_type: Relationship type (e.g., "KNOWS", "RELATES_TO").
            properties: Relationship properties.

        Returns:
            Relationship ID or element ID.

        Raises:
            Neo4jQueryError: If relationship creation fails.
        """
        properties = properties or {}
        if "id" not in properties:
            properties["id"] = str(uuid4())

        query = f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        CREATE (a)-[r:{rel_type} $props]->(b)
        RETURN elementId(r) AS rel_id, r.id AS id
        """

        results = self.run_query(query, {
            "from_id": from_id,
            "to_id": to_id,
            "props": properties,
        })

        if results:
            return results[0].get("id", results[0].get("rel_id", ""))
        return ""

    async def create_relationship_async(
        self,
        from_id: str,
        to_id: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a relationship asynchronously.

        Args:
            from_id: ID of the source node.
            to_id: ID of the target node.
            rel_type: Relationship type.
            properties: Relationship properties.

        Returns:
            Relationship ID.
        """
        properties = properties or {}
        if "id" not in properties:
            properties["id"] = str(uuid4())

        query = f"""
        MATCH (a {{id: $from_id}}), (b {{id: $to_id}})
        CREATE (a)-[r:{rel_type} $props]->(b)
        RETURN elementId(r) AS rel_id, r.id AS id
        """

        results = await self.run_query_async(query, {
            "from_id": from_id,
            "to_id": to_id,
            "props": properties,
        })

        if results:
            return results[0].get("id", results[0].get("rel_id", ""))
        return ""

    # -------------------------------------------------------------------------
    # Search Operations
    # -------------------------------------------------------------------------

    def search_nodes(
        self,
        label: str,
        query: str,
        limit: int = 10,
        properties: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search nodes by label with text matching on properties.

        Searches across name, content, and description properties by default.
        Uses case-insensitive CONTAINS matching.

        Args:
            label: Node label to search (e.g., "Entity", "Episode").
            query: Search query string.
            limit: Maximum number of results.
            properties: List of property names to search in.

        Returns:
            List of matching nodes as dictionaries.

        Raises:
            Neo4jQueryError: If search fails.
        """
        search_props = properties or ["name", "content", "description"]

        # Build WHERE clause for text search
        conditions = " OR ".join([
            f"toLower(n.{prop}) CONTAINS toLower($query)"
            for prop in search_props
        ])

        cypher = f"""
        MATCH (n:{label})
        WHERE {conditions}
        RETURN n, labels(n) AS labels, elementId(n) AS element_id
        LIMIT $limit
        """

        results = self.run_query(cypher, {"query": query, "limit": limit})

        nodes = []
        for record in results:
            node_data = dict(record.get("n", {}))
            node_data["_labels"] = record.get("labels", [])
            node_data["_element_id"] = record.get("element_id", "")
            nodes.append(node_data)

        return nodes

    async def search_nodes_async(
        self,
        label: str,
        query: str,
        limit: int = 10,
        properties: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search nodes asynchronously.

        Args:
            label: Node label to search.
            query: Search query string.
            limit: Maximum number of results.
            properties: List of property names to search in.

        Returns:
            List of matching nodes as dictionaries.
        """
        search_props = properties or ["name", "content", "description"]

        conditions = " OR ".join([
            f"toLower(n.{prop}) CONTAINS toLower($query)"
            for prop in search_props
        ])

        cypher = f"""
        MATCH (n:{label})
        WHERE {conditions}
        RETURN n, labels(n) AS labels, elementId(n) AS element_id
        LIMIT $limit
        """

        results = await self.run_query_async(cypher, {"query": query, "limit": limit})

        nodes = []
        for record in results:
            node_data = dict(record.get("n", {}))
            node_data["_labels"] = record.get("labels", [])
            node_data["_element_id"] = record.get("element_id", "")
            nodes.append(node_data)

        return nodes

    # -------------------------------------------------------------------------
    # Utility Methods
    # -------------------------------------------------------------------------

    def is_connected(self) -> bool:
        """Check if currently connected to Neo4j."""
        return self._connected and self._driver is not None

    def get_driver_info(self) -> Dict[str, Any]:
        """
        Get information about the current driver configuration.

        Returns:
            Dict with driver configuration details.
        """
        return {
            "uri": self._config.uri,
            "user": self._config.user,
            "database": self._config.database,
            "connected": self._connected,
            "pool_size": self._config.max_connection_pool_size,
        }
