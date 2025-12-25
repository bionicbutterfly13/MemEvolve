"""
Graphiti Client for MemEvolve Integration.

Direct connection to Graphiti service (Neo4j knowledge graph).
Simpler than webhook-based approach - single source of truth.

Usage:
    client = GraphitiClient.from_environment()
    results = await client.search(query="machine learning")
    await client.ingest(entities=[...], edges=[...])
"""

import asyncio
import hashlib
import hmac
import json
import logging
import os
import secrets
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

import httpx

logger = logging.getLogger(__name__)


@dataclass
class GraphitiConfig:
    """Configuration for Graphiti client."""
    base_url: str = ""
    hmac_secret: str = ""
    project_id: str = "memevolve"
    enabled: bool = True
    timeout_seconds: float = 30.0

    @classmethod
    def from_environment(cls) -> "GraphitiConfig":
        return cls(
            base_url=os.getenv("GRAPHITI_BASE_URL", ""),
            hmac_secret=os.getenv("GRAPHITI_HMAC_SECRET", ""),
            project_id=os.getenv("GRAPHITI_PROJECT_ID", "memevolve"),
            enabled=os.getenv("GRAPHITI_ENABLED", "true").lower() == "true",
            timeout_seconds=float(os.getenv("GRAPHITI_TIMEOUT", "30.0")),
        )

    def validate(self) -> list[str]:
        errors = []
        if self.enabled and not self.base_url:
            errors.append("GRAPHITI_BASE_URL is required")
        return errors


@dataclass
class GraphitiResponse:
    """Response wrapper."""
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    status_code: int = 0
    latency_ms: float = 0.0


class GraphitiClient:
    """
    Client for Graphiti knowledge graph service.

    Endpoints:
    - /api/graphiti/search - Query entities and relationships
    - /api/graphiti/ingest - Add entities and relationships
    - /health - Service health check
    """

    def __init__(self, config: GraphitiConfig):
        self.config = config
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def from_environment(cls) -> "GraphitiClient":
        return cls(GraphitiConfig.from_environment())

    async def __aenter__(self) -> "GraphitiClient":
        await self._ensure_client()
        return self

    async def __aexit__(self, *args) -> None:
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.config.timeout_seconds),
                limits=httpx.Limits(max_connections=10),
            )
        return self._client

    async def close(self) -> None:
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _generate_signature(self, payload: bytes) -> str:
        """Generate HMAC-SHA256 signature."""
        if not self.config.hmac_secret:
            return ""
        digest = hmac.new(
            self.config.hmac_secret.encode(),
            payload,
            hashlib.sha256
        ).hexdigest()
        return f"sha256={digest}"

    def _build_headers(self, payload: bytes) -> dict[str, str]:
        """Build request headers."""
        headers = {"Content-Type": "application/json"}
        if self.config.hmac_secret:
            headers["X-Webhook-Signature"] = self._generate_signature(payload)
            headers["X-Signed-At"] = datetime.now(timezone.utc).isoformat()
            headers["X-Nonce"] = secrets.token_hex(16)
        headers["X-Project-Id"] = self.config.project_id
        return headers

    async def _request(
        self,
        method: str,
        endpoint: str,
        payload: Optional[dict] = None,
    ) -> GraphitiResponse:
        """Execute HTTP request."""
        import time
        start = time.time()

        url = f"{self.config.base_url.rstrip('/')}{endpoint}"
        payload_bytes = json.dumps(payload or {}).encode()
        headers = self._build_headers(payload_bytes)

        try:
            client = await self._ensure_client()
            response = await client.request(
                method=method,
                url=url,
                content=payload_bytes if payload else None,
                headers=headers,
            )

            latency_ms = (time.time() - start) * 1000

            if response.status_code >= 400:
                return GraphitiResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {response.text[:200]}",
                    status_code=response.status_code,
                    latency_ms=latency_ms,
                )

            try:
                data = response.json()
            except json.JSONDecodeError:
                data = {"raw": response.text}

            return GraphitiResponse(
                success=True,
                data=data,
                status_code=response.status_code,
                latency_ms=latency_ms,
            )

        except httpx.TimeoutException as e:
            return GraphitiResponse(success=False, error=f"Timeout: {e}")
        except httpx.RequestError as e:
            return GraphitiResponse(success=False, error=f"Request error: {e}")

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    async def health(self) -> GraphitiResponse:
        """Check Graphiti service health."""
        return await self._request("GET", "/health")

    async def search(
        self,
        query: str,
        limit: int = 10,
        entity_types: Optional[list[str]] = None,
    ) -> GraphitiResponse:
        """
        Search the knowledge graph.

        Args:
            query: Search query (semantic search)
            limit: Max results
            entity_types: Filter by entity types
        """
        payload = {
            "query": query,
            "limit": limit,
            "project_id": self.config.project_id,
        }
        if entity_types:
            payload["entity_types"] = entity_types

        return await self._request("POST", "/api/graphiti/search", payload)

    async def ingest(
        self,
        entities: list[dict[str, Any]],
        edges: Optional[list[dict[str, Any]]] = None,
        source: str = "memevolve",
    ) -> GraphitiResponse:
        """
        Ingest entities and relationships into the graph.

        Args:
            entities: List of entity dicts with {name, type, properties}
            edges: List of edge dicts with {source, target, type, properties}
            source: Source identifier
        """
        payload = {
            "entities": entities,
            "edges": edges or [],
            "source": source,
            "project_id": self.config.project_id,
        }

        return await self._request("POST", "/api/graphiti/ingest", payload)

    async def add_episode(
        self,
        content: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> GraphitiResponse:
        """
        Add an episode (trajectory) to the graph.

        Graphiti will extract entities automatically.
        """
        payload = {
            "content": content,
            "metadata": metadata or {},
            "project_id": self.config.project_id,
        }

        return await self._request("POST", "/api/graphiti/ingest", payload)


# Convenience function
async def graphiti_request(operation: str, **kwargs) -> GraphitiResponse:
    """Execute a one-off Graphiti request."""
    async with GraphitiClient.from_environment() as client:
        method = getattr(client, operation, None)
        if method is None:
            return GraphitiResponse(success=False, error=f"Unknown operation: {operation}")
        return await method(**kwargs)
