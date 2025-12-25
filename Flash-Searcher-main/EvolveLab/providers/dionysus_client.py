"""
HTTP Client for Dionysus3-core integration via n8n webhooks.

Features:
- HMAC-SHA256 request signing (compatible with Dionysus hmac_utils.py)
- Replay protection via timestamp + nonce headers
- Exponential backoff retries for idempotent operations
- Circuit breaker pattern for fault tolerance
- Per-operation timeout configuration

Usage:
    client = DionysusClient.from_config()
    response = await client.recall(query="search term", memory_types=["strategic"])
"""

import asyncio
import hashlib
import hmac
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import httpx

from ..dionysus_config import DionysusConfig, get_dionysus_config

logger = logging.getLogger(__name__)


class OperationType(Enum):
    """Types of operations with different retry/timeout policies."""
    RECALL = "recall"       # Idempotent - safe to retry
    INGEST = "ingest"       # Non-idempotent - queue on failure
    EVOLVE = "evolve"       # Idempotent - safe to retry
    HEALTH = "health"       # Idempotent - quick check


@dataclass
class CircuitBreakerState:
    """Tracks circuit breaker state for fault tolerance."""
    failures: int = 0
    last_failure_time: float = 0.0
    is_open: bool = False

    def record_failure(self, threshold: int, recovery_seconds: float) -> None:
        """Record a failure and potentially open the circuit."""
        self.failures += 1
        self.last_failure_time = time.time()
        if self.failures >= threshold:
            self.is_open = True
            logger.warning(
                f"Circuit breaker OPEN after {self.failures} failures. "
                f"Recovery in {recovery_seconds}s"
            )

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failures = 0
        self.is_open = False

    def should_allow_request(self, recovery_seconds: float) -> bool:
        """Check if a request should be allowed through."""
        if not self.is_open:
            return True
        # Check if recovery period has passed
        elapsed = time.time() - self.last_failure_time
        if elapsed >= recovery_seconds:
            logger.info("Circuit breaker attempting recovery...")
            return True
        return False


@dataclass
class DionysusResponse:
    """Response wrapper with metadata."""
    success: bool
    data: Optional[dict[str, Any]] = None
    error: Optional[str] = None
    status_code: int = 0
    request_id: str = ""
    latency_ms: float = 0.0


class DionysusClient:
    """
    HTTP client for Dionysus3-core with HMAC authentication.

    Provides methods for:
    - recall: Retrieve memories (vector + graph search)
    - ingest: Store trajectories and entities
    - evolve: Update retrieval strategies
    - health: Check connectivity
    """

    def __init__(self, config: DionysusConfig):
        """
        Initialize the client with configuration.

        Args:
            config: DionysusConfig instance with connection settings.
        """
        self.config = config
        self._circuit_breaker = CircuitBreakerState()
        self._client: Optional[httpx.AsyncClient] = None

    @classmethod
    def from_config(cls) -> "DionysusClient":
        """Create client from environment configuration."""
        return cls(get_dionysus_config())

    async def __aenter__(self) -> "DionysusClient":
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, *args) -> None:
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(
                    connect=self.config.timeouts.connect_seconds,
                    read=self.config.timeouts.read_recall_seconds,
                    write=10.0,
                    pool=5.0,
                ),
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    def _generate_signature(self, payload: bytes) -> str:
        """
        Generate HMAC-SHA256 signature compatible with Dionysus.

        Format: "sha256=<hex_digest>"
        """
        digest = hmac.new(
            key=self.config.hmac_secret.encode("utf-8"),
            msg=payload,
            digestmod=hashlib.sha256,
        ).hexdigest()
        return f"sha256={digest}"

    def _build_headers(
        self,
        payload: bytes,
        request_id: str,
    ) -> dict[str, str]:
        """
        Build authenticated request headers with replay protection.

        Headers:
        - X-Webhook-Signature: HMAC-SHA256 signature
        - X-Signed-At: ISO timestamp
        - X-Nonce: Random hex string
        - X-Project-Id: Project identifier
        - X-Request-Id: Unique request ID
        """
        now = datetime.now(timezone.utc)
        nonce = secrets.token_hex(self.config.nonce_bytes)

        return {
            "Content-Type": "application/json",
            "X-Webhook-Signature": self._generate_signature(payload),
            "X-Signed-At": now.isoformat(),
            "X-Nonce": nonce,
            "X-Project-Id": self.config.project_id,
            "X-Request-Id": request_id,
        }

    def _get_timeout_for_operation(self, operation: OperationType) -> float:
        """Get appropriate timeout for operation type."""
        timeouts = self.config.timeouts
        return {
            OperationType.RECALL: timeouts.read_recall_seconds,
            OperationType.INGEST: timeouts.read_ingest_seconds,
            OperationType.EVOLVE: timeouts.read_evolve_seconds,
            OperationType.HEALTH: 5.0,
        }.get(operation, timeouts.read_recall_seconds)

    def _get_max_retries(self, operation: OperationType) -> int:
        """Get max retries for operation type."""
        retry_cfg = self.config.retry_config
        if operation in (OperationType.RECALL, OperationType.EVOLVE, OperationType.HEALTH):
            return retry_cfg.max_retries_recall
        return retry_cfg.max_retries_ingest  # 0 for non-idempotent

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        payload: dict[str, Any],
        operation: OperationType,
    ) -> DionysusResponse:
        """
        Execute request with retry logic and circuit breaker.

        Args:
            method: HTTP method (GET, POST)
            endpoint: API endpoint path
            payload: Request payload
            operation: Operation type for timeout/retry policy

        Returns:
            DionysusResponse with result or error
        """
        request_id = f"{self.config.request_id_prefix}-{uuid4().hex[:12]}"
        retry_cfg = self.config.retry_config
        max_retries = self._get_max_retries(operation)
        timeout = self._get_timeout_for_operation(operation)

        # Check circuit breaker
        if not self._circuit_breaker.should_allow_request(
            retry_cfg.circuit_breaker_recovery_seconds
        ):
            logger.warning(f"Circuit breaker OPEN - rejecting {operation.value}")
            return DionysusResponse(
                success=False,
                error="Circuit breaker open - Dionysus unavailable",
                request_id=request_id,
            )

        url = f"{self.config.webhook_base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        payload_bytes = json.dumps(payload).encode("utf-8")
        headers = self._build_headers(payload_bytes, request_id)

        client = await self._ensure_client()
        last_error: Optional[Exception] = None
        start_time = time.time()

        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    # Exponential backoff
                    delay = min(
                        retry_cfg.backoff_base_seconds * (2 ** (attempt - 1)),
                        retry_cfg.backoff_max_seconds,
                    )
                    logger.info(
                        f"Retry {attempt}/{max_retries} for {operation.value} "
                        f"after {delay:.1f}s delay"
                    )
                    await asyncio.sleep(delay)

                response = await client.request(
                    method=method,
                    url=url,
                    content=payload_bytes,
                    headers=headers,
                    timeout=httpx.Timeout(
                        connect=self.config.timeouts.connect_seconds,
                        read=timeout,
                        write=10.0,
                        pool=5.0,
                    ),
                )

                latency_ms = (time.time() - start_time) * 1000

                if response.status_code >= 500:
                    # Server error - record failure, may retry
                    self._circuit_breaker.record_failure(
                        retry_cfg.circuit_breaker_threshold,
                        retry_cfg.circuit_breaker_recovery_seconds,
                    )
                    if attempt < max_retries:
                        last_error = Exception(f"Server error: {response.status_code}")
                        continue

                    return DionysusResponse(
                        success=False,
                        error=f"Server error: {response.status_code}",
                        status_code=response.status_code,
                        request_id=request_id,
                        latency_ms=latency_ms,
                    )

                # Success or client error (4xx) - don't retry client errors
                self._circuit_breaker.record_success()

                if response.status_code >= 400:
                    return DionysusResponse(
                        success=False,
                        error=f"Client error: {response.status_code} - {response.text}",
                        status_code=response.status_code,
                        request_id=request_id,
                        latency_ms=latency_ms,
                    )

                try:
                    data = response.json()
                except json.JSONDecodeError:
                    data = {"raw": response.text}

                logger.debug(
                    f"{operation.value} completed in {latency_ms:.1f}ms "
                    f"(attempt {attempt + 1})"
                )

                return DionysusResponse(
                    success=True,
                    data=data,
                    status_code=response.status_code,
                    request_id=request_id,
                    latency_ms=latency_ms,
                )

            except httpx.TimeoutException as e:
                last_error = e
                logger.warning(
                    f"Timeout on {operation.value} attempt {attempt + 1}: {e}"
                )
                self._circuit_breaker.record_failure(
                    retry_cfg.circuit_breaker_threshold,
                    retry_cfg.circuit_breaker_recovery_seconds,
                )

            except httpx.RequestError as e:
                last_error = e
                logger.warning(
                    f"Request error on {operation.value} attempt {attempt + 1}: {e}"
                )
                self._circuit_breaker.record_failure(
                    retry_cfg.circuit_breaker_threshold,
                    retry_cfg.circuit_breaker_recovery_seconds,
                )

        # All retries exhausted
        latency_ms = (time.time() - start_time) * 1000
        return DionysusResponse(
            success=False,
            error=f"All {max_retries + 1} attempts failed: {last_error}",
            request_id=request_id,
            latency_ms=latency_ms,
        )

    # -------------------------------------------------------------------------
    # Public API Methods
    # -------------------------------------------------------------------------

    async def health(self) -> DionysusResponse:
        """
        Check connectivity to Dionysus via n8n.

        Returns:
            DionysusResponse indicating health status.
        """
        return await self._request_with_retry(
            method="GET",
            endpoint="/health",
            payload={"ping": True},
            operation=OperationType.HEALTH,
        )

    async def recall(
        self,
        query: str,
        context: Optional[str] = None,
        memory_types: Optional[list[str]] = None,
        session_id: Optional[str] = None,
        top_k: int = 5,
    ) -> DionysusResponse:
        """
        Retrieve memories from Dionysus (vector + graph search).

        Args:
            query: Search query text
            context: Optional context for retrieval
            memory_types: Filter by memory types (strategic, operational, etc.)
            session_id: Session for context attribution
            top_k: Maximum results to return

        Returns:
            DionysusResponse with memory items.
        """
        payload = {
            "query": query,
            "context": context,
            "memory_types": memory_types or ["strategic", "operational", "semantic"],
            "session_id": session_id,
            "project_id": self.config.project_id,
            "top_k": top_k,
        }

        return await self._request_with_retry(
            method="POST",
            endpoint="/recall",
            payload=payload,
            operation=OperationType.RECALL,
        )

    async def ingest(
        self,
        trajectory: dict[str, Any],
        entities: Optional[list[dict[str, Any]]] = None,
        edges: Optional[list[dict[str, Any]]] = None,
        session_id: Optional[str] = None,
        memory_type: str = "episodic",
    ) -> DionysusResponse:
        """
        Store trajectory and entities in Dionysus.

        Args:
            trajectory: Full trajectory data
            entities: Extracted entities for Graphiti
            edges: Extracted relationships for Graphiti
            session_id: Session attribution
            memory_type: Type of memory to create

        Returns:
            DionysusResponse with ingestion status.
        """
        payload = {
            "trajectory": trajectory,
            "entities": entities or [],
            "edges": edges or [],
            "session_id": session_id,
            "project_id": self.config.project_id,
            "memory_type": memory_type,
        }

        return await self._request_with_retry(
            method="POST",
            endpoint="/ingest",
            payload=payload,
            operation=OperationType.INGEST,
        )

    async def evolve(
        self,
        strategy_updates: dict[str, Any],
        validation_results: Optional[dict[str, Any]] = None,
    ) -> DionysusResponse:
        """
        Send meta-evolution results to update Dionysus strategies.

        Args:
            strategy_updates: Winning strategies from AutoEvolver
            validation_results: Performance metrics for the strategies

        Returns:
            DionysusResponse with update confirmation.
        """
        payload = {
            "strategy_updates": strategy_updates,
            "validation_results": validation_results,
            "project_id": self.config.project_id,
        }

        return await self._request_with_retry(
            method="POST",
            endpoint="/evolve",
            payload=payload,
            operation=OperationType.EVOLVE,
        )


# Convenience function for one-off requests
async def dionysus_request(
    operation: str,
    **kwargs,
) -> DionysusResponse:
    """
    Execute a one-off Dionysus request.

    Args:
        operation: One of "health", "recall", "ingest", "evolve"
        **kwargs: Arguments for the operation

    Returns:
        DionysusResponse with result.
    """
    async with DionysusClient.from_config() as client:
        method = getattr(client, operation, None)
        if method is None:
            return DionysusResponse(
                success=False,
                error=f"Unknown operation: {operation}",
            )
        return await method(**kwargs)
