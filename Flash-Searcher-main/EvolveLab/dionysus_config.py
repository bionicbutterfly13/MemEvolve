"""
Configuration schema for Dionysus Memory Provider integration.

Handles connection to Dionysus3-core via n8n webhooks with HMAC authentication.
Supports feature flags for gradual rollout and canary deployments.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum


class DionysusEnvironment(Enum):
    """Deployment environment"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class DionysusTimeouts:
    """Timeout configuration per operation type"""
    connect_seconds: float = 2.0
    read_recall_seconds: float = 10.0
    read_ingest_seconds: float = 20.0
    read_evolve_seconds: float = 30.0


@dataclass
class DionysusRetryConfig:
    """Retry and circuit breaker configuration"""
    max_retries_recall: int = 2  # Idempotent, safe to retry
    max_retries_ingest: int = 0  # Non-idempotent, queue instead
    backoff_base_seconds: float = 0.5
    backoff_max_seconds: float = 8.0
    circuit_breaker_threshold: int = 5  # 5xx streak to trip
    circuit_breaker_recovery_seconds: float = 30.0


@dataclass
class DionysusFeatureFlags:
    """Feature flags for gradual rollout"""
    enabled: bool = True
    graph_writes_enabled: bool = True
    vector_writes_enabled: bool = True
    entity_extraction_enabled: bool = True
    session_tracking_enabled: bool = True
    canary_project_id: Optional[str] = None  # If set, only this project uses Dionysus


@dataclass
class DionysusConfig:
    """
    Main configuration for Dionysus Memory Provider.

    Environment variables (override dataclass defaults):
        DIONYSUS_WEBHOOK_BASE_URL: Base URL for n8n webhooks
        DIONYSUS_HMAC_SECRET: Shared secret for HMAC-SHA256 signing
        DIONYSUS_PROJECT_ID: Project identifier for multi-tenant isolation
        DIONYSUS_ENVIRONMENT: development/staging/production
        DIONYSUS_ENABLED: Master switch (true/false)
        DIONYSUS_GRAPH_WRITES_ENABLED: Enable Neo4j/Graphiti writes
        DIONYSUS_VECTOR_WRITES_ENABLED: Enable pgvector writes
        DIONYSUS_ENTITY_EXTRACTION: Enable entity extraction from trajectories
        DIONYSUS_SESSION_TRACKING: Enable session attribution
        DIONYSUS_CANARY_PROJECT: Limit to specific project ID for canary
    """
    # Connection settings
    webhook_base_url: str = ""
    hmac_secret: str = ""
    project_id: str = "memevolve-default"
    environment: DionysusEnvironment = DionysusEnvironment.DEVELOPMENT

    # Timeouts and retries
    timeouts: DionysusTimeouts = field(default_factory=DionysusTimeouts)
    retry_config: DionysusRetryConfig = field(default_factory=DionysusRetryConfig)

    # Feature flags
    feature_flags: DionysusFeatureFlags = field(default_factory=DionysusFeatureFlags)

    # Request settings
    request_id_prefix: str = "memevolve"
    nonce_bytes: int = 16
    timestamp_skew_seconds: int = 300  # 5 minute window

    # Fallback behavior
    fallback_to_local_cache: bool = True
    local_cache_dir: str = "./storage/dionysus/cache"
    local_cache_ttl_seconds: int = 3600  # 1 hour

    # Payload limits
    max_trajectory_size_bytes: int = 1_000_000  # 1MB
    max_entities_per_ingest: int = 100
    max_edges_per_ingest: int = 500

    @classmethod
    def from_environment(cls) -> "DionysusConfig":
        """
        Load configuration from environment variables.

        Returns:
            DionysusConfig with values from env vars, falling back to defaults.
        """
        # Parse environment enum
        env_str = os.getenv("DIONYSUS_ENVIRONMENT", "development").lower()
        try:
            environment = DionysusEnvironment(env_str)
        except ValueError:
            environment = DionysusEnvironment.DEVELOPMENT

        # Parse feature flags
        def parse_bool(key: str, default: bool) -> bool:
            val = os.getenv(key, "").lower()
            if val in ("true", "1", "yes"):
                return True
            elif val in ("false", "0", "no"):
                return False
            return default

        feature_flags = DionysusFeatureFlags(
            enabled=parse_bool("DIONYSUS_ENABLED", True),
            graph_writes_enabled=parse_bool("DIONYSUS_GRAPH_WRITES_ENABLED", True),
            vector_writes_enabled=parse_bool("DIONYSUS_VECTOR_WRITES_ENABLED", True),
            entity_extraction_enabled=parse_bool("DIONYSUS_ENTITY_EXTRACTION", True),
            session_tracking_enabled=parse_bool("DIONYSUS_SESSION_TRACKING", True),
            canary_project_id=os.getenv("DIONYSUS_CANARY_PROJECT"),
        )

        # Parse timeouts
        timeouts = DionysusTimeouts(
            connect_seconds=float(os.getenv("DIONYSUS_CONNECT_TIMEOUT", "2.0")),
            read_recall_seconds=float(os.getenv("DIONYSUS_RECALL_TIMEOUT", "10.0")),
            read_ingest_seconds=float(os.getenv("DIONYSUS_INGEST_TIMEOUT", "20.0")),
            read_evolve_seconds=float(os.getenv("DIONYSUS_EVOLVE_TIMEOUT", "30.0")),
        )

        # Parse retry config
        retry_config = DionysusRetryConfig(
            max_retries_recall=int(os.getenv("DIONYSUS_RETRY_COUNT", "2")),
            circuit_breaker_threshold=int(os.getenv("DIONYSUS_CIRCUIT_THRESHOLD", "5")),
        )

        return cls(
            webhook_base_url=os.getenv("DIONYSUS_WEBHOOK_BASE_URL", ""),
            hmac_secret=os.getenv("DIONYSUS_HMAC_SECRET", ""),
            project_id=os.getenv("DIONYSUS_PROJECT_ID", "memevolve-default"),
            environment=environment,
            timeouts=timeouts,
            retry_config=retry_config,
            feature_flags=feature_flags,
            local_cache_dir=os.getenv("DIONYSUS_CACHE_DIR", "./storage/dionysus/cache"),
        )

    def is_enabled_for_project(self, project_id: Optional[str] = None) -> bool:
        """
        Check if Dionysus is enabled for a specific project.

        Respects canary deployment settings.

        Args:
            project_id: Project to check. If None, uses configured project_id.

        Returns:
            True if Dionysus should be used for this project.
        """
        if not self.feature_flags.enabled:
            return False

        if self.feature_flags.canary_project_id:
            check_project = project_id or self.project_id
            return check_project == self.feature_flags.canary_project_id

        return True

    def validate(self) -> list[str]:
        """
        Validate configuration for required fields.

        Returns:
            List of validation error messages (empty if valid).
        """
        errors = []

        if self.feature_flags.enabled:
            if not self.webhook_base_url:
                errors.append("DIONYSUS_WEBHOOK_BASE_URL is required when Dionysus is enabled")
            if not self.hmac_secret:
                errors.append("DIONYSUS_HMAC_SECRET is required when Dionysus is enabled")
            if len(self.hmac_secret) < 32:
                errors.append("DIONYSUS_HMAC_SECRET should be at least 32 characters")

        return errors


# Singleton instance for easy access
_config_instance: Optional[DionysusConfig] = None


def get_dionysus_config() -> DionysusConfig:
    """
    Get the Dionysus configuration singleton.

    Loads from environment on first call, caches for subsequent calls.

    Returns:
        DionysusConfig instance.
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = DionysusConfig.from_environment()
    return _config_instance


def reset_dionysus_config() -> None:
    """Reset the configuration singleton (useful for testing)."""
    global _config_instance
    _config_instance = None
