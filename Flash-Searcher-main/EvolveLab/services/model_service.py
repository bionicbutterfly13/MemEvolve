"""
Model Service for MemEvolve Mental Models

Manages mental models via Graphiti knowledge graph.
Based on Yufik's neuronal packet theory - models predict and learn from errors.

Architecture:
- Mental models stored as Entity nodes in Neo4j via Graphiti
- Predictions stored as Episode events
- Model accuracy tracked through prediction error resolution
- PREDICTS relationships connect models to their predictions

Key Concepts:
- MentalModel: A predictive pattern learned from agent experiences
- Prediction: A model's expectation given context
- Resolution: Comparing prediction to actual outcome (error tracking)
- Revision: Updating model when error rate exceeds threshold

Usage:
    service = ModelService.from_environment()
    await service.initialize()

    # Create a model from learned patterns
    model_id = await service.create_model(
        name="WebSearchPattern",
        domain="task",
        description="Patterns for effective web searches"
    )

    # Generate prediction
    prediction = await service.generate_prediction(
        model_id=model_id,
        context={"query": "find population data"}
    )

    # Resolve with actual outcome
    await service.resolve_prediction(
        prediction_id=prediction["id"],
        observation={"result": "success", "method": "official_stats"}
    )
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..providers.graphiti_client import GraphitiClient, GraphitiConfig, GraphitiResponse

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration
# =============================================================================

REVISION_ERROR_THRESHOLD = 0.5  # Flag for revision when error > 50%
PREDICTION_TTL_HOURS = 24
MAX_MODELS_PER_CONTEXT = 5


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ModelDomain:
    """Mental model domains."""
    USER = "user"          # User behavior patterns
    SELF = "self"          # Agent self-model
    WORLD = "world"        # Environmental patterns
    TASK = "task"          # Task-specific patterns


@dataclass
class ModelStatus:
    """Model lifecycle status."""
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


@dataclass
class MentalModel:
    """A mental model for prediction."""
    id: str
    name: str
    domain: str
    description: Optional[str] = None
    patterns: List[str] = field(default_factory=list)
    prediction_templates: List[Dict[str, Any]] = field(default_factory=list)
    accuracy: float = 0.5
    prediction_count: int = 0
    error_sum: float = 0.0
    status: str = "active"
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Prediction:
    """A prediction from a mental model."""
    id: str
    model_id: str
    prediction: Dict[str, Any]
    context: Dict[str, Any]
    confidence: float
    observation: Optional[Dict[str, Any]] = None
    error: Optional[float] = None
    resolved_at: Optional[datetime] = None
    created_at: Optional[datetime] = None


class ModelServiceError(Exception):
    """Base exception for model service errors."""
    pass


class ModelNotFoundError(ModelServiceError):
    """Model not found."""
    pass


class PredictionNotFoundError(ModelServiceError):
    """Prediction not found."""
    pass


# =============================================================================
# Model Service
# =============================================================================

class ModelService:
    """
    Service for managing mental models via Graphiti.

    Mental models are stored as Entity nodes with:
    - Automatic relationship extraction from patterns
    - PREDICTS edges to prediction Episodes
    - Accuracy tracking through error resolution
    """

    def __init__(self, config: Optional[GraphitiConfig] = None):
        """
        Initialize model service.

        Args:
            config: Graphiti configuration. Uses environment if not provided.
        """
        self.config = config or GraphitiConfig.from_environment()
        self._client: Optional[GraphitiClient] = None

        # In-memory model cache (Graphiti is source of truth)
        self._models: Dict[str, MentalModel] = {}
        self._predictions: Dict[str, Prediction] = {}

        # Logger
        self.logger = logging.getLogger(f"{__name__}.ModelService")
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "[%(asctime)s] [Model] [%(levelname)s] %(message)s"
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.DEBUG)

    @classmethod
    def from_environment(cls) -> "ModelService":
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

    # =========================================================================
    # Initialization
    # =========================================================================

    async def initialize(self) -> bool:
        """
        Initialize model service.

        - Validates configuration
        - Tests Graphiti connectivity
        - Loads existing models from graph

        Returns:
            True if initialization successful
        """
        self.logger.info("Initializing ModelService...")

        # Validate config
        errors = self.config.validate()
        if errors:
            self.logger.warning(f"Configuration warnings: {errors}")

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

        # Load existing models
        await self._load_models_from_graph()

        self.logger.info(f"ModelService ready ({len(self._models)} models loaded)")
        return True

    async def _load_models_from_graph(self) -> None:
        """Load models from Graphiti graph."""
        if not self.config.enabled:
            return

        try:
            response = await self._get_client().search(
                query="MentalModel",
                limit=100,
                entity_types=["MentalModel"],
            )

            if response.success and response.data:
                results = response.data.get("results", response.data.get("edges", []))
                for item in results:
                    model_data = item.get("properties", item)
                    model_id = model_data.get("id", str(uuid4()))

                    self._models[model_id] = MentalModel(
                        id=model_id,
                        name=model_data.get("name", "Unknown"),
                        domain=model_data.get("domain", "task"),
                        description=model_data.get("description"),
                        accuracy=model_data.get("accuracy", 0.5),
                        status=model_data.get("status", "active"),
                    )

                self.logger.debug(f"Loaded {len(self._models)} models from graph")

        except Exception as e:
            self.logger.warning(f"Failed to load models: {e}")

    # =========================================================================
    # Model CRUD
    # =========================================================================

    async def create_model(
        self,
        name: str,
        domain: str = "task",
        description: Optional[str] = None,
        patterns: Optional[List[str]] = None,
        prediction_templates: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> MentalModel:
        """
        Create a new mental model.

        Args:
            name: Model name
            domain: Model domain (user, self, world, task)
            description: Model description
            patterns: Initial patterns this model captures
            prediction_templates: Templates for generating predictions
            metadata: Additional metadata

        Returns:
            Created MentalModel
        """
        model_id = str(uuid4())
        now = datetime.now(timezone.utc)

        model = MentalModel(
            id=model_id,
            name=name,
            domain=domain,
            description=description,
            patterns=patterns or [],
            prediction_templates=prediction_templates or [],
            accuracy=0.5,  # Start neutral
            prediction_count=0,
            error_sum=0.0,
            status="active",
            created_at=now,
            updated_at=now,
            metadata=metadata or {},
        )

        # Store in cache
        self._models[model_id] = model

        # Ingest to Graphiti
        if self.config.enabled:
            await self._ingest_model(model)

        self.logger.info(f"Created model: {name} (id={model_id[:8]}, domain={domain})")
        return model

    async def get_model(self, model_id: str) -> Optional[MentalModel]:
        """Get model by ID."""
        return self._models.get(model_id)

    async def list_models(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 20,
    ) -> List[MentalModel]:
        """
        List models with optional filtering.

        Args:
            domain: Filter by domain
            status: Filter by status
            limit: Max results

        Returns:
            List of MentalModel instances
        """
        models = list(self._models.values())

        if domain:
            models = [m for m in models if m.domain == domain]

        if status:
            models = [m for m in models if m.status == status]

        # Sort by accuracy descending
        models.sort(key=lambda m: m.accuracy, reverse=True)

        return models[:limit]

    async def get_relevant_models(
        self,
        context: Dict[str, Any],
        max_models: int = MAX_MODELS_PER_CONTEXT,
    ) -> List[MentalModel]:
        """
        Get models relevant to context.

        Uses domain hints and accuracy for selection.

        Args:
            context: Current context
            max_models: Max models to return

        Returns:
            List of relevant MentalModel instances
        """
        domain_hint = context.get("domain_hint", context.get("domain"))

        # Get active models
        active_models = [m for m in self._models.values() if m.status == "active"]

        if domain_hint:
            # Prioritize matching domain, then by accuracy
            active_models.sort(
                key=lambda m: (0 if m.domain == domain_hint else 1, -m.accuracy)
            )
        else:
            # Sort by accuracy
            active_models.sort(key=lambda m: -m.accuracy)

        return active_models[:max_models]

    async def update_model(
        self,
        model_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        status: Optional[str] = None,
        patterns: Optional[List[str]] = None,
    ) -> MentalModel:
        """
        Update model properties.

        Args:
            model_id: Model ID
            name: New name
            description: New description
            status: New status
            patterns: Updated patterns

        Returns:
            Updated MentalModel

        Raises:
            ModelNotFoundError: If model not found
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model not found: {model_id}")

        model = self._models[model_id]

        if name is not None:
            model.name = name
        if description is not None:
            model.description = description
        if status is not None:
            model.status = status
        if patterns is not None:
            model.patterns = patterns

        model.updated_at = datetime.now(timezone.utc)

        # Update in Graphiti
        if self.config.enabled:
            await self._ingest_model(model, event_type="model_updated")

        self.logger.info(f"Updated model: {model.name} (id={model_id[:8]})")
        return model

    async def deprecate_model(self, model_id: str) -> MentalModel:
        """
        Deprecate a model (soft delete).

        Args:
            model_id: Model ID

        Returns:
            Deprecated MentalModel
        """
        return await self.update_model(model_id, status="deprecated")

    # =========================================================================
    # Prediction Generation
    # =========================================================================

    async def generate_prediction(
        self,
        model_id: str,
        context: Dict[str, Any],
    ) -> Prediction:
        """
        Generate a prediction from a model.

        Args:
            model_id: Model to use
            context: Context for prediction

        Returns:
            Generated Prediction

        Raises:
            ModelNotFoundError: If model not found
        """
        if model_id not in self._models:
            raise ModelNotFoundError(f"Model not found: {model_id}")

        model = self._models[model_id]

        if model.status != "active":
            raise ModelServiceError(f"Model '{model.name}' is not active")

        # Generate prediction content
        prediction_content = self._generate_prediction_content(model, context)
        confidence = self._estimate_confidence(model, context)

        prediction_id = str(uuid4())
        now = datetime.now(timezone.utc)

        prediction = Prediction(
            id=prediction_id,
            model_id=model_id,
            prediction=prediction_content,
            context=context,
            confidence=confidence,
            created_at=now,
        )

        # Store in cache
        self._predictions[prediction_id] = prediction

        # Ingest to Graphiti
        if self.config.enabled:
            await self._ingest_prediction(prediction, model)

        self.logger.info(
            f"Generated prediction from '{model.name}' "
            f"(confidence={confidence:.2f})"
        )

        return prediction

    def _generate_prediction_content(
        self,
        model: MentalModel,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate prediction content from model and context."""
        user_message = context.get("user_message", context.get("query", ""))

        # Try matching prediction templates
        if model.prediction_templates:
            for template in model.prediction_templates:
                trigger = template.get("trigger", "").lower()
                if trigger and trigger in user_message.lower():
                    return {
                        "source": "template",
                        "template_trigger": template.get("trigger"),
                        "prediction": template.get("predict"),
                        "suggestion": template.get("suggest"),
                        "model_name": model.name,
                        "domain": model.domain,
                    }

        # No template match - generate based on domain
        domain_predictions = {
            "user": "User will likely follow established patterns",
            "self": "Agent should apply learned strategies",
            "world": "Environment will behave as previously observed",
            "task": "Task approach should follow successful patterns",
        }

        return {
            "source": "domain_default",
            "prediction": domain_predictions.get(model.domain, "General prediction"),
            "model_name": model.name,
            "domain": model.domain,
            "patterns": model.patterns[:3] if model.patterns else [],
            "context_summary": user_message[:100] if user_message else None,
        }

    def _estimate_confidence(
        self,
        model: MentalModel,
        context: Dict[str, Any],
    ) -> float:
        """Estimate confidence for prediction."""
        base_confidence = model.accuracy

        # Boost if template matches
        user_message = context.get("user_message", context.get("query", ""))
        if model.prediction_templates:
            for template in model.prediction_templates:
                trigger = template.get("trigger", "").lower()
                if trigger and trigger in user_message.lower():
                    return min(0.9, base_confidence + 0.2)

        return base_confidence

    # =========================================================================
    # Prediction Resolution
    # =========================================================================

    async def resolve_prediction(
        self,
        prediction_id: str,
        observation: Dict[str, Any],
        error: Optional[float] = None,
    ) -> Prediction:
        """
        Resolve a prediction with actual outcome.

        Updates model accuracy based on prediction error.

        Args:
            prediction_id: Prediction to resolve
            observation: What actually happened
            error: Pre-calculated error (if not provided, calculated)

        Returns:
            Resolved Prediction

        Raises:
            PredictionNotFoundError: If prediction not found
        """
        if prediction_id not in self._predictions:
            raise PredictionNotFoundError(f"Prediction not found: {prediction_id}")

        prediction = self._predictions[prediction_id]

        if prediction.resolved_at:
            raise ModelServiceError(f"Prediction already resolved: {prediction_id}")

        # Calculate error if not provided
        if error is None:
            error = self._calculate_error(prediction.prediction, observation)

        # Update prediction
        prediction.observation = observation
        prediction.error = error
        prediction.resolved_at = datetime.now(timezone.utc)

        # Update model accuracy
        await self._update_model_accuracy(prediction.model_id, error)

        # Check if model needs revision
        model = self._models.get(prediction.model_id)
        if model and model.accuracy < REVISION_ERROR_THRESHOLD:
            self.logger.warning(
                f"Model '{model.name}' accuracy ({model.accuracy:.2f}) "
                f"below threshold ({REVISION_ERROR_THRESHOLD}), flagging for revision"
            )
            model.metadata["needs_revision"] = True
            model.metadata["revision_reason"] = "High error rate"

        # Ingest resolution to Graphiti
        if self.config.enabled:
            await self._ingest_resolution(prediction, model)

        self.logger.info(
            f"Resolved prediction {prediction_id[:8]} "
            f"(error={error:.2f}, model_accuracy={model.accuracy:.2f if model else 'N/A'})"
        )

        return prediction

    def _calculate_error(
        self,
        prediction: Dict[str, Any],
        observation: Dict[str, Any],
    ) -> float:
        """
        Calculate error between prediction and observation.

        Returns error score 0.0 (perfect) to 1.0 (completely wrong).
        """
        # Check for explicit accuracy markers
        if observation.get("was_accurate") is True:
            return 0.1
        if observation.get("was_accurate") is False:
            return 0.9

        # Simple word overlap calculation
        pred_text = str(prediction.get("prediction", prediction))
        obs_text = str(observation.get("actual", observation.get("result", observation)))

        pred_words = set(pred_text.lower().split())
        obs_words = set(obs_text.lower().split())

        if not pred_words or not obs_words:
            return 0.5

        overlap = len(pred_words & obs_words)
        total = len(pred_words | obs_words)

        if total == 0:
            return 0.5

        similarity = overlap / total
        return max(0.0, min(1.0, 1.0 - similarity))

    async def _update_model_accuracy(self, model_id: str, error: float) -> None:
        """Update model accuracy based on new prediction error."""
        if model_id not in self._models:
            return

        model = self._models[model_id]

        # Rolling accuracy: weighted average
        model.prediction_count += 1
        model.error_sum += error

        # Calculate new accuracy (1 - average_error)
        avg_error = model.error_sum / model.prediction_count
        model.accuracy = max(0.0, min(1.0, 1.0 - avg_error))
        model.updated_at = datetime.now(timezone.utc)

    async def get_unresolved_predictions(
        self,
        model_id: Optional[str] = None,
        limit: int = 50,
    ) -> List[Prediction]:
        """Get predictions that haven't been resolved."""
        predictions = [
            p for p in self._predictions.values()
            if p.resolved_at is None
        ]

        if model_id:
            predictions = [p for p in predictions if p.model_id == model_id]

        # Sort by created_at descending
        predictions.sort(key=lambda p: p.created_at or datetime.min, reverse=True)

        return predictions[:limit]

    async def get_models_needing_revision(self) -> List[MentalModel]:
        """Get models flagged for revision."""
        return [
            m for m in self._models.values()
            if m.metadata.get("needs_revision") or m.accuracy < REVISION_ERROR_THRESHOLD
        ]

    # =========================================================================
    # Graphiti Integration
    # =========================================================================

    async def _ingest_model(
        self,
        model: MentalModel,
        event_type: str = "model_created",
    ) -> Optional[GraphitiResponse]:
        """Ingest model to Graphiti."""
        try:
            episode_content = f"""
Mental Model Event: {event_type}
Model ID: {model.id}
Model Name: {model.name}
Domain: {model.domain}
Status: {model.status}
Accuracy: {model.accuracy:.2f}
Description: {model.description or 'No description'}
Patterns: {', '.join(model.patterns[:5]) if model.patterns else 'None defined'}
"""

            response = await self._get_client().add_episode(
                content=episode_content,
                metadata={
                    "event_type": event_type,
                    "model_id": model.id,
                    "model_name": model.name,
                    "domain": model.domain,
                    "entity_type": "MentalModel",
                    "source": "memevolve_model_service",
                },
            )

            if response.success:
                self.logger.debug(
                    f"Ingested model {event_type}: {model.name} "
                    f"({response.latency_ms:.0f}ms)"
                )
            else:
                self.logger.warning(f"Model ingest failed: {response.error}")

            return response

        except Exception as e:
            self.logger.error(f"Model ingest error: {e}")
            return None

    async def _ingest_prediction(
        self,
        prediction: Prediction,
        model: MentalModel,
    ) -> Optional[GraphitiResponse]:
        """Ingest prediction to Graphiti."""
        try:
            episode_content = f"""
Prediction Generated
Model: {model.name} (domain: {model.domain})
Prediction ID: {prediction.id}
Confidence: {prediction.confidence:.2f}
Prediction: {json.dumps(prediction.prediction)}
Context: {json.dumps(prediction.context)[:200]}
"""

            response = await self._get_client().add_episode(
                content=episode_content,
                metadata={
                    "event_type": "prediction_generated",
                    "prediction_id": prediction.id,
                    "model_id": model.id,
                    "model_name": model.name,
                    "confidence": prediction.confidence,
                    "source": "memevolve_model_service",
                },
            )

            return response

        except Exception as e:
            self.logger.error(f"Prediction ingest error: {e}")
            return None

    async def _ingest_resolution(
        self,
        prediction: Prediction,
        model: Optional[MentalModel],
    ) -> Optional[GraphitiResponse]:
        """Ingest prediction resolution to Graphiti."""
        try:
            model_name = model.name if model else "Unknown"
            model_accuracy = model.accuracy if model else 0.0

            episode_content = f"""
Prediction Resolved
Model: {model_name}
Prediction ID: {prediction.id}
Error: {prediction.error:.2f}
Model Accuracy (updated): {model_accuracy:.2f}
Observation: {json.dumps(prediction.observation)[:200]}
"""

            response = await self._get_client().add_episode(
                content=episode_content,
                metadata={
                    "event_type": "prediction_resolved",
                    "prediction_id": prediction.id,
                    "model_id": prediction.model_id,
                    "error": prediction.error,
                    "model_accuracy": model_accuracy,
                    "source": "memevolve_model_service",
                },
            )

            return response

        except Exception as e:
            self.logger.error(f"Resolution ingest error: {e}")
            return None

    # =========================================================================
    # Context Manager
    # =========================================================================

    async def __aenter__(self) -> "ModelService":
        await self.initialize()
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.close()


# =============================================================================
# Service Factory
# =============================================================================

_model_service_instance: Optional[ModelService] = None


def get_model_service() -> ModelService:
    """Get or create the ModelService singleton."""
    global _model_service_instance
    if _model_service_instance is None:
        _model_service_instance = ModelService.from_environment()
    return _model_service_instance


def reset_model_service() -> None:
    """Reset the ModelService singleton (for testing)."""
    global _model_service_instance
    _model_service_instance = None
