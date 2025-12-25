"""
Entity Extractor for Dionysus/Graphiti Integration

Extracts entities and relationships from agent trajectories for:
- Graph storage in Neo4j via Graphiti
- Knowledge base building from agent experiences
- Cross-agent knowledge sharing

Features:
- Named entity recognition (people, organizations, tools, sources)
- Relationship extraction (uses, queries, learns_from, etc.)
- PII sanitization (emails, phone numbers, SSN patterns)
- Temporal fact tracking (valid_at timestamps)
- Confidence scoring for extracted entities

Output format is compatible with Graphiti's node/edge schema.
"""

import hashlib
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
from uuid import uuid4

from EvolveLab.memory_types import TrajectoryData

logger = logging.getLogger(__name__)


# Entity types for Graphiti nodes
class EntityType:
    PERSON = "Person"
    ORGANIZATION = "Organization"
    TOOL = "Tool"
    SOURCE = "Source"
    CONCEPT = "Concept"
    TASK = "Task"
    FACT = "Fact"
    PATTERN = "Pattern"


# Relationship types for Graphiti edges
class RelationType:
    USES = "USES"
    QUERIES = "QUERIES"
    LEARNS_FROM = "LEARNS_FROM"
    PRODUCES = "PRODUCES"
    RELATES_TO = "RELATES_TO"
    PART_OF = "PART_OF"
    CEO_OF = "CEO_OF"
    WORKS_AT = "WORKS_AT"
    LOCATED_IN = "LOCATED_IN"
    LISTED_ON = "LISTED_ON"
    FOUNDED_BY = "FOUNDED_BY"


@dataclass
class ExtractedEntity:
    """Represents an extracted entity for Graphiti."""
    id: str
    name: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    source_text: str = ""
    valid_at: Optional[str] = None


@dataclass
class ExtractedEdge:
    """Represents an extracted relationship for Graphiti."""
    id: str
    source_id: str
    target_id: str
    type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.8
    valid_at: Optional[str] = None
    invalid_at: Optional[str] = None


# PII patterns for sanitization
PII_PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
    "phone_us": re.compile(r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "phone_intl": re.compile(r"\b\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b"),
    "ssn": re.compile(r"\b\d{3}[-.\s]?\d{2}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d{4}[-.\s]?){3}\d{4}\b"),
    "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
}

# Common tool patterns
TOOL_PATTERNS = [
    r"\b(search(?:ed)?|crawl(?:ed)?|fetch(?:ed)?|query|queries|queried)\s+(?:using\s+)?(\w+(?:Tool|API|Search|Crawler)?)\b",
    r"\busing\s+(\w+(?:Tool|API|Search))\b",
    r"\b(Google|Bing|DuckDuckGo|Wikipedia|arXiv|PubMed)\s+search\b",
]

# Source patterns (URLs, domains)
SOURCE_PATTERN = re.compile(
    r"\b(?:https?://)?(?:www\.)?([a-zA-Z0-9-]+(?:\.[a-zA-Z]{2,})+)(?:/[^\s]*)?\b"
)


class EntityExtractor:
    """
    Extracts entities and relationships from agent trajectories.

    Usage:
        extractor = EntityExtractor()
        entities, edges = extractor.extract(trajectory_data)
    """

    def __init__(
        self,
        sanitize_pii: bool = True,
        min_confidence: float = 0.5,
        project_id: Optional[str] = None,
    ):
        """
        Initialize the entity extractor.

        Args:
            sanitize_pii: Whether to remove PII from extracted content
            min_confidence: Minimum confidence threshold for entities
            project_id: Project ID for attribution
        """
        self.sanitize_pii = sanitize_pii
        self.min_confidence = min_confidence
        self.project_id = project_id or "memevolve"
        self._seen_entities: Set[str] = set()

    def extract(
        self, trajectory_data: TrajectoryData
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Extract entities and edges from a trajectory.

        Args:
            trajectory_data: Input trajectory with query, steps, result

        Returns:
            (entities, edges) tuple with Graphiti-compatible dicts
        """
        entities: List[ExtractedEntity] = []
        edges: List[ExtractedEdge] = []

        # Reset seen entities for this extraction
        self._seen_entities.clear()

        # Current timestamp for valid_at
        now = datetime.now(timezone.utc).isoformat()

        # Extract from query
        query_entities = self._extract_from_text(
            trajectory_data.query, "query", now
        )
        entities.extend(query_entities)

        # Create task entity for the trajectory
        task_id = self._make_entity_id("task", trajectory_data.query)
        task_entity = ExtractedEntity(
            id=task_id,
            name=self._truncate(trajectory_data.query, 100),
            type=EntityType.TASK,
            properties={
                "full_query": trajectory_data.query,
                "step_count": len(trajectory_data.trajectory),
                "success": trajectory_data.result is not None,
            },
            confidence=1.0,
            valid_at=now,
        )
        entities.append(task_entity)

        # Extract from each step
        for i, step in enumerate(trajectory_data.trajectory):
            step_entities, step_edges = self._extract_from_step(
                step, i, task_id, now
            )
            entities.extend(step_entities)
            edges.extend(step_edges)

        # Extract from result if present
        if trajectory_data.result:
            result_text = str(trajectory_data.result)
            result_entities = self._extract_from_text(result_text, "result", now)
            entities.extend(result_entities)

            # Link result entities to task
            for entity in result_entities:
                edge = ExtractedEdge(
                    id=self._make_edge_id(task_id, entity.id, RelationType.PRODUCES),
                    source_id=task_id,
                    target_id=entity.id,
                    type=RelationType.PRODUCES,
                    confidence=0.9,
                    valid_at=now,
                )
                edges.append(edge)

        # Filter by confidence and deduplicate
        entities = self._deduplicate_entities(entities)
        entities = [e for e in entities if e.confidence >= self.min_confidence]

        # Convert to Graphiti-compatible dicts
        entity_dicts = [self._entity_to_dict(e) for e in entities]
        edge_dicts = [self._edge_to_dict(e) for e in edges]

        logger.debug(
            f"Extracted {len(entity_dicts)} entities, {len(edge_dicts)} edges"
        )

        return entity_dicts, edge_dicts

    def _extract_from_text(
        self, text: str, source: str, timestamp: str
    ) -> List[ExtractedEntity]:
        """Extract entities from a text string."""
        if not text:
            return []

        # Sanitize PII first
        if self.sanitize_pii:
            text = self._sanitize_pii(text)

        entities = []

        # Extract sources (URLs/domains)
        for match in SOURCE_PATTERN.finditer(text):
            domain = match.group(1).lower()
            entity_id = self._make_entity_id("source", domain)

            if entity_id not in self._seen_entities:
                self._seen_entities.add(entity_id)
                entities.append(
                    ExtractedEntity(
                        id=entity_id,
                        name=domain,
                        type=EntityType.SOURCE,
                        properties={"domain": domain, "extracted_from": source},
                        confidence=0.9,
                        source_text=match.group(0),
                        valid_at=timestamp,
                    )
                )

        # Extract tools
        for pattern in TOOL_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                tool_name = match.group(2) if len(match.groups()) > 1 else match.group(1)
                tool_name = tool_name.strip()
                entity_id = self._make_entity_id("tool", tool_name)

                if entity_id not in self._seen_entities:
                    self._seen_entities.add(entity_id)
                    entities.append(
                        ExtractedEntity(
                            id=entity_id,
                            name=tool_name,
                            type=EntityType.TOOL,
                            properties={"extracted_from": source},
                            confidence=0.85,
                            source_text=match.group(0),
                            valid_at=timestamp,
                        )
                    )

        return entities

    def _extract_from_step(
        self,
        step: Dict[str, Any],
        step_index: int,
        task_id: str,
        timestamp: str,
    ) -> Tuple[List[ExtractedEntity], List[ExtractedEdge]]:
        """Extract entities and edges from a trajectory step."""
        entities = []
        edges = []

        # Common step fields
        action = step.get("action", step.get("tool", ""))
        observation = step.get("observation", step.get("result", ""))
        thought = step.get("thought", step.get("reasoning", ""))

        # Extract from action
        if action:
            action_str = str(action)
            action_entities = self._extract_from_text(
                action_str, f"step_{step_index}_action", timestamp
            )
            entities.extend(action_entities)

            # Create edges from task to tools used
            for entity in action_entities:
                if entity.type == EntityType.TOOL:
                    edges.append(
                        ExtractedEdge(
                            id=self._make_edge_id(task_id, entity.id, RelationType.USES),
                            source_id=task_id,
                            target_id=entity.id,
                            type=RelationType.USES,
                            properties={"step": step_index},
                            confidence=0.9,
                            valid_at=timestamp,
                        )
                    )
                elif entity.type == EntityType.SOURCE:
                    edges.append(
                        ExtractedEdge(
                            id=self._make_edge_id(task_id, entity.id, RelationType.QUERIES),
                            source_id=task_id,
                            target_id=entity.id,
                            type=RelationType.QUERIES,
                            properties={"step": step_index},
                            confidence=0.85,
                            valid_at=timestamp,
                        )
                    )

        # Extract from observation
        if observation:
            obs_str = str(observation)
            obs_entities = self._extract_from_text(
                obs_str, f"step_{step_index}_observation", timestamp
            )
            entities.extend(obs_entities)

        # Extract from thought/reasoning
        if thought:
            thought_str = str(thought)
            thought_entities = self._extract_from_text(
                thought_str, f"step_{step_index}_thought", timestamp
            )
            entities.extend(thought_entities)

        return entities, edges

    def _sanitize_pii(self, text: str) -> str:
        """Remove PII patterns from text."""
        for pii_type, pattern in PII_PATTERNS.items():
            text = pattern.sub(f"[{pii_type.upper()}_REDACTED]", text)
        return text

    def _make_entity_id(self, entity_type: str, name: str) -> str:
        """Generate a deterministic entity ID."""
        key = f"{self.project_id}:{entity_type}:{name.lower()}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _make_edge_id(
        self, source_id: str, target_id: str, rel_type: str
    ) -> str:
        """Generate a deterministic edge ID."""
        key = f"{source_id}:{rel_type}:{target_id}"
        return hashlib.md5(key.encode()).hexdigest()[:16]

    def _truncate(self, text: str, max_len: int) -> str:
        """Truncate text to max length."""
        if len(text) <= max_len:
            return text
        return text[: max_len - 3] + "..."

    def _deduplicate_entities(
        self, entities: List[ExtractedEntity]
    ) -> List[ExtractedEntity]:
        """Deduplicate entities by ID, keeping highest confidence."""
        by_id: Dict[str, ExtractedEntity] = {}
        for entity in entities:
            if entity.id not in by_id or entity.confidence > by_id[entity.id].confidence:
                by_id[entity.id] = entity
        return list(by_id.values())

    def _entity_to_dict(self, entity: ExtractedEntity) -> Dict[str, Any]:
        """Convert ExtractedEntity to Graphiti-compatible dict."""
        return {
            "id": entity.id,
            "name": entity.name,
            "type": entity.type,
            "properties": {
                **entity.properties,
                "confidence": entity.confidence,
                "project_id": self.project_id,
            },
            "valid_at": entity.valid_at,
            "source_text": entity.source_text[:200] if entity.source_text else None,
        }

    def _edge_to_dict(self, edge: ExtractedEdge) -> Dict[str, Any]:
        """Convert ExtractedEdge to Graphiti-compatible dict."""
        return {
            "id": edge.id,
            "source_id": edge.source_id,
            "target_id": edge.target_id,
            "type": edge.type,
            "properties": {
                **edge.properties,
                "confidence": edge.confidence,
                "project_id": self.project_id,
            },
            "valid_at": edge.valid_at,
            "invalid_at": edge.invalid_at,
        }
