// ============================================================================
// DIONYSUS MEMORY SYSTEM - Neo4j Schema
// ============================================================================
// Graph schema for Dionysus memory persistence layer.
// Integrates with Graphiti for entity extraction and relationship management.
//
// Memory Architecture:
// - Session: User interaction boundaries for context tracking
// - Memory: Core memory nodes with temporal validity (valid_at/invalid_at)
// - MentalModel: Predictive patterns learned from experience
// - Goal: Agent objectives with priority and status tracking
// - Entity: Graphiti-extracted entities from episodes
// - Episode: Raw experience data from trajectories
// ============================================================================


// ============================================================================
// CONSTRAINTS - Unique Identifiers
// ============================================================================

// Session: Represents a bounded interaction period
// Used for session continuity across devices and context reconstruction
CREATE CONSTRAINT session_id_unique IF NOT EXISTS
FOR (s:Session) REQUIRE s.id IS UNIQUE;

// Memory: Core memory nodes - base constraint for all memory types
CREATE CONSTRAINT memory_id_unique IF NOT EXISTS
FOR (m:Memory) REQUIRE m.id IS UNIQUE;

// Strategic Memory: Long-term patterns and domain knowledge
CREATE CONSTRAINT strategic_memory_id_unique IF NOT EXISTS
FOR (m:Strategic) REQUIRE m.id IS UNIQUE;

// Episodic Memory: Time-bound experiences with context
CREATE CONSTRAINT episodic_memory_id_unique IF NOT EXISTS
FOR (m:Episodic) REQUIRE m.id IS UNIQUE;

// Semantic Memory: Facts and conceptual knowledge
CREATE CONSTRAINT semantic_memory_id_unique IF NOT EXISTS
FOR (m:Semantic) REQUIRE m.id IS UNIQUE;

// Procedural Memory: Skills and how-to knowledge
CREATE CONSTRAINT procedural_memory_id_unique IF NOT EXISTS
FOR (m:Procedural) REQUIRE m.id IS UNIQUE;

// Working Memory: Short-term context (cleared between sessions)
CREATE CONSTRAINT working_memory_id_unique IF NOT EXISTS
FOR (m:Working) REQUIRE m.id IS UNIQUE;

// MentalModel: Predictive patterns with confidence scores
// Used for active inference and expectation generation
CREATE CONSTRAINT mental_model_id_unique IF NOT EXISTS
FOR (m:MentalModel) REQUIRE m.id IS UNIQUE;

// Goal: Agent objectives driving behavior
CREATE CONSTRAINT goal_id_unique IF NOT EXISTS
FOR (g:Goal) REQUIRE g.id IS UNIQUE;

// Entity: Graphiti-extracted entities from natural language
// Entities are extracted automatically from Episode content
CREATE CONSTRAINT entity_id_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.id IS UNIQUE;

// Episode: Raw experience data from agent trajectories
// Contains the natural language text that Graphiti processes
CREATE CONSTRAINT episode_id_unique IF NOT EXISTS
FOR (e:Episode) REQUIRE e.id IS UNIQUE;


// ============================================================================
// INDEXES - Query Performance Optimization
// ============================================================================

// -- Session Indexes --
// Session lookup by user for multi-user systems
CREATE INDEX session_user_id IF NOT EXISTS
FOR (s:Session) ON (s.user_id);

// Session lookup by device for cross-device continuity
CREATE INDEX session_device_id IF NOT EXISTS
FOR (s:Session) ON (s.device_id);

// Session ordering by creation time
CREATE INDEX session_created_at IF NOT EXISTS
FOR (s:Session) ON (s.created_at);


// -- Memory Indexes (applies to all memory types) --
// Memory ordering by creation time for recency queries
CREATE INDEX memory_created_at IF NOT EXISTS
FOR (m:Memory) ON (m.created_at);

// Memory importance for relevance ranking
CREATE INDEX memory_importance IF NOT EXISTS
FOR (m:Memory) ON (m.importance);

// Temporal validity - valid_at for when memory became true
CREATE INDEX memory_valid_at IF NOT EXISTS
FOR (m:Memory) ON (m.valid_at);

// Temporal validity - invalid_at for when memory was superseded
// NULL means still valid; set when newer information invalidates this
CREATE INDEX memory_invalid_at IF NOT EXISTS
FOR (m:Memory) ON (m.invalid_at);


// -- MentalModel Indexes --
// Model lookup by name for pattern matching
CREATE INDEX mental_model_name IF NOT EXISTS
FOR (m:MentalModel) ON (m.name);

// Model ordering by creation time
CREATE INDEX mental_model_created_at IF NOT EXISTS
FOR (m:MentalModel) ON (m.created_at);

// Model confidence for prediction quality ranking
CREATE INDEX mental_model_confidence IF NOT EXISTS
FOR (m:MentalModel) ON (m.confidence);


// -- Goal Indexes --
// Goal filtering by status (active, completed, abandoned)
CREATE INDEX goal_status IF NOT EXISTS
FOR (g:Goal) ON (g.status);

// Goal ordering by priority for action selection
CREATE INDEX goal_priority IF NOT EXISTS
FOR (g:Goal) ON (g.priority);


// -- Entity Indexes (Graphiti integration) --
// Entity lookup by name for search and linking
CREATE INDEX entity_name IF NOT EXISTS
FOR (e:Entity) ON (e.name);

// Entity filtering by type (Person, Tool, Concept, etc.)
CREATE INDEX entity_type IF NOT EXISTS
FOR (e:Entity) ON (e.type);


// -- Episode Indexes --
// Episode ordering by creation time
CREATE INDEX episode_created_at IF NOT EXISTS
FOR (e:Episode) ON (e.created_at);


// ============================================================================
// NODE LABELS - Schema Documentation
// ============================================================================

// Session Node
// Represents a bounded interaction period with temporal context.
// Properties:
//   id: string (UUID) - Unique session identifier
//   created_at: datetime - Session start time
//   user_id: string - User identifier for multi-user systems
//   device_id: string - Device identifier for cross-device continuity
//
// Example:
// CREATE (s:Session {
//   id: 'sess-uuid-1234',
//   created_at: datetime(),
//   user_id: 'user-123',
//   device_id: 'device-abc'
// })


// Memory Node (Base Label)
// All memory types share this base label plus their specific subtype.
// Properties:
//   id: string (UUID) - Unique memory identifier
//   content: string - The actual memory content
//   importance: float (0.0-1.0) - Relevance/importance score
//   created_at: datetime - When memory was stored
//   valid_at: datetime - When this information became true
//   invalid_at: datetime|null - When this was superseded (null = still valid)
//
// Memory Subtypes (additional labels):
//   :Strategic - Long-term patterns, domain knowledge, strategic insights
//   :Episodic - Time-bound experiences, what happened when
//   :Semantic - Facts, concepts, declarative knowledge
//   :Procedural - Skills, how-to knowledge, action sequences
//   :Working - Short-term context, cleared between sessions


// MentalModel Node
// Predictive patterns for active inference and expectation generation.
// Properties:
//   id: string (UUID) - Unique model identifier
//   name: string - Human-readable model name
//   pattern: string - The pattern this model represents
//   prediction: string - What this model predicts
//   confidence: float (0.0-1.0) - Model confidence/reliability
//   created_at: datetime - When model was learned
//
// Example:
// CREATE (m:MentalModel {
//   id: 'model-uuid-5678',
//   name: 'UserPreference:DarkMode',
//   pattern: 'User mentions eye strain or prefers dark themes',
//   prediction: 'User will appreciate dark mode UI recommendations',
//   confidence: 0.85,
//   created_at: datetime()
// })


// Goal Node
// Agent objectives that drive behavior and action selection.
// Properties:
//   id: string (UUID) - Unique goal identifier
//   description: string - What needs to be achieved
//   status: string - 'active', 'completed', 'abandoned', 'blocked'
//   priority: integer (0-100) - Higher = more important
//
// Example:
// CREATE (g:Goal {
//   id: 'goal-uuid-9abc',
//   description: 'Complete user authentication flow',
//   status: 'active',
//   priority: 80
// })


// Entity Node (Graphiti Integration)
// Entities extracted by Graphiti from natural language episode content.
// Properties:
//   id: string (UUID) - Unique entity identifier
//   name: string - Entity name as extracted
//   type: string - Entity type (Person, Tool, Concept, Location, etc.)
//   properties: map - Additional extracted properties
//
// Example:
// CREATE (e:Entity {
//   id: 'entity-uuid-def0',
//   name: 'Python',
//   type: 'Tool',
//   properties: {category: 'programming language', version: '3.11'}
// })


// Episode Node (Graphiti Integration)
// Raw experience data from agent trajectories.
// Graphiti processes this content to extract entities and relationships.
// Properties:
//   id: string (UUID) - Unique episode identifier
//   content: string - Natural language episode text
//   metadata: map - Additional context (session_id, success, etc.)
//   created_at: datetime - When episode occurred
//
// Example:
// CREATE (ep:Episode {
//   id: 'episode-uuid-1234',
//   content: 'Task: Fix authentication bug. Agent succeeded with...',
//   metadata: {session_id: 'sess-123', success: true},
//   created_at: datetime()
// })


// ============================================================================
// RELATIONSHIP TYPES - Schema Documentation
// ============================================================================

// (Session)-[:HAS_MEMORY]->(Memory)
// Links a session to memories created during that session.
// Enables session-scoped memory retrieval and context reconstruction.
//
// Example:
// MATCH (s:Session {id: 'sess-123'})
// CREATE (s)-[:HAS_MEMORY {created_at: datetime()}]->(m:Memory:Episodic {
//   id: 'mem-456',
//   content: 'User asked about authentication',
//   importance: 0.7,
//   created_at: datetime(),
//   valid_at: datetime()
// })


// (Memory)-[:RELATES_TO]->(Memory)
// Semantic relationships between memories.
// Properties:
//   type: string - Relationship type (supports, contradicts, elaborates, etc.)
//   strength: float (0.0-1.0) - Relationship strength
//
// Example:
// MATCH (m1:Memory {id: 'mem-123'}), (m2:Memory {id: 'mem-456'})
// CREATE (m1)-[:RELATES_TO {type: 'supports', strength: 0.8}]->(m2)


// (MentalModel)-[:PREDICTS]->(Outcome)
// Links a mental model to predicted outcomes.
// Outcome can be any node type (Goal, Entity, Memory, etc.)
// Properties:
//   confidence: float - Prediction confidence at time of creation
//   validated: boolean - Whether prediction was validated
//   validated_at: datetime - When prediction was validated
//
// Example:
// MATCH (m:MentalModel {id: 'model-123'}), (g:Goal {id: 'goal-456'})
// CREATE (m)-[:PREDICTS {confidence: 0.75, validated: false}]->(g)


// (Entity)-[:RELATES_TO]->(Entity)
// Graphiti-extracted relationships between entities.
// Properties:
//   type: string - Relationship type (uses, created_by, depends_on, etc.)
//   fact: string - Natural language fact describing relationship
//   valid_at: datetime - When relationship became true
//   invalid_at: datetime|null - When relationship was invalidated
//
// Example:
// MATCH (e1:Entity {name: 'FastAPI'}), (e2:Entity {name: 'Python'})
// CREATE (e1)-[:RELATES_TO {
//   type: 'implemented_in',
//   fact: 'FastAPI is implemented in Python',
//   valid_at: datetime()
// }]->(e2)


// (Episode)-[:CONTAINS]->(Entity)
// Links episodes to entities extracted from them.
// Enables provenance tracking - knowing where entities came from.
// Properties:
//   extracted_at: datetime - When extraction occurred
//   confidence: float - Extraction confidence
//
// Example:
// MATCH (ep:Episode {id: 'ep-123'}), (e:Entity {name: 'Neo4j'})
// CREATE (ep)-[:CONTAINS {extracted_at: datetime(), confidence: 0.9}]->(e)


// (Goal)-[:REQUIRES]->(Goal)
// Goal dependency relationships for hierarchical goal decomposition.
//
// Example:
// MATCH (g1:Goal {id: 'parent-goal'}), (g2:Goal {id: 'sub-goal'})
// CREATE (g1)-[:REQUIRES]->(g2)


// (Memory)-[:SUPERSEDES]->(Memory)
// Tracks when newer memories invalidate older ones.
// Used for temporal validity management.
//
// Example:
// MATCH (m1:Memory {id: 'new-mem'}), (m2:Memory {id: 'old-mem'})
// SET m2.invalid_at = datetime()
// CREATE (m1)-[:SUPERSEDES]->(m2)


// ============================================================================
// FULL-TEXT SEARCH INDEXES
// ============================================================================

// Full-text search on memory content for semantic retrieval
CREATE FULLTEXT INDEX memory_content_fulltext IF NOT EXISTS
FOR (m:Memory) ON EACH [m.content];

// Full-text search on entity names for entity lookup
CREATE FULLTEXT INDEX entity_name_fulltext IF NOT EXISTS
FOR (e:Entity) ON EACH [e.name];

// Full-text search on episode content for experience retrieval
CREATE FULLTEXT INDEX episode_content_fulltext IF NOT EXISTS
FOR (ep:Episode) ON EACH [ep.content];


// ============================================================================
// UTILITY QUERIES - Common Patterns
// ============================================================================

// Get all valid memories for a session (memories not yet invalidated)
// MATCH (s:Session {id: $sessionId})-[:HAS_MEMORY]->(m:Memory)
// WHERE m.invalid_at IS NULL
// RETURN m ORDER BY m.importance DESC, m.created_at DESC

// Get memories by type with temporal validity
// MATCH (m:Strategic)
// WHERE m.valid_at <= datetime() AND (m.invalid_at IS NULL OR m.invalid_at > datetime())
// RETURN m ORDER BY m.importance DESC LIMIT 10

// Get entities related to a specific entity
// MATCH (e1:Entity {name: $entityName})-[r:RELATES_TO]-(e2:Entity)
// RETURN e1, r, e2

// Get episode with all extracted entities
// MATCH (ep:Episode {id: $episodeId})-[:CONTAINS]->(e:Entity)
// RETURN ep, collect(e) as entities

// Get mental model predictions with validation status
// MATCH (m:MentalModel)-[p:PREDICTS]->(outcome)
// WHERE m.confidence > 0.7
// RETURN m.name, m.pattern, p.validated, outcome

// Reconstruct session context
// MATCH (s:Session {id: $sessionId})-[:HAS_MEMORY]->(m:Memory)
// WITH m ORDER BY m.created_at
// RETURN collect(m.content) as sessionContext
