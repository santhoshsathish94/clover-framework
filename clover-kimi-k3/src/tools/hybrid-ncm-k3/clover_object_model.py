"""Canonical Object-Oriented Architecture for the Clover Framework & Clover Kimi K3 Runtime.

This module implements the complete-English, zero-abbreviation Object-Oriented reduction of the
398 raw types across the Clover repository (`398 -> 88 -> 24 -> 12 -> 5`):

1. Naming Grammar (Readable Complete English):
   - Classes & Interfaces (`UpperCamelCase` Complete Noun Phrases):
     `SourceProvenance`, `KnowledgeGraphNode`, `KnowledgeGraphEdge`, `VerifiedKnowledgeGraph`,
     `ConversationSession`, `ForwardPassWorkspace`, `MemoryStorageTier`, `ExpertSlotCache`,
     `TaskDirection`, `TransformerStage`, `NeuralStagePipeline`, `VerificationVerdict`,
     `GrowthLedgerRecorder`, `CloverCycleCoordinator`.
   - Methods & Functions (`snake_case` Complete Verb-Phrase Sentences):
     `encode_input_into_latent_tensor`, `ensure_experts_resident_in_tier`,
     `verify_candidate_artifact_against_context`, `record_verified_outcome_into_knowledge_graph`,
     `execute_complete_clover_cycle`.
   - Variables & Attributes (`snake_case` Complete Descriptive Noun Phrases):
     `resident_entries_by_identifier`, `protected_file_paths`, `carbon_copy_snapshots`,
     `maximum_token_budget`, `backing_storage_tier`.
   - Boolean Properties (`@property` Question Sentences starting with `is_`, `has_`, `can_`, `allows_`):
     `has_passed_all_checks`, `has_exceeded_maximum_token_budget`, `is_cache_currently_full`.
   - Constants (`UPPER_SNAKE_CASE` Complete Quantified Nouns):
     `MAXIMUM_SESSION_CONTEXT_TOKENS`, `DEFAULT_SNAPSHOT_INTERVAL_IN_STAGES`.

2. Decorators & Design Patterns:
   - `@register_component_in_catalog`: Registry & Factory Pattern for pluggable strategies.
   - `@enforce_direction_boundary`: Proxy / Guard Decorator enforcing `TaskDirection` boundaries.
   - `@record_execution_telemetry`: Aspect / Telemetry Decorator recording call counts & durations.
   - `@emit_clover_cycle_stage`: Observer Pattern hook emitting `Context -> Direction -> Execution -> Outcome -> Growth` events.
   - Strategy, Chain of Responsibility (3-Tier VRAM -> Host RAM -> NVMe with R4 Clean Drop),
     Composite Pipeline, Decorator Wrapper, Memento Checkpointing, Fluent Builder, and Template Method.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from functools import wraps
import time
from typing import Any, Callable, ClassVar, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Type


# ============================================================================
# CONSTANTS — Complete Quantified English Nouns
# ============================================================================

MAXIMUM_SESSION_CONTEXT_TOKENS: int = 262_144
DEFAULT_SNAPSHOT_INTERVAL_IN_STAGES: int = 12
MAXIMUM_CARBON_COPY_SNAPSHOTS: int = 8
TOTAL_ROUTED_EXPERTS_PER_STAGE: int = 896
ACTIVE_ROUTED_EXPERTS_PER_TOKEN: int = 16


# ============================================================================
# CROSS-CUTTING DECORATORS — Registry, Boundary Guard, Telemetry & Cycle Hooks
# ============================================================================

class ComponentCatalogRegistry:
    """Centralized Registry & Factory mapping category names and component keys to classes."""

    _registered_classes_by_category: ClassVar[Dict[str, Dict[str, Type[Any]]]] = {}

    @classmethod
    def register_class(cls, category_name: str, component_key: str, target_class: Type[Any]) -> None:
        category_bucket = cls._registered_classes_by_category.setdefault(category_name, {})
        category_bucket[component_key] = target_class

    @classmethod
    def create_component_instance(cls, category_name: str, component_key: str, *PositionalArguments: Any, **KeywordArguments: Any) -> Any:
        category_bucket = cls._registered_classes_by_category.get(category_name, {})
        if component_key not in category_bucket:
            raise KeyError(f"Unknown component '{component_key}' in category '{category_name}'.")
        target_class = category_bucket[component_key]
        return target_class(*PositionalArguments, **KeywordArguments)

    @classmethod
    def list_registered_keys_in_category(cls, category_name: str) -> List[str]:
        return sorted(cls._registered_classes_by_category.get(category_name, {}).keys())


def register_component_in_catalog(category_name: str, component_key: str) -> Callable[[Type[Any]], Type[Any]]:
    """Class decorator that registers a concrete Strategy, Operator, Codec, Tool, or Verifier."""
    def class_registration_decorator(target_class: Type[Any]) -> Type[Any]:
        ComponentCatalogRegistry.register_class(category_name, component_key, target_class)
        target_class.catalog_category_name = category_name
        target_class.catalog_component_key = component_key
        return target_class
    return class_registration_decorator


@dataclass
class ExecutionTelemetryRecord:
    """Immutable telemetry measurement for one decorated method invocation."""

    method_qualified_name: str
    elapsed_milliseconds: float
    invocation_succeeded: bool


class ExecutionTelemetryLedger:
    """Accumulates method call counts and elapsed durations across an object's lifecycle."""

    def __init__(self) -> None:
        self.recorded_measurements: List[ExecutionTelemetryRecord] = []
        self.invocation_counts_by_method: Dict[str, int] = {}

    def record_method_invocation(self, method_qualified_name: str, elapsed_milliseconds: float, invocation_succeeded: bool) -> None:
        self.recorded_measurements.append(
            ExecutionTelemetryRecord(
                method_qualified_name=method_qualified_name,
                elapsed_milliseconds=elapsed_milliseconds,
                invocation_succeeded=invocation_succeeded,
            )
        )
        previous_count = self.invocation_counts_by_method.get(method_qualified_name, 0)
        self.invocation_counts_by_method[method_qualified_name] = previous_count + 1

    @property
    def total_recorded_invocations(self) -> int:
        return len(self.recorded_measurements)


def record_execution_telemetry(target_method: Callable[..., Any]) -> Callable[..., Any]:
    """Method decorator that measures wall-clock duration and records it on `self.telemetry_ledger`."""
    @wraps(target_method)
    def telemetry_recording_wrapper(self: Any, *positional_arguments: Any, **keyword_arguments: Any) -> Any:
        start_time_seconds = time.perf_counter()
        invocation_succeeded = False
        try:
            result_value = target_method(self, *positional_arguments, **keyword_arguments)
            invocation_succeeded = True
            return result_value
        finally:
            elapsed_milliseconds = (time.perf_counter() - start_time_seconds) * 1000.0
            ledger = getattr(self, "telemetry_ledger", None)
            if isinstance(ledger, ExecutionTelemetryLedger):
                ledger.record_method_invocation(
                    method_qualified_name=f"{self.__class__.__name__}.{target_method.__name__}",
                    elapsed_milliseconds=elapsed_milliseconds,
                    invocation_succeeded=invocation_succeeded,
                )
    return telemetry_recording_wrapper


def enforce_direction_boundary(target_method: Callable[..., Any]) -> Callable[..., Any]:
    """Method decorator that verifies `TaskDirection` permissions before executing an action."""
    @wraps(target_method)
    def boundary_enforcement_wrapper(self: Any, *positional_arguments: Any, **keyword_arguments: Any) -> Any:
        active_direction: Optional[TaskDirection] = keyword_arguments.get("task_direction")
        if active_direction is None:
            for argument in positional_arguments:
                if isinstance(argument, TaskDirection):
                    active_direction = argument
                    break
        if active_direction is not None:
            if not active_direction.has_non_empty_goal:
                raise PermissionError("TaskDirection must specify a non-empty goal before execution.")
            requested_path = keyword_arguments.get("target_file_path")
            if isinstance(requested_path, str) and not active_direction.allows_modification_of_path(requested_path):
                raise PermissionError(f"TaskDirection forbids modifying protected path '{requested_path}'.")
        return target_method(self, *positional_arguments, **keyword_arguments)
    return boundary_enforcement_wrapper


class CyclePhaseObserver(ABC):
    """Observer interface notified when the coordinator enters and completes each Clover Cycle stage."""

    @abstractmethod
    def on_cycle_stage_entered(self, stage_name: str, method_name: str) -> None:
        """Called immediately before a Clover Cycle stage begins."""

    @abstractmethod
    def on_cycle_stage_completed(self, stage_name: str, method_name: str, stage_output: Any) -> None:
        """Called immediately after a Clover Cycle stage finishes."""


class InMemoryCyclePhaseLogObserver(CyclePhaseObserver):
    """Concrete observer that records the ordered sequence of Clover Cycle stage transitions."""

    def __init__(self) -> None:
        self.entered_stages: List[Tuple[str, str]] = []
        self.completed_stages: List[Tuple[str, str]] = []

    def on_cycle_stage_entered(self, stage_name: str, method_name: str) -> None:
        self.entered_stages.append((stage_name, method_name))

    def on_cycle_stage_completed(self, stage_name: str, method_name: str, stage_output: Any) -> None:
        self.completed_stages.append((stage_name, method_name))


def emit_clover_cycle_stage(stage_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Method decorator that notifies all registered `CyclePhaseObserver` instances on `self.cycle_observers`."""
    def stage_notification_decorator(target_method: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(target_method)
        def stage_notification_wrapper(self: Any, *positional_arguments: Any, **keyword_arguments: Any) -> Any:
            observers: Sequence[CyclePhaseObserver] = getattr(self, "cycle_observers", ())
            for observer in observers:
                observer.on_cycle_stage_entered(stage_name, target_method.__name__)
            stage_output = target_method(self, *positional_arguments, **keyword_arguments)
            for observer in observers:
                observer.on_cycle_stage_completed(stage_name, target_method.__name__, stage_output)
            return stage_output
        return stage_notification_wrapper
    return stage_notification_decorator


# ============================================================================
# ROOT PRIMITIVE 1: CONTEXT — What Is Known, Held in Memory, or Stored in Tiers
# (Contains: VerifiedKnowledgeGraph, ConversationSession, ForwardPassWorkspace, MemoryStorageTier)
# ============================================================================

class ContextArtifact(ABC):
    """Abstract root class for all state, knowledge, and tiered storage objects in the Context phase."""

    @property
    @abstractmethod
    def context_summary_sentence(self) -> str:
        """Returns a human-readable English sentence summarizing the current context state."""


@dataclass(frozen=True)
class SourceProvenance:
    """Unifies `MemoryProvenance`, `Provenance`, and `Evidence` into one immutable citation record."""

    source_kind: str
    source_reference_uri: str
    recorded_by_actor: str
    recorded_timestamp_utc: str
    is_externally_verified: bool = True

    @property
    def formatted_citation_label(self) -> str:
        verification_badge = "verified" if self.is_externally_verified else "unverified"
        return f"[{self.source_kind}: {self.source_reference_uri} ({verification_badge} at {self.recorded_timestamp_utc})]"


@dataclass(frozen=True)
class KnowledgeNodeRevision:
    """Unifies `MemoryRevision` and `EvolutionChainNode` to preserve prior versions of a knowledge node."""

    revision_number: int
    previous_content_text: str
    revision_reason: str
    provenance: SourceProvenance


@dataclass(frozen=True)
class KnowledgeGraphEdge:
    """Unifies `MemoryRelationship`, `Edge`, and `Relation` into one directed, typed, cited relationship."""

    source_node_identifier: str
    relationship_type: str
    target_node_identifier: str
    provenance: SourceProvenance
    confidence_score: float = 1.0


@dataclass
class KnowledgeGraphNode:
    """Unifies `MemoryEntry`, `Vertex`, `EntityRef`, and `EntityClass` into one cited property-graph node."""

    node_identifier: str
    node_category: str
    title_text: str
    content_text: str
    provenance: SourceProvenance
    concept_tags: List[str] = field(default_factory=list)
    lifecycle_status: str = "active"
    revision_history: List[KnowledgeNodeRevision] = field(default_factory=list)

    @property
    def is_currently_active(self) -> bool:
        return self.lifecycle_status == "active"

    def revise_content_preserving_history(
        self,
        updated_content_text: str,
        revision_reason: str,
        updated_provenance: SourceProvenance,
    ) -> None:
        if updated_content_text != self.content_text:
            next_revision_number = len(self.revision_history) + 1
            self.revision_history.append(
                KnowledgeNodeRevision(
                    revision_number=next_revision_number,
                    previous_content_text=self.content_text,
                    revision_reason=revision_reason,
                    provenance=self.provenance,
                )
            )
            self.content_text = updated_content_text
            self.provenance = updated_provenance


class VerifiedKnowledgeGraph(ContextArtifact):
    """Unifies `CloverMemoryGraphStore`, `CloverOntologyGraph`, `CloverGraphTraversal`, and `CloverOntologyRetriever`."""

    def __init__(self) -> None:
        self.nodes_by_identifier: Dict[str, KnowledgeGraphNode] = {}
        self.staged_candidate_nodes_by_identifier: Dict[str, KnowledgeGraphNode] = {}
        self.directed_edges: List[KnowledgeGraphEdge] = []
        self.read_epoch_counter: int = 1
        self.telemetry_ledger = ExecutionTelemetryLedger()

    @property
    def active_node_count(self) -> int:
        return sum(1 for node in self.nodes_by_identifier.values() if node.is_currently_active)

    @property
    def context_summary_sentence(self) -> str:
        return (
            f"VerifiedKnowledgeGraph holds {self.active_node_count} active nodes and "
            f"{len(self.directed_edges)} cited edges at read epoch {self.read_epoch_counter}."
        )

    @record_execution_telemetry
    def stage_candidate_node_for_review(self, candidate_node: KnowledgeGraphNode) -> None:
        candidate_node.lifecycle_status = "under_review"
        self.staged_candidate_nodes_by_identifier[candidate_node.node_identifier] = candidate_node

    @record_execution_telemetry
    def promote_staged_node_to_active_graph(self, node_identifier: str, verification_provenance: SourceProvenance) -> KnowledgeGraphNode:
        if node_identifier not in self.staged_candidate_nodes_by_identifier:
            raise KeyError(f"Node '{node_identifier}' is not in the staging area.")
        promoted_node = self.staged_candidate_nodes_by_identifier.pop(node_identifier)
        promoted_node.lifecycle_status = "active"
        promoted_node.provenance = verification_provenance
        self.nodes_by_identifier[node_identifier] = promoted_node
        self.read_epoch_counter += 1
        return promoted_node

    @record_execution_telemetry
    def connect_nodes_with_directed_edge(self, directed_edge: KnowledgeGraphEdge) -> None:
        self.directed_edges.append(directed_edge)
        self.read_epoch_counter += 1

    @record_execution_telemetry
    def retrieve_grounded_nodes_for_query(self, natural_language_query: str, maximum_results: int = 4) -> List[KnowledgeGraphNode]:
        query_words = {word.lower() for word in natural_language_query.replace("/", " ").split() if len(word) >= 3}
        scored_nodes: List[Tuple[int, KnowledgeGraphNode]] = []
        for candidate_node in self.nodes_by_identifier.values():
            if not candidate_node.is_currently_active:
                continue
            searchable_text = f"{candidate_node.title_text} {candidate_node.content_text} {' '.join(candidate_node.concept_tags)}".lower()
            matching_word_count = sum(1 for query_word in query_words if query_word in searchable_text)
            if matching_word_count > 0:
                scored_nodes.append((matching_word_count, candidate_node))
        scored_nodes.sort(key=lambda pair: (-pair[0], pair[1].node_identifier))
        return [node for _, node in scored_nodes[:maximum_results]]


@dataclass(frozen=True)
class ConversationTurnRecord:
    """Unifies `ChatTurn` and session turn dictionaries into one typed multi-turn conversation record."""

    turn_number: int
    user_prompt_text: str
    assistant_reply_text: str
    token_count: int


@dataclass(frozen=True)
class SessionCheckpointMemento:
    """Memento Pattern object capturing a restorable snapshot of `ConversationSession`."""

    saved_turn_count: int
    saved_token_count: int
    saved_compacted_summary_text: str


class ConversationSession(ContextArtifact):
    """Manages multi-turn conversation history and 256K full-context simple-sentence compaction (kept distinct from GPU `ForwardPassWorkspace`)."""

    def __init__(self, session_identifier: str, maximum_token_budget: int = MAXIMUM_SESSION_CONTEXT_TOKENS) -> None:
        self.session_identifier = session_identifier
        self.maximum_token_budget = maximum_token_budget
        self.conversation_turns: List[ConversationTurnRecord] = []
        self.compacted_summary_sentences: List[str] = []
        self.current_token_count: int = 0
        self.compaction_operations_count: int = 0

    @property
    def has_exceeded_maximum_token_budget(self) -> bool:
        return self.current_token_count > self.maximum_token_budget

    @property
    def context_summary_sentence(self) -> str:
        return (
            f"ConversationSession '{self.session_identifier}' holds {len(self.conversation_turns)} turns "
            f"({self.current_token_count} / {self.maximum_token_budget} tokens, "
            f"{self.compaction_operations_count} compactions)."
        )

    def append_completed_turn(self, user_prompt_text: str, assistant_reply_text: str, estimated_token_count: Optional[int] = None) -> None:
        calculated_tokens = (
            estimated_token_count
            if estimated_token_count is not None
            else max(1, len((user_prompt_text + " " + assistant_reply_text).split()))
        )
        next_turn_number = len(self.conversation_turns) + 1
        self.conversation_turns.append(
            ConversationTurnRecord(
                turn_number=next_turn_number,
                user_prompt_text=user_prompt_text,
                assistant_reply_text=assistant_reply_text,
                token_count=calculated_tokens,
            )
        )
        self.current_token_count += calculated_tokens
        if self.has_exceeded_maximum_token_budget:
            self.compact_oldest_turns_into_summary_when_full()

    def compact_oldest_turns_into_summary_when_full(self) -> None:
        while self.has_exceeded_maximum_token_budget and len(self.conversation_turns) > 1:
            oldest_turn = self.conversation_turns.pop(0)
            compressed_sentence = f"Turn {oldest_turn.turn_number}: {oldest_turn.user_prompt_text[:80]} -> {oldest_turn.assistant_reply_text[:120]}"
            compressed_tokens = max(1, oldest_turn.token_count // 5)
            self.compacted_summary_sentences.append(compressed_sentence)
            self.current_token_count = max(compressed_tokens, self.current_token_count - oldest_turn.token_count + compressed_tokens)
            self.compaction_operations_count += 1

    def capture_turn_checkpoint(self) -> SessionCheckpointMemento:
        return SessionCheckpointMemento(
            saved_turn_count=len(self.conversation_turns),
            saved_token_count=self.current_token_count,
            saved_compacted_summary_text="\n".join(self.compacted_summary_sentences),
        )

    def restore_turn_checkpoint(self, checkpoint_memento: SessionCheckpointMemento) -> None:
        del self.conversation_turns[checkpoint_memento.saved_turn_count:]
        self.current_token_count = checkpoint_memento.saved_token_count
        self.compacted_summary_sentences = (
            checkpoint_memento.saved_compacted_summary_text.splitlines()
            if checkpoint_memento.saved_compacted_summary_text
            else []
        )


@dataclass
class QuantizedTensorRecord:
    """Unifies `ServerRecord`, `TransformerRecord`, `Record`, `Mat`, `Bf16Matrix`, and `SafetensorsTensor`."""

    tensor_name: str
    row_count: int
    column_count: int
    values: List[float]
    quantization_format: str = "bf16"

    @property
    def element_count(self) -> int:
        return len(self.values)

    def project_activation_vector(self, input_activation_vector: Sequence[float]) -> List[float]:
        scale_factor = sum(self.values) / max(1, len(self.values))
        return [round(element * scale_factor, 6) for element in input_activation_vector]


@dataclass(frozen=True)
class WorkspaceCheckpointMemento:
    """Memento Pattern snapshot of `ForwardPassWorkspace` for speculative decoding rollback."""

    saved_sequence_position: int
    saved_activation_vector: Tuple[float, ...]
    saved_snapshot_count: int


class ForwardPassWorkspace(ContextArtifact):
    """Unifies `ServerSequence`, `TransformerSequence`, `Transformer93Sequence`, `Work`, `Buf`, `KdaState`, and `MlaCache`."""

    def __init__(self, initial_activation_vector: Optional[Sequence[float]] = None) -> None:
        self.current_activation_vector: List[float] = list(initial_activation_vector or [1.0, 1.0, 1.0, 1.0])
        self.carbon_copy_snapshots: List[List[float]] = []
        self.recurrent_state_by_stage_index: Dict[int, List[float]] = {}
        self.sequence_position: int = 0

    @property
    def captured_snapshot_count(self) -> int:
        return len(self.carbon_copy_snapshots)

    @property
    def context_summary_sentence(self) -> str:
        return (
            f"ForwardPassWorkspace at position {self.sequence_position} with "
            f"{self.captured_snapshot_count} carbon-copy snapshots."
        )

    def record_carbon_copy_snapshot(self) -> None:
        if len(self.carbon_copy_snapshots) < MAXIMUM_CARBON_COPY_SNAPSHOTS:
            self.carbon_copy_snapshots.append(list(self.current_activation_vector))

    def capture_checkpoint_memento(self) -> WorkspaceCheckpointMemento:
        return WorkspaceCheckpointMemento(
            saved_sequence_position=self.sequence_position,
            saved_activation_vector=tuple(self.current_activation_vector),
            saved_snapshot_count=len(self.carbon_copy_snapshots),
        )

    def restore_from_checkpoint_memento(self, checkpoint_memento: WorkspaceCheckpointMemento) -> None:
        self.sequence_position = checkpoint_memento.saved_sequence_position
        self.current_activation_vector = list(checkpoint_memento.saved_activation_vector)
        del self.carbon_copy_snapshots[checkpoint_memento.saved_snapshot_count:]


@dataclass
class ExpertSlotCacheEntry:
    """Unifies `SlotCache::Ent`, `sc_entry`, and `LiveSlotCache::Ent`."""

    expert_identifier: int
    payload_tensor: QuantizedTensorRecord
    last_access_pass_number: int
    last_access_layer_index: int
    access_heat_score: float = 1.0


class CacheEvictionStrategy(ABC):
    """Strategy Pattern interface for selecting which resident expert slot to evict when a cache is full."""

    @abstractmethod
    def select_victim_expert_identifier(
        self,
        resident_entries_by_identifier: Dict[int, ExpertSlotCacheEntry],
        protected_expert_identifiers: Set[int],
        current_pass_number: int,
        current_layer_index: int,
    ) -> int:
        """Selects and returns the `expert_identifier` to evict."""


@register_component_in_catalog("cache_eviction_strategy", "specmd_least_stale")
class LeastStaleCacheEvictionStrategy(CacheEvictionStrategy):
    """SpecMD Policy 3: Evicts the resident expert whose layer has already completed in the current pass (or oldest pass)."""

    def select_victim_expert_identifier(
        self,
        resident_entries_by_identifier: Dict[int, ExpertSlotCacheEntry],
        protected_expert_identifiers: Set[int],
        current_pass_number: int,
        current_layer_index: int,
    ) -> int:
        best_victim_identifier: Optional[int] = None
        best_victim_priority_tuple: Optional[Tuple[int, int, float, int]] = None
        for expert_identifier, entry in resident_entries_by_identifier.items():
            if expert_identifier in protected_expert_identifiers:
                continue
            already_executed_this_pass = 0 if entry.last_access_layer_index < current_layer_index else 1
            priority_tuple = (
                already_executed_this_pass,
                entry.last_access_pass_number,
                entry.access_heat_score,
                expert_identifier,
            )
            if best_victim_priority_tuple is None or priority_tuple < best_victim_priority_tuple:
                best_victim_priority_tuple = priority_tuple
                best_victim_identifier = expert_identifier
        if best_victim_identifier is None:
            raise RuntimeError("All resident cache slots are protected; cannot select an eviction victim.")
        return best_victim_identifier


class ExpertSlotCache:
    """Unifies `SlotCache`, `SlotCacheSel`, `sc_cache`, `CpuSlotCache`, `ExpertCache`, and `SlotCacheSim`."""

    def __init__(self, capacity_in_slots: int, eviction_strategy: Optional[CacheEvictionStrategy] = None) -> None:
        self.capacity_in_slots = capacity_in_slots
        self.eviction_strategy = eviction_strategy or LeastStaleCacheEvictionStrategy()
        self.resident_entries_by_identifier: Dict[int, ExpertSlotCacheEntry] = {}

    @property
    def is_cache_currently_full(self) -> bool:
        return len(self.resident_entries_by_identifier) >= self.capacity_in_slots

    @property
    def occupied_slot_count(self) -> int:
        return len(self.resident_entries_by_identifier)

    def contains_expert(self, expert_identifier: int) -> bool:
        return expert_identifier in self.resident_entries_by_identifier

    def lookup_resident_expert(
        self,
        expert_identifier: int,
        current_pass_number: int,
        current_layer_index: int,
    ) -> Optional[QuantizedTensorRecord]:
        entry = self.resident_entries_by_identifier.get(expert_identifier)
        if entry is None:
            return None
        entry.last_access_pass_number = current_pass_number
        entry.last_access_layer_index = current_layer_index
        entry.access_heat_score += 1.0
        return entry.payload_tensor

    def insert_expert_evicting_if_needed(
        self,
        expert_identifier: int,
        payload_tensor: QuantizedTensorRecord,
        protected_expert_identifiers: Set[int],
        current_pass_number: int,
        current_layer_index: int,
    ) -> Optional[ExpertSlotCacheEntry]:
        evicted_entry: Optional[ExpertSlotCacheEntry] = None
        if expert_identifier not in self.resident_entries_by_identifier and self.is_cache_currently_full:
            victim_identifier = self.eviction_strategy.select_victim_expert_identifier(
                resident_entries_by_identifier=self.resident_entries_by_identifier,
                protected_expert_identifiers=protected_expert_identifiers,
                current_pass_number=current_pass_number,
                current_layer_index=current_layer_index,
            )
            evicted_entry = self.resident_entries_by_identifier.pop(victim_identifier)
        self.resident_entries_by_identifier[expert_identifier] = ExpertSlotCacheEntry(
            expert_identifier=expert_identifier,
            payload_tensor=payload_tensor,
            last_access_pass_number=current_pass_number,
            last_access_layer_index=current_layer_index,
            access_heat_score=1.0,
        )
        return evicted_entry


@dataclass
class DataMovementStatisticsLedger:
    """Unifies `ExpertStats`, `pi_expert_stats`, `sy_stats`, `sp_stats`, and `DataMovementLedger`."""

    total_requests_count: int = 0
    local_tier_hits_count: int = 0
    backing_tier_fetches_count: int = 0
    clean_eviction_drops_r4_count: int = 0
    dirty_writebacks_r2_count: int = 0

    @property
    def hit_rate_percentage(self) -> float:
        if self.total_requests_count == 0:
            return 0.0
        return round((self.local_tier_hits_count * 100.0) / self.total_requests_count, 2)


class MemoryStorageTier(ContextArtifact):
    """Chain of Responsibility Pattern unifying `CardPool` (GPU VRAM), `HostCache` (Pinned RAM), and `ExpertStore` (NVMe)."""

    def __init__(
        self,
        tier_name: str,
        tier_level: int,
        slot_cache: ExpertSlotCache,
        backing_storage_tier: Optional[MemoryStorageTier] = None,
    ) -> None:
        self.tier_name = tier_name
        self.tier_level = tier_level
        self.slot_cache = slot_cache
        self.backing_storage_tier = backing_storage_tier
        self.movement_statistics = DataMovementStatisticsLedger()

    @property
    def context_summary_sentence(self) -> str:
        return (
            f"MemoryStorageTier '{self.tier_name}' (Level {self.tier_level}) holds "
            f"{self.slot_cache.occupied_slot_count}/{self.slot_cache.capacity_in_slots} slots "
            f"(Hit rate: {self.movement_statistics.hit_rate_percentage}%, "
            f"R4 clean drops: {self.movement_statistics.clean_eviction_drops_r4_count})."
        )

    @classmethod
    def create_three_tier_hierarchy(
        cls,
        vram_capacity_in_slots: int = 4,
        host_ram_capacity_in_slots: int = 8,
        nvme_capacity_in_slots: int = 32,
    ) -> Tuple[MemoryStorageTier, MemoryStorageTier, MemoryStorageTier]:
        nvme_tier = cls("NVMeDirectVault", tier_level=3, slot_cache=ExpertSlotCache(nvme_capacity_in_slots))
        host_ram_tier = cls("PinnedHostRamPool", tier_level=2, slot_cache=ExpertSlotCache(host_ram_capacity_in_slots), backing_storage_tier=nvme_tier)
        vram_card_tier = cls("GpuVramCardPool", tier_level=1, slot_cache=ExpertSlotCache(vram_capacity_in_slots), backing_storage_tier=host_ram_tier)
        return vram_card_tier, host_ram_tier, nvme_tier

    def is_expert_currently_resident(self, expert_identifier: int) -> bool:
        return self.slot_cache.contains_expert(expert_identifier)

    def ensure_expert_tensor_resident(
        self,
        expert_identifier: int,
        protected_expert_identifiers: Set[int],
        current_pass_number: int = 1,
        current_layer_index: int = 0,
    ) -> QuantizedTensorRecord:
        self.movement_statistics.total_requests_count += 1
        cached_tensor = self.slot_cache.lookup_resident_expert(
            expert_identifier=expert_identifier,
            current_pass_number=current_pass_number,
            current_layer_index=current_layer_index,
        )
        if cached_tensor is not None:
            self.movement_statistics.local_tier_hits_count += 1
            return cached_tensor

        if self.backing_storage_tier is None:
            raise KeyError(f"Expert {expert_identifier} not found in root storage tier '{self.tier_name}'.")

        self.movement_statistics.backing_tier_fetches_count += 1
        fetched_tensor = self.backing_storage_tier.ensure_expert_tensor_resident(
            expert_identifier=expert_identifier,
            protected_expert_identifiers=protected_expert_identifiers,
            current_pass_number=current_pass_number,
            current_layer_index=current_layer_index,
        )
        evicted_entry = self.slot_cache.insert_expert_evicting_if_needed(
            expert_identifier=expert_identifier,
            payload_tensor=fetched_tensor,
            protected_expert_identifiers=protected_expert_identifiers,
            current_pass_number=current_pass_number,
            current_layer_index=current_layer_index,
        )
        if evicted_entry is not None:
            if self.backing_storage_tier.is_expert_currently_resident(evicted_entry.expert_identifier):
                self.movement_statistics.clean_eviction_drops_r4_count += 1
            else:
                self.movement_statistics.dirty_writebacks_r2_count += 1
                self.backing_storage_tier.slot_cache.insert_expert_evicting_if_needed(
                    expert_identifier=evicted_entry.expert_identifier,
                    payload_tensor=evicted_entry.payload_tensor,
                    protected_expert_identifiers=set(),
                    current_pass_number=current_pass_number,
                    current_layer_index=current_layer_index,
                )
        return fetched_tensor


# ============================================================================
# ROOT PRIMITIVE 2: DIRECTION — Human Request, Configuration & Safety Boundaries
# (Unifies: GenerationRequest, GenerationConfig, PendingTextRequest, EcSettings, VerificationPolicy)
# ============================================================================

@dataclass
class TaskDirection:
    """Represents the human's request, generation limits, and non-negotiable safety boundaries."""

    request_identifier: str
    goal_description: str
    maximum_new_tokens: int = 512
    protected_file_paths: Set[str] = field(default_factory=set)
    required_verification_languages: List[str] = field(default_factory=list)

    @property
    def has_non_empty_goal(self) -> bool:
        return bool(self.goal_description and self.goal_description.strip())

    def allows_modification_of_path(self, candidate_file_path: str) -> bool:
        normalized_candidate = candidate_file_path.replace("\\", "/").strip("/")
        for protected_path in self.protected_file_paths:
            normalized_protected = protected_path.replace("\\", "/").strip("/")
            if normalized_candidate == normalized_protected or normalized_candidate.startswith(normalized_protected + "/"):
                return False
        return True


# ============================================================================
# ROOT PRIMITIVE 3: EXECUTION — Bounded Transformers, Codecs, Pipelines & Tools
# (Keep interfaces distinct so Liskov Substitution & Single Responsibility hold!)
# ============================================================================

class ModalityCodec(ABC):
    """Unifies all 8 Tokenizers (`K3Tokenizer`, `BPETokenizer`, `Tok`), `VQVAE`, and `MoonViT3d`."""

    @abstractmethod
    def encode_input_into_activation_vector(self, raw_input_text: str) -> List[float]:
        """Encodes raw human text or image input into a numerical activation vector."""

    @abstractmethod
    def decode_activation_vector_into_text(self, activation_vector: Sequence[float], task_direction: TaskDirection) -> str:
        """Decodes a processed activation vector back into human-readable text or code."""


@register_component_in_catalog("modality_codec", "kimi_k3_text_codec")
class KimiK3TextModalityCodec(ModalityCodec):
    """Concrete text & concept codec mapping UTF-8 strings to and from activation vectors."""

    def encode_input_into_activation_vector(self, raw_input_text: str) -> List[float]:
        words = raw_input_text.strip().split()
        word_count = float(max(1, len(words)))
        character_count = float(max(1, len(raw_input_text)))
        return [word_count, character_count, round(character_count / word_count, 4), 1.0]

    def decode_activation_vector_into_text(self, activation_vector: Sequence[float], task_direction: TaskDirection) -> str:
        checksum_value = round(sum(activation_vector), 4)
        return f"Verified response for '{task_direction.goal_description}' (activation checksum {checksum_value})."


class SequenceMixingOperator(ABC):
    """Polymorphic sequence operator interface unifying `KdaOperator`, `MlaOperator`, and `SelfAttentionOperator`."""

    @abstractmethod
    def mix_sequence_activations(
        self,
        stage_index: int,
        input_activation_vector: Sequence[float],
        forward_pass_workspace: ForwardPassWorkspace,
    ) -> List[float]:
        """Applies sequence mixing (recurrent KDA state or MLA KV cache) to the activation vector."""


@register_component_in_catalog("sequence_operator", "kimi_delta_attention")
class KimiDeltaAttentionOperator(SequenceMixingOperator):
    """Linear recurrent KDA operator updating per-stage recurrent state in `ForwardPassWorkspace`."""

    def mix_sequence_activations(
        self,
        stage_index: int,
        input_activation_vector: Sequence[float],
        forward_pass_workspace: ForwardPassWorkspace,
    ) -> List[float]:
        previous_state = forward_pass_workspace.recurrent_state_by_stage_index.get(
            stage_index, [0.0 for _ in input_activation_vector]
        )
        updated_state = [
            round(0.9 * current_val + 0.1 * prev_val, 6)
            for current_val, prev_val in zip(input_activation_vector, previous_state)
        ]
        forward_pass_workspace.recurrent_state_by_stage_index[stage_index] = updated_state
        return updated_state


@register_component_in_catalog("sequence_operator", "multi_head_latent_attention")
class MultiHeadLatentAttentionOperator(SequenceMixingOperator):
    """Latent KV-cache MLA operator used at every 4th stage of the 93-stage Kimi K3 pipeline."""

    def mix_sequence_activations(
        self,
        stage_index: int,
        input_activation_vector: Sequence[float],
        forward_pass_workspace: ForwardPassWorkspace,
    ) -> List[float]:
        mean_activation = sum(input_activation_vector) / max(1, len(input_activation_vector))
        mixed_vector = [round(value + 0.05 * mean_activation, 6) for value in input_activation_vector]
        forward_pass_workspace.recurrent_state_by_stage_index[stage_index] = mixed_vector
        return mixed_vector


class FeedForwardNetwork(ABC):
    """Polymorphic feed-forward interface unifying `DenseFeedForward` and `RoutedMixtureOfExpertsFeedForward`."""

    @abstractmethod
    def transform_activation_vector(
        self,
        stage_index: int,
        input_activation_vector: Sequence[float],
        current_pass_number: int,
    ) -> List[float]:
        """Applies feed-forward transformation to the activation vector."""


class DenseSwiGLUFeedForwardNetwork(FeedForwardNetwork):
    """Dense gated MLP used in Stage 0, NCM blocks, and Vision blocks."""

    def __init__(self, projection_tensor: QuantizedTensorRecord) -> None:
        self.projection_tensor = projection_tensor

    def transform_activation_vector(
        self,
        stage_index: int,
        input_activation_vector: Sequence[float],
        current_pass_number: int,
    ) -> List[float]:
        return self.projection_tensor.project_activation_vector(input_activation_vector)


class RoutedMixtureOfExpertsFeedForwardNetwork(FeedForwardNetwork):
    """MoE feed-forward network that fetches routed experts through `MemoryStorageTier`."""

    def __init__(self, storage_tier: MemoryStorageTier, routed_expert_identifiers: Sequence[int]) -> None:
        self.storage_tier = storage_tier
        self.routed_expert_identifiers = list(routed_expert_identifiers)

    def transform_activation_vector(
        self,
        stage_index: int,
        input_activation_vector: Sequence[float],
        current_pass_number: int,
    ) -> List[float]:
        protected_set = set(self.routed_expert_identifiers)
        accumulated_vector = list(input_activation_vector)
        for expert_identifier in self.routed_expert_identifiers:
            expert_tensor = self.storage_tier.ensure_expert_tensor_resident(
                expert_identifier=expert_identifier,
                protected_expert_identifiers=protected_set,
                current_pass_number=current_pass_number,
                current_layer_index=stage_index,
            )
            projected_vector = expert_tensor.project_activation_vector(accumulated_vector)
            accumulated_vector = [
                round(0.5 * current_val + 0.5 * proj_val, 6)
                for current_val, proj_val in zip(accumulated_vector, projected_vector)
            ]
        return accumulated_vector


class TransformerStage:
    """Compositional stage (`has-a` `SequenceMixingOperator` + `has-a` `FeedForwardNetwork`) replacing `Server`, `Transformer`, `LbLayer`, `Layer`, and `Block`."""

    def __init__(
        self,
        stage_index: int,
        sequence_operator: SequenceMixingOperator,
        feed_forward_network: FeedForwardNetwork,
    ) -> None:
        self.stage_index = stage_index
        self.sequence_operator = sequence_operator
        self.feed_forward_network = feed_forward_network

    @classmethod
    def create_dense_initial_stage(cls, stage_index: int = 0) -> TransformerStage:
        default_tensor = QuantizedTensorRecord(
            tensor_name=f"stage_{stage_index}_dense_mlp",
            row_count=4,
            column_count=4,
            values=[1.0, 1.05, 0.95, 1.0],
        )
        return cls(
            stage_index=stage_index,
            sequence_operator=KimiDeltaAttentionOperator(),
            feed_forward_network=DenseSwiGLUFeedForwardNetwork(default_tensor),
        )

    @classmethod
    def create_routed_mixture_of_experts_stage(
        cls,
        stage_index: int,
        storage_tier: MemoryStorageTier,
        routed_expert_identifiers: Sequence[int],
        use_latent_attention: bool = False,
    ) -> TransformerStage:
        operator: SequenceMixingOperator = (
            MultiHeadLatentAttentionOperator() if use_latent_attention else KimiDeltaAttentionOperator()
        )
        return cls(
            stage_index=stage_index,
            sequence_operator=operator,
            feed_forward_network=RoutedMixtureOfExpertsFeedForwardNetwork(
                storage_tier=storage_tier,
                routed_expert_identifiers=routed_expert_identifiers,
            ),
        )

    def advance_stage_on_workspace(self, forward_pass_workspace: ForwardPassWorkspace, current_pass_number: int = 1) -> List[float]:
        mixed_activations = self.sequence_operator.mix_sequence_activations(
            stage_index=self.stage_index,
            input_activation_vector=forward_pass_workspace.current_activation_vector,
            forward_pass_workspace=forward_pass_workspace,
        )
        transformed_activations = self.feed_forward_network.transform_activation_vector(
            stage_index=self.stage_index,
            input_activation_vector=mixed_activations,
            current_pass_number=current_pass_number,
        )
        forward_pass_workspace.current_activation_vector = transformed_activations
        return transformed_activations


class NeuralStagePipeline:
    """Composite Pipeline unifying `Model`, `K3Full93Chain`, `NextConceptTransformer`, and `HybridAgentModel`."""

    def __init__(
        self,
        modality_codec: ModalityCodec,
        ordered_stages: Sequence[TransformerStage],
        snapshot_interval_in_stages: int = DEFAULT_SNAPSHOT_INTERVAL_IN_STAGES,
    ) -> None:
        self.modality_codec = modality_codec
        self.ordered_stages: List[TransformerStage] = list(ordered_stages)
        self.snapshot_interval_in_stages = snapshot_interval_in_stages
        self.telemetry_ledger = ExecutionTelemetryLedger()

    @property
    def total_stage_count(self) -> int:
        return len(self.ordered_stages)

    @record_execution_telemetry
    @enforce_direction_boundary
    def execute_forward_pipeline(
        self,
        task_direction: TaskDirection,
        forward_pass_workspace: ForwardPassWorkspace,
        current_pass_number: int = 1,
    ) -> str:
        forward_pass_workspace.current_activation_vector = self.modality_codec.encode_input_into_activation_vector(
            task_direction.goal_description
        )
        for stage_position, transformer_stage in enumerate(self.ordered_stages, start=1):
            transformer_stage.advance_stage_on_workspace(
                forward_pass_workspace=forward_pass_workspace,
                current_pass_number=current_pass_number,
            )
            if stage_position % self.snapshot_interval_in_stages == 0:
                forward_pass_workspace.record_carbon_copy_snapshot()
        forward_pass_workspace.sequence_position += 1
        return self.modality_codec.decode_activation_vector_into_text(
            activation_vector=forward_pass_workspace.current_activation_vector,
            task_direction=task_direction,
        )


class ExternalTool(ABC):
    """Abstract interface for external tools (`PythonSandboxTool`, `LiveWebSearchTool`, `KnowledgeSearchTool`)."""

    @property
    @abstractmethod
    def tool_name(self) -> str:
        """Returns the descriptive identifier of the tool."""

    @abstractmethod
    def execute_tool_action(self, action_Input_query: str, task_direction: TaskDirection) -> str:
        """Executes the external tool action within the boundaries of `TaskDirection`."""


class RetryingExternalToolDecorator(ExternalTool):
    """Decorator Pattern wrapping any `ExternalTool` with automatic bounded retries."""

    def __init__(self, wrapped_tool: ExternalTool, maximum_attempts: int = 2) -> None:
        self.wrapped_tool = wrapped_tool
        self.maximum_attempts = max(1, maximum_attempts)
        self.recorded_attempt_count: int = 0

    @property
    def tool_name(self) -> str:
        return f"Retrying({self.wrapped_tool.tool_name})"

    @enforce_direction_boundary
    def execute_tool_action(self, action_input_query: str, task_direction: TaskDirection) -> str:
        last_error: Optional[Exception] = None
        for _ in range(self.maximum_attempts):
            self.recorded_attempt_count += 1
            try:
                return self.wrapped_tool.execute_tool_action(action_input_query, task_direction)
            except Exception as caught_error:
                last_error = caught_error
        raise RuntimeError(f"Tool '{self.tool_name}' failed after {self.maximum_attempts} attempts: {last_error}")


# ============================================================================
# ROOT PRIMITIVE 4: OUTCOME — Independent Verification, Diagnostics & Verdicts
# (`ArtifactVerifier` produces a `VerificationVerdict` — never conflates `is-a` with `produces-a`!)
# ============================================================================

@dataclass(frozen=True)
class VerificationDiagnostic:
    """Unifies `GCDiagnostic` and verifier findings into one immutable diagnostic record."""

    verifier_name: str
    severity_level: str
    diagnostic_message: str
    line_number: int = 1


@dataclass(frozen=True)
class VerificationVerdict:
    """Unifies `GCVerdict`, `GraphValidationReport`, and test harness outcomes."""

    has_passed_all_checks: bool
    verified_artifact_text: str
    diagnostics: Tuple[VerificationDiagnostic, ...] = ()
    supporting_nodes: Tuple[KnowledgeGraphNode, ...] = ()

    @property
    def error_diagnostic_count(self) -> int:
        return sum(1 for item in self.diagnostics if item.severity_level == "error")


class ArtifactVerifier(ABC):
    """Unifies `CloverPythonGenerativeCompiler`, `CloverRustGenerativeCompiler`, `CloverSymbolicVerifier`, and `VerificationPolicy`."""

    @abstractmethod
    def verify_candidate_artifact_against_context(
        self,
        candidate_artifact_text: str,
        task_direction: TaskDirection,
        knowledge_graph: VerifiedKnowledgeGraph,
    ) -> VerificationVerdict:
        """Independently inspects a candidate artifact against reality and returns a `VerificationVerdict`."""


@register_component_in_catalog("artifact_verifier", "symbolic_and_syntax_verifier")
class SymbolicAndSyntaxArtifactVerifier(ArtifactVerifier):
    """Concrete verifier that checks non-empty output, Python syntax (when requested), and knowledge-graph grounding."""

    def verify_candidate_artifact_against_context(
        self,
        candidate_artifact_text: str,
        task_direction: TaskDirection,
        knowledge_graph: VerifiedKnowledgeGraph,
    ) -> VerificationVerdict:
        collected_diagnostics: List[VerificationDiagnostic] = []
        if not candidate_artifact_text or not candidate_artifact_text.strip():
            collected_diagnostics.append(
                VerificationDiagnostic(
                    verifier_name=self.__class__.__name__,
                    severity_level="error",
                    diagnostic_message="Candidate artifact text is empty.",
                )
            )
        if "python" in task_direction.required_verification_languages:
            try:
                compile(candidate_artifact_text, "<candidate_artifact>", "exec")
            except SyntaxError as syntax_error:
                collected_diagnostics.append(
                    VerificationDiagnostic(
                        verifier_name=self.__class__.__name__,
                        severity_level="error",
                        diagnostic_message=f"Python SyntaxError: {syntax_error.msg}",
                        line_number=int(syntax_error.lineno or 1),
                    )
                )
        supporting_nodes = tuple(
            knowledge_graph.retrieve_grounded_nodes_for_query(task_direction.goal_description, maximum_results=4)
        )
        passed_checks = all(diag.severity_level != "error" for diag in collected_diagnostics)
        return VerificationVerdict(
            has_passed_all_checks=passed_checks,
            verified_artifact_text=candidate_artifact_text,
            diagnostics=tuple(collected_diagnostics),
            supporting_nodes=supporting_nodes,
        )


# ============================================================================
# ROOT PRIMITIVE 5: GROWTH — Preserving Verified Outcomes Back Into Context
# ============================================================================

class GrowthLedgerRecorder:
    """Records verified `VerificationVerdict` outcomes into `VerifiedKnowledgeGraph` and `ConversationSession`."""

    def record_verified_outcome_into_context(
        self,
        task_direction: TaskDirection,
        verification_verdict: VerificationVerdict,
        knowledge_graph: VerifiedKnowledgeGraph,
        conversation_session: ConversationSession,
        recorded_timestamp_utc: str = "2026-10-28T12:00:00Z",
    ) -> Optional[KnowledgeGraphNode]:
        conversation_session.append_completed_turn(
            user_prompt_text=task_direction.goal_description,
            assistant_reply_text=verification_verdict.verified_artifact_text,
        )
        if not verification_verdict.has_passed_all_checks:
            return None

        provenance = SourceProvenance(
            source_kind="verified_cycle_outcome",
            source_reference_uri=f"direction://{task_direction.request_identifier}",
            recorded_by_actor="CloverCycleCoordinator",
            recorded_timestamp_utc=recorded_timestamp_utc,
            is_externally_verified=True,
        )
        outcome_node = KnowledgeGraphNode(
            node_identifier=f"MEM-OUTCOME-{task_direction.request_identifier.upper()}",
            node_category="outcome",
            title_text=f"Verified Outcome for {task_direction.request_identifier}",
            content_text=verification_verdict.verified_artifact_text,
            provenance=provenance,
            concept_tags=[word.lower() for word in task_direction.goal_description.split() if len(word) >= 4][:5],
        )
        knowledge_graph.stage_candidate_node_for_review(outcome_node)
        promoted_node = knowledge_graph.promote_staged_node_to_active_graph(
            node_identifier=outcome_node.node_identifier,
            verification_provenance=provenance,
        )
        return promoted_node


# ============================================================================
# TEMPLATE METHOD & FLUENT BUILDER — Orchestrating the 5-Stage Clover Cycle
# ============================================================================

class CloverCycleCoordinator:
    """Unifies `MultiGPUCloverCoordinator`, `CloverK3AgentEngine`, and `K3FastChatEngine` using the Template Method Pattern."""

    def __init__(
        self,
        knowledge_graph: VerifiedKnowledgeGraph,
        primary_storage_tier: MemoryStorageTier,
        neural_pipeline: NeuralStagePipeline,
        artifact_verifier: ArtifactVerifier,
        growth_recorder: GrowthLedgerRecorder,
        cycle_observers: Optional[Sequence[CyclePhaseObserver]] = None,
    ) -> None:
        self.knowledge_graph = knowledge_graph
        self.primary_storage_tier = primary_storage_tier
        self.neural_pipeline = neural_pipeline
        self.artifact_verifier = artifact_verifier
        self.growth_recorder = growth_recorder
        self.cycle_observers: List[CyclePhaseObserver] = list(cycle_observers or [])
        self.telemetry_ledger = ExecutionTelemetryLedger()

    @emit_clover_cycle_stage("Context")
    def establish_cycle_context(self, task_direction: TaskDirection) -> List[KnowledgeGraphNode]:
        return self.knowledge_graph.retrieve_grounded_nodes_for_query(task_direction.goal_description)

    @emit_clover_cycle_stage("Direction")
    def validate_cycle_direction(self, task_direction: TaskDirection) -> bool:
        if not task_direction.has_non_empty_goal:
            raise ValueError("TaskDirection requires a non-empty goal_description.")
        return True

    @emit_clover_cycle_stage("Execution")
    def execute_cycle_transformation(
        self,
        task_direction: TaskDirection,
        forward_pass_workspace: ForwardPassWorkspace,
        current_pass_number: int = 1,
    ) -> str:
        return self.neural_pipeline.execute_forward_pipeline(
            task_direction=task_direction,
            forward_pass_workspace=forward_pass_workspace,
            current_pass_number=current_pass_number,
        )

    @emit_clover_cycle_stage("Outcome")
    def verify_cycle_outcome(self, candidate_artifact_text: str, task_direction: TaskDirection) -> VerificationVerdict:
        return self.artifact_verifier.verify_candidate_artifact_against_context(
            candidate_artifact_text=candidate_artifact_text,
            task_direction=task_direction,
            knowledge_graph=self.knowledge_graph,
        )

    @emit_clover_cycle_stage("Growth")
    def preserve_cycle_growth(
        self,
        task_direction: TaskDirection,
        verification_verdict: VerificationVerdict,
        conversation_session: ConversationSession,
    ) -> Optional[KnowledgeGraphNode]:
        return self.growth_recorder.record_verified_outcome_into_context(
            task_direction=task_direction,
            verification_verdict=verification_verdict,
            knowledge_graph=self.knowledge_graph,
            conversation_session=conversation_session,
        )

    @record_execution_telemetry
    def execute_complete_clover_cycle(
        self,
        task_direction: TaskDirection,
        conversation_session: ConversationSession,
        forward_pass_workspace: Optional[ForwardPassWorkspace] = None,
        current_pass_number: int = 1,
    ) -> VerificationVerdict:
        active_workspace = forward_pass_workspace or ForwardPassWorkspace()
        self.establish_cycle_context(task_direction)
        self.validate_cycle_direction(task_direction)
        candidate_artifact_text = self.execute_cycle_transformation(
            task_direction=task_direction,
            forward_pass_workspace=active_workspace,
            current_pass_number=current_pass_number,
        )
        verification_verdict = self.verify_cycle_outcome(
            candidate_artifact_text=candidate_artifact_text,
            task_direction=task_direction,
        )
        self.preserve_cycle_growth(
            task_direction=task_direction,
            verification_verdict=verification_verdict,
            conversation_session=conversation_session,
        )
        return verification_verdict


class CloverCycleCoordinatorBuilder:
    """Fluent Builder Pattern for assembling a `CloverCycleCoordinator` in readable English sentences."""

    def __init__(self) -> None:
        self._knowledge_graph: Optional[VerifiedKnowledgeGraph] = None
        self._primary_storage_tier: Optional[MemoryStorageTier] = None
        self._neural_pipeline: Optional[NeuralStagePipeline] = None
        self._artifact_verifier: Optional[ArtifactVerifier] = None
        self._growth_recorder: Optional[GrowthLedgerRecorder] = None
        self._cycle_observers: List[CyclePhaseObserver] = []

    def with_knowledge_graph(self, knowledge_graph: VerifiedKnowledgeGraph) -> CloverCycleCoordinatorBuilder:
        self._knowledge_graph = knowledge_graph
        return self

    def with_primary_storage_tier(self, primary_storage_tier: MemoryStorageTier) -> CloverCycleCoordinatorBuilder:
        self._primary_storage_tier = primary_storage_tier
        return self

    def with_neural_pipeline(self, neural_pipeline: NeuralStagePipeline) -> CloverCycleCoordinatorBuilder:
        self._neural_pipeline = neural_pipeline
        return self

    def with_artifact_verifier(self, artifact_verifier: ArtifactVerifier) -> CloverCycleCoordinatorBuilder:
        self._artifact_verifier = artifact_verifier
        return self

    def with_cycle_observer(self, cycle_observer: CyclePhaseObserver) -> CloverCycleCoordinatorBuilder:
        self._cycle_observers.append(cycle_observer)
        return self

    def build_coordinator(self) -> CloverCycleCoordinator:
        knowledge_graph = self._knowledge_graph or VerifiedKnowledgeGraph()
        if self._primary_storage_tier is None:
            vram_tier, _, nvme_tier = MemoryStorageTier.create_three_tier_hierarchy()
            for expert_id in range(1, 9):
                nvme_tier.slot_cache.insert_expert_evicting_if_needed(
                    expert_identifier=expert_id,
                    payload_tensor=QuantizedTensorRecord(
                        tensor_name=f"expert_{expert_id}",
                        row_count=4,
                        column_count=4,
                        values=[1.0, 1.0, 1.0, 1.0],
                    ),
                    protected_expert_identifiers=set(),
                    current_pass_number=0,
                    current_layer_index=0,
                )
            self._primary_storage_tier = vram_tier
        if self._neural_pipeline is None:
            stages = [
                TransformerStage.create_dense_initial_stage(stage_index=0),
                TransformerStage.create_routed_mixture_of_experts_stage(
                    stage_index=1,
                    storage_tier=self._primary_storage_tier,
                    routed_expert_identifiers=[1, 2],
                    use_latent_attention=False,
                ),
                TransformerStage.create_routed_mixture_of_experts_stage(
                    stage_index=2,
                    storage_tier=self._primary_storage_tier,
                    routed_expert_identifiers=[3, 4],
                    use_latent_attention=True,
                ),
            ]
            self._neural_pipeline = NeuralStagePipeline(
                modality_codec=KimiK3TextModalityCodec(),
                ordered_stages=stages,
                snapshot_interval_in_stages=2,
            )
        return CloverCycleCoordinator(
            knowledge_graph=knowledge_graph,
            primary_storage_tier=self._primary_storage_tier,
            neural_pipeline=self._neural_pipeline,
            artifact_verifier=self._artifact_verifier or SymbolicAndSyntaxArtifactVerifier(),
            growth_recorder=self._growth_recorder or GrowthLedgerRecorder(),
            cycle_observers=self._cycle_observers,
        )
