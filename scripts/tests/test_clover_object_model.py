"""Unit tests for the complete-English Clover Object-Oriented Model (`clover_object_model.py`)."""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "clover-kimi-k3" / "src" / "tools" / "hybrid-ncm-k3"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

from clover_object_model import (
    CloverCycleCoordinatorBuilder,
    ComponentCatalogRegistry,
    ConversationSession,
    ExternalTool,
    ForwardPassWorkspace,
    InMemoryCyclePhaseLogObserver,
    KnowledgeGraphEdge,
    KnowledgeGraphNode,
    MemoryStorageTier,
    QuantizedTensorRecord,
    RetryingExternalToolDecorator,
    SourceProvenance,
    TaskDirection,
    VerifiedKnowledgeGraph,
)


class FlakyCalculationTool(ExternalTool):
    """Test double that fails on its first call and succeeds on its second call."""

    def __init__(self) -> None:
        self.call_counter = 0

    @property
    def tool_name(self) -> str:
        return "FlakyCalculationTool"

    def execute_tool_action(self, action_input_query: str, task_direction: TaskDirection) -> str:
        self.call_counter += 1
        if self.call_counter == 1:
            raise RuntimeError("Transient calculation failure on first attempt.")
        return f"Computed result for: {action_input_query}"


class TestCloverObjectModel(unittest.TestCase):
    def test_01_registry_decorator_and_catalog_factory(self) -> None:
        eviction_keys = ComponentCatalogRegistry.list_registered_keys_in_category("cache_eviction_strategy")
        self.assertIn("specmd_least_stale", eviction_keys)
        operator_keys = ComponentCatalogRegistry.list_registered_keys_in_category("sequence_operator")
        self.assertIn("kimi_delta_attention", operator_keys)
        self.assertIn("multi_head_latent_attention", operator_keys)
        strategy = ComponentCatalogRegistry.create_component_instance("cache_eviction_strategy", "specmd_least_stale")
        self.assertEqual(strategy.__class__.__name__, "LeastStaleCacheEvictionStrategy")

    def test_02_three_tier_storage_chain_of_responsibility_and_r4_clean_drop(self) -> None:
        vram_tier, host_ram_tier, nvme_tier = MemoryStorageTier.create_three_tier_hierarchy(
            vram_capacity_in_slots=2,
            host_ram_capacity_in_slots=4,
            nvme_capacity_in_slots=8,
        )
        for expert_id in range(1, 6):
            nvme_tier.slot_cache.insert_expert_evicting_if_needed(
                expert_identifier=expert_id,
                payload_tensor=QuantizedTensorRecord(f"expert_{expert_id}", 4, 4, [1.0, 1.0, 1.0, 1.0]),
                protected_expert_identifiers=set(),
                current_pass_number=0,
                current_layer_index=0,
            )

        vram_tier.ensure_expert_tensor_resident(1, protected_expert_identifiers={1}, current_pass_number=1, current_layer_index=1)
        vram_tier.ensure_expert_tensor_resident(2, protected_expert_identifiers={2}, current_pass_number=1, current_layer_index=2)
        self.assertEqual(vram_tier.slot_cache.occupied_slot_count, 2)

        # Hit on expert 1 in VRAM
        vram_tier.ensure_expert_tensor_resident(1, protected_expert_identifiers={1}, current_pass_number=1, current_layer_index=3)
        self.assertEqual(vram_tier.movement_statistics.local_tier_hits_count, 1)

        # Fetching expert 3 evicts expert 2 (whose layer 2 < current layer 3), and since expert 2 is still in Host RAM,
        # it triggers an R4 Clean Drop with zero dirty writeback!
        vram_tier.ensure_expert_tensor_resident(3, protected_expert_identifiers={3}, current_pass_number=1, current_layer_index=3)
        self.assertEqual(vram_tier.movement_statistics.clean_eviction_drops_r4_count, 1)
        self.assertEqual(vram_tier.movement_statistics.dirty_writebacks_r2_count, 0)
        self.assertTrue(host_ram_tier.is_expert_currently_resident(2))

    def test_03_memento_checkpoints_and_256k_session_compaction(self) -> None:
        session = ConversationSession(session_identifier="session_alpha", maximum_token_budget=20)
        session.append_completed_turn("Explain Clover Context", "Context reads the running system first.", estimated_token_count=10)
        checkpoint = session.capture_turn_checkpoint()
        session.append_completed_turn("Unwanted speculative turn", "Temporary reply", estimated_token_count=8)
        self.assertEqual(len(session.conversation_turns), 2)

        session.restore_turn_checkpoint(checkpoint)
        self.assertEqual(len(session.conversation_turns), 1)
        self.assertEqual(session.current_token_count, 10)

        # Crossing maximum_token_budget triggers automatic simple-sentence compaction
        session.append_completed_turn("Second turn crossing budget", "Verified second turn outcome.", estimated_token_count=15)
        self.assertFalse(session.has_exceeded_maximum_token_budget)
        self.assertGreaterEqual(session.compaction_operations_count, 1)
        self.assertEqual(len(session.compacted_summary_sentences), 1)

        workspace = ForwardPassWorkspace([1.0, 2.0, 3.0, 4.0])
        ws_checkpoint = workspace.capture_checkpoint_memento()
        workspace.current_activation_vector = [9.0, 9.0, 9.0, 9.0]
        workspace.sequence_position = 5
        workspace.restore_from_checkpoint_memento(ws_checkpoint)
        self.assertEqual(workspace.sequence_position, 0)
        self.assertEqual(workspace.current_activation_vector, [1.0, 2.0, 3.0, 4.0])

    def test_04_retrying_tool_decorator_and_direction_boundary_guard(self) -> None:
        direction = TaskDirection(
            request_identifier="req_001",
            goal_description="Calculate orbital period",
            protected_file_paths={"AGENTS.md", "src/locked"},
        )
        self.assertFalse(direction.allows_modification_of_path("AGENTS.md"))
        self.assertFalse(direction.allows_modification_of_path("src/locked/kernel.c"))
        self.assertTrue(direction.allows_modification_of_path("src/open/module.py"))

        retrying_tool = RetryingExternalToolDecorator(FlakyCalculationTool(), maximum_attempts=3)
        output = retrying_tool.execute_tool_action("25 * 4", task_direction=direction)
        self.assertIn("Computed result for: 25 * 4", output)
        self.assertEqual(retrying_tool.recorded_attempt_count, 2)

    def test_05_fluent_builder_observer_and_complete_clover_cycle(self) -> None:
        graph = VerifiedKnowledgeGraph()
        initial_provenance = SourceProvenance(
            source_kind="canonical_specification",
            source_reference_uri="AGENTS.md#the-cycle",
            recorded_by_actor="system_seed",
            recorded_timestamp_utc="2026-10-28T00:00:00Z",
        )
        seed_node = KnowledgeGraphNode(
            node_identifier="MEM-CONCEPT-CLOVER-CYCLE-001",
            node_category="concept",
            title_text="The Five-Stage Clover Cycle",
            content_text="Context, Direction, Execution, Outcome, and Growth govern every engineering cycle.",
            provenance=initial_provenance,
            concept_tags=["clover", "cycle", "context", "growth"],
        )
        graph.stage_candidate_node_for_review(seed_node)
        graph.promote_staged_node_to_active_graph(seed_node.node_identifier, initial_provenance)

        phase_observer = InMemoryCyclePhaseLogObserver()
        coordinator = (
            CloverCycleCoordinatorBuilder()
            .with_knowledge_graph(graph)
            .with_cycle_observer(phase_observer)
            .build_coordinator()
        )

        direction = TaskDirection(
            request_identifier="cycle_test_01",
            goal_description="Explain the Clover cycle stages and context grounding",
        )
        session = ConversationSession(session_identifier="session_cycle_01")
        verdict = coordinator.execute_complete_clover_cycle(
            task_direction=direction,
            conversation_session=session,
        )

        self.assertTrue(verdict.has_passed_all_checks)
        self.assertEqual(len(verdict.supporting_nodes), 1)
        self.assertEqual(
            [stage for stage, _ in phase_observer.completed_stages],
            ["Context", "Direction", "Execution", "Outcome", "Growth"],
        )
        self.assertIn("MEM-OUTCOME-CYCLE_TEST_01", graph.nodes_by_identifier)
        self.assertGreaterEqual(coordinator.telemetry_ledger.total_recorded_invocations, 1)


if __name__ == "__main__":
    unittest.main()
