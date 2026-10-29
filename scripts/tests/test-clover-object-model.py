# coding: clover-hyphen
"""Hyphenated complete-English unit test suite (`test-clover-object-model.py`) exercising `clover-object-model.py`."""

from pathlib import Path
import sys
import unittest

REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "clover-ai-system" / "src" / "tools" / "hybrid-ncm-k3"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

import clover_hyphen_runtime  # noqa: F401
from clover_object_model import (
    class-clover-cycle-coordinator-builder,
    class-component-catalog-registry,
    class-conversation-session,
    class-external-tool,
    class-forward-pass-workspace,
    class-in-memory-cycle-phase-log-observer,
    class-knowledge-graph-node,
    class-memory-storage-tier,
    class-quantized-tensor-record,
    class-retrying-external-tool-decorator,
    class-source-provenance,
    class-task-direction,
    class-verified-knowledge-graph,
)


class class-flaky-calculation-tool(class-external-tool):
    """Test double that fails on its first call and succeeds on its second call."""

    def __init__(self) -> None:
        self.variable-call-counter = 0

    @property
    def variable-tool-name(self) -> str:
        return "class-flaky-calculation-tool"

    def function-execute-tool-action(
        self,
        variable-action-input-query: str,
        variable-task-direction: class-task-direction,
    ) -> str:
        self.variable-call-counter += 1
        if self.variable-call-counter == 1:
            raise RuntimeError("Transient calculation failure on first attempt.")
        return f"Computed result for: {variable-action-input-query}"


class TestCloverObjectModel(unittest.TestCase):
    def test_01_registry_decorator_and_catalog_factory(self) -> None:
        variable-eviction-keys = class-component-catalog-registry.function-list-registered-keys-in-category(
            "cache-eviction-strategy"
        )
        self.assertIn("specmd-least-stale", variable-eviction-keys)
        variable-operator-keys = class-component-catalog-registry.function-list-registered-keys-in-category(
            "sequence-operator"
        )
        self.assertIn("kimi-delta-attention", variable-operator-keys)
        self.assertIn("multi-head-latent-attention", variable-operator-keys)
        variable-strategy = class-component-catalog-registry.function-create-component-instance(
            "cache-eviction-strategy",
            "specmd-least-stale",
        )
        self.assertIsNotNone(variable-strategy)

    def test_02_three_tier_storage_chain_of_responsibility_and_r4_clean_drop(self) -> None:
        variable-vram-tier, variable-host-ram-tier, variable-nvme-tier = (
            class-memory-storage-tier.function-create-three-tier-hierarchy(
                variable-vram-capacity-in-slots=2,
                variable-host-ram-capacity-in-slots=4,
                variable-nvme-capacity-in-slots=8,
            )
        )
        for variable-expert-id in range(1, 6):
            variable-nvme-tier.variable-slot-cache.function-insert-expert-evicting-if-needed(
                variable-expert-identifier=variable-expert-id,
                variable-payload-tensor=class-quantized-tensor-record(
                    f"expert-{variable-expert-id}", 4, 4, [1.0, 1.0, 1.0, 1.0]
                ),
                variable-protected-expert-identifiers=set(),
                variable-current-pass-number=0,
                variable-current-layer-index=0,
            )

        variable-vram-tier.function-ensure-expert-tensor-resident(
            1, variable-protected-expert-identifiers={1}, variable-current-pass-number=1, variable-current-layer-index=1
        )
        variable-vram-tier.function-ensure-expert-tensor-resident(
            2, variable-protected-expert-identifiers={2}, variable-current-pass-number=1, variable-current-layer-index=2
        )
        self.assertEqual(variable-vram-tier.variable-slot-cache.variable-occupied-slot-count, 2)

        variable-vram-tier.function-ensure-expert-tensor-resident(
            1, variable-protected-expert-identifiers={1}, variable-current-pass-number=1, variable-current-layer-index=3
        )
        self.assertEqual(variable-vram-tier.variable-movement-statistics.variable-local-tier-hits-count, 1)

        variable-vram-tier.function-ensure-expert-tensor-resident(
            3, variable-protected-expert-identifiers={3}, variable-current-pass-number=1, variable-current-layer-index=3
        )
        self.assertEqual(variable-vram-tier.variable-movement-statistics.variable-clean-eviction-drops-r4-count, 1)
        self.assertEqual(variable-vram-tier.variable-movement-statistics.variable-dirty-writebacks-r2-count, 0)
        self.assertTrue(variable-host-ram-tier.function-is-expert-currently-resident(2))

    def test_03_memento_checkpoints_and_256k_session_compaction(self) -> None:
        variable-session = class-conversation-session(
            variable-session-identifier="session-alpha",
            variable-maximum-token-budget=20,
        )
        variable-session.function-append-completed-turn(
            "Explain Clover Context",
            "Context reads the running system first.",
            variable-estimated-token-count=10,
        )
        variable-checkpoint = variable-session.function-capture-turn-checkpoint()
        variable-session.function-append-completed-turn(
            "Unwanted speculative turn",
            "Temporary reply",
            variable-estimated-token-count=8,
        )
        self.assertEqual(len(variable-session.variable-conversation-turns), 2)

        variable-session.function-restore-turn-checkpoint(variable-checkpoint)
        self.assertEqual(len(variable-session.variable-conversation-turns), 1)
        self.assertEqual(variable-session.variable-current-token-count, 10)

        variable-session.function-append-completed-turn(
            "Second turn crossing budget",
            "Verified second turn outcome.",
            variable-estimated-token-count=15,
        )
        self.assertFalse(variable-session.flag-has-exceeded-maximum-token-budget)
        self.assertGreaterEqual(variable-session.variable-compaction-operations-count, 1)

        variable-workspace = class-forward-pass-workspace([1.0, 2.0, 3.0, 4.0])
        variable-ws-checkpoint = variable-workspace.function-capture-checkpoint-memento()
        variable-workspace.variable-current-activation-vector = [9.0, 9.0, 9.0, 9.0]
        variable-workspace.variable-sequence-position = 5
        variable-workspace.function-restore-from-checkpoint-memento(variable-ws-checkpoint)
        self.assertEqual(variable-workspace.variable-sequence-position, 0)
        self.assertEqual(variable-workspace.variable-current-activation-vector, [1.0, 2.0, 3.0, 4.0])

    def test_04_retrying_tool_decorator_and_direction_boundary_guard(self) -> None:
        variable-direction = class-task-direction(
            variable-request-identifier="req-001",
            variable-goal-description="Calculate orbital period",
            variable-protected-file-paths={"AGENTS.md", "src/locked"},
        )
        self.assertFalse(variable-direction.function-allows-modification-of-path("AGENTS.md"))
        self.assertFalse(variable-direction.function-allows-modification-of-path("src/locked/kernel.c"))
        self.assertTrue(variable-direction.function-allows-modification-of-path("src/open/module.py"))

        variable-retrying-tool = class-retrying-external-tool-decorator(
            class-flaky-calculation-tool(),
            variable-maximum-attempts=3,
        )
        variable-output = variable-retrying-tool.function-execute-tool-action(
            "25 * 4",
            variable-task-direction=variable-direction,
        )
        self.assertIn("Computed result for: 25 * 4", variable-output)
        self.assertEqual(variable-retrying-tool.variable-recorded-attempt-count, 2)

    def test_05_fluent_builder_observer_and_complete_clover_cycle(self) -> None:
        variable-graph = class-verified-knowledge-graph()
        variable-initial-provenance = class-source-provenance(
            variable-source-kind="canonical-specification",
            variable-source-reference-uri="AGENTS.md#the-cycle",
            variable-recorded-by-actor="system-seed",
            variable-recorded-timestamp-utc="2026-10-28T00:00:00Z",
        )
        variable-seed-node = class-knowledge-graph-node(
            variable-node-identifier="MEM-CONCEPT-CLOVER-CYCLE-001",
            variable-node-category="concept",
            variable-title-text="The Five-Stage Clover Cycle",
            variable-content-text="Context, Direction, Execution, Outcome, and Growth govern every engineering cycle.",
            variable-provenance=variable-initial-provenance,
            variable-concept-tags=["clover", "cycle", "context", "growth"],
        )
        variable-graph.function-stage-candidate-node-for-review(variable-seed-node)
        variable-graph.function-promote-staged-node-to-active-graph(
            variable-seed-node.variable-node-identifier,
            variable-initial-provenance,
        )

        variable-phase-observer = class-in-memory-cycle-phase-log-observer()
        variable-coordinator = (
            class-clover-cycle-coordinator-builder()
            .function-with-knowledge-graph(variable-graph)
            .function-with-cycle-observer(variable-phase-observer)
            .function-build-coordinator()
        )

        variable-direction = class-task-direction(
            variable-request-identifier="cycle-test-01",
            variable-goal-description="Explain the Clover cycle stages and context grounding",
        )
        variable-session = class-conversation-session(variable-session-identifier="session-cycle-01")
        variable-verdict = variable-coordinator.function-execute-complete-clover-cycle(
            variable-task-direction=variable-direction,
            variable-conversation-session=variable-session,
        )

        self.assertTrue(variable-verdict.flag-has-passed-all-checks)
        self.assertEqual(len(variable-verdict.variable-supporting-nodes), 1)
        self.assertEqual(
            [variable-stage for variable-stage, _ in variable-phase-observer.variable-completed-stages],
            ["Context", "Direction", "Execution", "Outcome", "Growth"],
        )
        self.assertIn("MEM-OUTCOME-CYCLE-TEST-01", variable-graph.variable-nodes-by-identifier)
        self.assertGreaterEqual(variable-coordinator.variable-telemetry-ledger.variable-total-recorded-invocations, 1)

    def test_06_switchable_foundation_model_backends_across_kimi_nemotron_and_lfm3(self) -> None:
        """Verifies hot-switching Layer 1 (the Foundation Model Backbone) across Kimi K3, Nemotron 3.5 Super VL, and LFM3-1.7B."""
        variable-coordinator = (
            class-clover-cycle-coordinator-builder()
            .function-with-foundation-model-key("kimi-k3")
            .function-build-coordinator()
        )
        self.assertEqual(variable-coordinator.variable-active-model-key, "kimi-k3")

        # Switch Layer 1 to NVIDIA Nemotron 3.5 Super VL (`120B-A12B`, 88 layers, Mamba-2 + Latent ReLU^2 MoE + NoPE GQA)
        variable-nemo-backend = variable-coordinator.function-switch-foundation-model-backend(
            "nemotron-3.5-super-vl"
        )
        self.assertEqual(variable-coordinator.variable-active-model-key, "nemotron-3.5-super-vl")
        self.assertEqual(variable-nemo-backend.variable-total-layer-count, 88)
        self.assertEqual(variable-nemo-backend.variable-hidden-dimension, 4096)

        variable-session = class-conversation-session(variable-session-identifier="session-switch-models")
        variable-nemo-direction = class-task-direction(
            variable-request-identifier="nemo-turn-01",
            variable-goal-description="Verify Nemotron 3.5 Super VL Layer-1 switching in the 7-layer Clover stack.",
        )
        variable-nemo-decoded = variable-coordinator.function-execute-cycle-transformation(
            variable-task-direction=variable-nemo-direction,
            variable-forward-pass-workspace=class-forward-pass-workspace(),
        )
        self.assertIn("[Nemotron-3.5-Super-VL]", variable-nemo-decoded)
        variable-nemo-verdict = variable-coordinator.function-execute-complete-clover-cycle(
            variable-task-direction=variable-nemo-direction,
            variable-conversation-session=variable-session,
        )
        self.assertTrue(variable-nemo-verdict.flag-has-passed-all-checks)

        # Switch Layer 1 to Liquid AI LFM3-1.7B (`1.73B`, 24 layers, ShortConv + Sliding/Full GQA)
        variable-lfm3-backend = variable-coordinator.function-switch-foundation-model-backend(
            "lfm3-1.7b"
        )
        self.assertEqual(variable-coordinator.variable-active-model-key, "lfm3-1.7b")
        self.assertEqual(variable-lfm3-backend.variable-total-layer-count, 24)
        self.assertEqual(variable-lfm3-backend.variable-hidden-dimension, 2048)

        variable-lfm3-direction = class-task-direction(
            variable-request-identifier="lfm3-turn-01",
            variable-goal-description="Verify LFM3-1.7B Layer-1 switching in the 7-layer Clover stack.",
        )
        variable-lfm3-decoded = variable-coordinator.function-execute-cycle-transformation(
            variable-task-direction=variable-lfm3-direction,
            variable-forward-pass-workspace=class-forward-pass-workspace(),
        )
        self.assertIn("[LFM3-1.7B]", variable-lfm3-decoded)
        variable-lfm3-verdict = variable-coordinator.function-execute-complete-clover-cycle(
            variable-task-direction=variable-lfm3-direction,
            variable-conversation-session=variable-session,
        )
        self.assertTrue(variable-lfm3-verdict.flag-has-passed-all-checks)


