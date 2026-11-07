"""Clover Reusable Predictive Representation Engine (`rpre`).

Unifies the four core JEPA architectures into a single modular engine with Clover's
Outcome -> Growth closed-loop learning cycle:
1. I-JEPA (`facebookresearch/ijepa`): Context-only sparse encoding, 2D multi-block semantic masking,
   and target-normalized latent token prediction.
2. Audio-JEPA (`LudovicTuncay/Audio-JEPA`): 2D time-frequency spectrogram tokenization, centered
   L2-normalized cosine/MSE loss (`norm_mse`), and random/time-frequency block masking.
3. EB-JEPA (`facebookresearch/eb_jepa`): Energy-based anti-collapse regularization (`HingeStdLoss`,
   `CovarianceLoss`, `TemporalSimilarityLoss`, `InverseDynamicsLoss`, `BCS` Epps-Pulley), parallel &
   autoregressive latent unrolling, and receding-horizon `CEMPlanner` / `MPPIPlanner`.
4. MJEPA (`facebookresearch/MJEPA`): Single shared Transformer encoder across modalities with modality
   type embeddings, 4-level hierarchical target extraction, and 3 intra-modal + 6 cross-modal
   predictive objectives (`v2v`, `a2a`, `av2av`, `a2v`, `v2a`, `av2a`, `av2v`, `a2av`, `v2av`).
5. Clover Outcome-Driven Growth: Closed-loop residual transition & cross-modal calibration updated
   from observed outcome discrepancies (`z_observed - z_predicted`) with variance-covariance and
   non-regression gating.
"""
from .engine import (
    ActionConditionedPredictor,
    AudioSpectrogramAdapter,
    CrossModalPredictor,
    LatentTokenPredictor,
    OutcomeResidualAdapter,
    ReusablePredictiveEngine,
    SharedHierarchicalEncoder,
    StateVectorAdapter,
    VisualPatchAdapter,
)
from .losses import (
    BCSLoss,
    CenteredNormMSELoss,
    CovarianceLoss,
    HingeStdLoss,
    InverseDynamicsLoss,
    InverseDynamicsMLP,
    MJEPA9PathLoss,
    TemporalSimilarityLoss,
    VCLoss,
    VC_IDM_Sim_Regularizer,
)
from .masks import (
    apply_masks,
    sample_audio_tf_masks,
    sample_multiblock_2d_masks,
    sample_random_masks,
)
from .planner import (
    CEMPlanner,
    CloverCycleAgent,
    MPPIPlanner,
    PlanningResult,
    ReprTargetDistObjective,
)

__all__ = [
    "ActionConditionedPredictor",
    "AudioSpectrogramAdapter",
    "BCSLoss",
    "CEMPlanner",
    "CenteredNormMSELoss",
    "CloverCycleAgent",
    "CovarianceLoss",
    "CrossModalPredictor",
    "HingeStdLoss",
    "InverseDynamicsLoss",
    "InverseDynamicsMLP",
    "LatentTokenPredictor",
    "MJEPA9PathLoss",
    "MPPIPlanner",
    "OutcomeResidualAdapter",
    "PlanningResult",
    "ReprTargetDistObjective",
    "ReusablePredictiveEngine",
    "SharedHierarchicalEncoder",
    "StateVectorAdapter",
    "TemporalSimilarityLoss",
    "VCLoss",
    "VC_IDM_Sim_Regularizer",
    "VisualPatchAdapter",
    "apply_masks",
    "sample_audio_tf_masks",
    "sample_multiblock_2d_masks",
    "sample_random_masks",
]
