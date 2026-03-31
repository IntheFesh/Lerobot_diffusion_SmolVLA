"""
configuration_phaseqflow.py
===========================

Configuration object for the upgraded (skill/value/latent-flow) PhaseQFlow
policy.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseQFlowConfig:
    """Configuration for the PhaseQFlow policy.

    The defaults are intentionally lightweight so the model can be trained on
    commodity hardware.
    """

    # Legacy temporal phase settings (kept for backward compatibility)
    num_phases: int = 4
    phase_embedding_dim: int = 32

    # 1) Learned skill phase
    num_skills: int = 16
    use_vq_phase: bool = True
    skill_embedding_dim: int = 32
    vq_commitment_cost: float = 0.25

    # 2) Value-guided weighting
    use_quality_weight: bool = True
    use_value_guided_weight: bool = True
    value_weight_beta: float = 1.0
    quality_weight_min: float = 0.5
    quality_weight_max: float = 1.0

    # 3) Latent flow matching
    use_flow_matching: bool = False
    use_latent_flow: bool = True
    latent_dim: int = 32

    # Backbone/model size controls (lightweight defaults)
    backbone_type: str = "dit"
    dit_hidden_dim: int = 256
    dit_num_layers: int = 4
    dit_num_heads: int = 4

    # Runtime / shapes
    observation_horizon: Optional[int] = None
    action_horizon: Optional[int] = None
