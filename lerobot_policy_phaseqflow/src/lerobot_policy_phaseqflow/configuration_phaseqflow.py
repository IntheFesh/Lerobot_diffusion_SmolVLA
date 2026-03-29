"""
configuration_phaseqflow.py
===========================

Configuration for the upgraded PhaseQFlow policy.

This version shifts from temporal phases to latent skill phases, supports
value-guided weighting, and enables latent-space flow matching.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class PhaseQFlowConfig:
    """Configuration for PhaseQFlow.

    The default settings preserve backward compatibility while enabling the
    new components when toggled on.
    """

    # Legacy fields (kept for compatibility with existing checkpoints/configs)
    num_phases: int = 4
    use_quality_weight: bool = True
    use_flow_matching: bool = False
    phase_embedding_dim: int = 32
    quality_weight_min: float = 0.5
    quality_weight_max: float = 1.0

    # 1) Latent skill phase
    num_skills: int = 16
    use_vq_phase: bool = True
    skill_embedding_dim: int = 32
    gumbel_temperature: float = 1.0

    # 2) Value-guided weighting
    use_value_guided_weight: bool = True
    value_weight_beta: float = 2.0
    critic_hidden_dim: int = 256

    # 3) Latent flow matching
    latent_dim: int = 32
    use_latent_flow: bool = True

    # 4) Backbone modernization
    backbone_type: str = "dit"
    dit_hidden_dim: int = 256
    dit_num_layers: int = 4
    dit_num_heads: int = 8

    # Processor/runtime options
    action_buffer_maxlen: int = 128

    observation_horizon: Optional[int] = None
    action_horizon: Optional[int] = None
