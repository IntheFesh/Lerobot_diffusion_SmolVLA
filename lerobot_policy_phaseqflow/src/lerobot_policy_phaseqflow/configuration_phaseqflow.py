"""
configuration_phaseqflow.py
===========================

This module defines the configuration class for the PhaseQFlow policy.  The
configuration captures parameters controlling the phase discretization,
quality weighting, flow matching, and other aspects of the policy.
"""

from dataclasses import dataclass, field
from typing import Optional

@dataclass
class PhaseQFlowConfig:
    """Configuration for the PhaseQFlow policy.

    Parameters
    ----------
    num_phases : int
        Number of discrete phases used to partition each episode.  A value of
        4 corresponds to dividing the trajectory into four equal parts.
    use_quality_weight : bool
        If ``True``, enables quality‑weighted imitation loss.  Quality
        weights are derived from action smoothness.
    use_flow_matching : bool
        If ``True``, uses a flow matching objective instead of
        traditional diffusion denoising.  Flow matching typically
        requires fewer sampling steps at inference time.
    phase_embedding_dim : int
        Dimensionality of the phase embedding vector injected into the
        policy network.
    quality_weight_min : float
        Minimum per‑sample weight applied to the loss.  Values in
        ``[0.0, 1.0]``.
    quality_weight_max : float
        Maximum per‑sample weight applied to the loss.  Values in
        ``[0.0, 1.0]``.
    """

    num_phases: int = 4
    use_quality_weight: bool = True
    use_flow_matching: bool = False
    phase_embedding_dim: int = 32
    quality_weight_min: float = 0.5
    quality_weight_max: float = 1.0

    # Additional attributes may be added here as new features are
    # incorporated.  When integrating with LeRobot, these fields can be
    # mapped directly to CLI arguments via Hydra or argparse.

    # Placeholders for future configuration options (kept optional so
    # that instantiation does not require them).  Users can extend
    # this dataclass as needed.
    observation_horizon: Optional[int] = None
    action_horizon: Optional[int] = None
