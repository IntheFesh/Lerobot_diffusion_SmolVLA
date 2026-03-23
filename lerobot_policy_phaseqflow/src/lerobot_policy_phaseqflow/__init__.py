"""
lerobot_policy_phaseqflow
========================

This package implements PhaseQFlow, a phase‑aware and quality‑weighted
generative policy for LeRobot.  It exposes a custom configuration class,
a model class, and processors for phase and quality handling.  When
installed, the policy is registered as a LeRobot plug‑in under the
`lerobot.policies` entry point group.
"""

from .configuration_phaseqflow import PhaseQFlowConfig
from .modeling_phaseqflow import PhaseQFlowPolicy
from .processor_phaseqflow import PhaseQFlowProcessor

__all__ = [
    "PhaseQFlowConfig",
    "PhaseQFlowPolicy",
    "PhaseQFlowProcessor",
]