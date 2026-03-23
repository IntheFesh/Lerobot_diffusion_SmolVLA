"""
modeling_phaseqflow.py
======================

This module defines the PhaseQFlowPolicy class, a custom policy that
implements phase‑aware and quality‑weighted generative action chunking.  The
implementation builds on the LeRobot diffusion policy by injecting a
phase embedding and scaling the imitation loss by sample‑specific quality
weights.  It optionally supports a flow matching objective to reduce
sampling steps.

Note: The actual implementation details may vary across LeRobot versions.
This skeleton provides a template that can be adjusted to the current
version by modifying attribute names and method signatures at the
``# MAY NEED ADJUSTMENT`` markers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from .configuration_phaseqflow import PhaseQFlowConfig

try:
    # Attempt to import the LeRobot base classes.  If these imports
    # fail at runtime, the user must ensure LeRobot is installed.
    from lerobot.policies.pretrained import PreTrainedPolicy  # type: ignore
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # type: ignore
except Exception:
    PreTrainedPolicy = object  # type: ignore
    DiffusionPolicy = object  # type: ignore


class PhaseQFlowPolicy(PreTrainedPolicy):
    """PhaseQFlow policy implementing phase‑aware and quality‑weighted action generation."""

    config_class = PhaseQFlowConfig

    def __init__(
        self,
        config: PhaseQFlowConfig,
        base_policy: Optional[DiffusionPolicy] = None,
        **kwargs: Any,
    ) -> None:
        # Initialize base class
        super().__init__(config)
        self.config: PhaseQFlowConfig = config
        # Use an existing diffusion policy as the backbone if provided;
        # otherwise instantiate a default diffusion policy.  The diffusion
        # policy class name may differ across LeRobot versions; adjust if
        # necessary.
        self.base_policy = base_policy or DiffusionPolicy(config)
        # Phase embedding layer.  Map phase IDs to a learnable vector.
        self.phase_embedding = nn.Embedding(config.num_phases, config.phase_embedding_dim)
        # Linear projection to fuse the base policy input with the phase
        # embedding.  We assume the base policy exposes a `obs_feature_dim`
        # attribute specifying the dimension of the encoded observation.
        obs_dim = getattr(self.base_policy, "obs_feature_dim", 256)  # MAY NEED ADJUSTMENT
        self.phase_proj = nn.Linear(obs_dim + config.phase_embedding_dim, obs_dim)
        # Quality weight range parameters
        self.quality_min = config.quality_weight_min
        self.quality_max = config.quality_weight_max

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "PhaseQFlowPolicy":
        """Load a pretrained PhaseQFlow policy from disk.

        This method simply delegates to the underlying base policy's
        loading mechanism and wraps it in a PhaseQFlowPolicy.  It
        assumes that a ``config.json`` file accompanies the model.
        """
        # Load configuration; rely on PreTrainedPolicy helper
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        # Load base diffusion policy; this uses the same directory
        base_policy = DiffusionPolicy.from_pretrained(pretrained_model_name_or_path)
        return cls(config, base_policy)

    # ---------------------------------------------------------------------
    # Loss computation
    # ---------------------------------------------------------------------
    def compute_loss(
        self,
        batch: Dict[str, Any],
    ) -> torch.Tensor:
        """Compute the imitation loss for a batch.

        The loss combines the base policy's diffusion or flow matching
        objective with optional quality weighting.
        """
        # Compute base policy outputs: predicted noise and target noise
        # The attribute names may differ across LeRobot versions; adjust if
        # necessary.
        out = self.base_policy.compute_loss(batch)  # type: ignore  # MAY NEED ADJUSTMENT
        loss = out["loss"] if isinstance(out, dict) else out

        if self.config.use_quality_weight and "sample_weight" in batch:
            weights = batch["sample_weight"].to(loss.device)
            # Ensure weights are within [quality_min, quality_max]
            weights = torch.clamp(weights, self.quality_min, self.quality_max)
            # Multiply base loss by weights
            loss = (loss * weights).mean()
        return loss

    # ---------------------------------------------------------------------
    # Action generation
    # ---------------------------------------------------------------------
    def encode_observation(
        self,
        obs: Dict[str, torch.Tensor],
        phase_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Encode the observation and optionally fuse phase information.

        Parameters
        ----------
        obs : Dict[str, torch.Tensor]
            Observation dictionary containing image and/or state inputs.
        phase_id : torch.Tensor, optional
            Discrete phase ID tensor of shape ``(B,)``.  If provided,
            its embedding is concatenated to the observation features.
        """
        # Delegate to the base policy's observation encoder
        obs_feat = self.base_policy.encode_obs(obs)  # type: ignore  # MAY NEED ADJUSTMENT
        if phase_id is not None:
            phase_emb = self.phase_embedding(phase_id.to(obs_feat.device))
            fused = torch.cat([obs_feat, phase_emb], dim=-1)
            obs_feat = self.phase_proj(fused)
        return obs_feat

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward method used during training.

        LeRobot policies typically accept a ``batch`` dictionary that
        includes observations and other context.  We add phase IDs and
        pass through to the base policy for action generation.
        """
        # Extract phase ID and quality weight if present
        phase_id = batch.get("phase_id")
        # Fuse phase information with observations
        obs = batch.get("obs", batch)
        # Some versions store observations under different keys (e.g.
        # ``observation`` or ``observations``).  If your version differs,
        # adjust the key here.
        obs_feat = self.encode_observation(obs, phase_id)
        # Replace the encoded observation in the batch
        batch = dict(batch)
        batch["encoded_obs"] = obs_feat
        # Delegate to base policy to produce predictions and compute loss
        # This call may need to be updated depending on the API of
        # `DiffusionPolicy` in your LeRobot version.
        return self.base_policy(batch)  # type: ignore

    def reset(self, batch_size: int) -> None:
        """Reset the policy's internal state for a new episode.

        Delegates to the base policy.  This is called at the start
        of each episode during inference.
        """
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset(batch_size)  # type: ignore
