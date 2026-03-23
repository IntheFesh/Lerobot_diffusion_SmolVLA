"""
processor_phaseqflow.py
=======================

This module defines the data processor for PhaseQFlow.  A processor
transforms raw dataset samples into batched tensors for training and
inference.  Specifically, this processor computes phase labels and
sample quality weights on‑the‑fly.  It also interfaces with the
underlying LeRobot processors and converters for image and state
handling.

The current implementation is deliberately lightweight.  It may need
modification to align with future versions of LeRobot.  See comments
marked with "MAY NEED ADJUSTMENT" for potential changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

try:
    # Import base classes from LeRobot if available
    from lerobot.processor import Processor  # type: ignore
    from lerobot.processor.converters import (
        convert_images_to_tensor,
        convert_states_to_tensor,
    )  # type: ignore
except Exception:
    # Fallback base class if LeRobot is not installed
    class Processor:  # type: ignore
        pass
    def convert_images_to_tensor(x: Any) -> Any:  # type: ignore
        return x
    def convert_states_to_tensor(x: Any) -> Any:  # type: ignore
        return x


@dataclass
class ProcessorConfig:
    """Configuration for PhaseQFlowProcessor.

    This simple config mirrors the PhaseQFlow policy configuration to
    specify the number of phases and whether to compute quality
    weights.  It also accepts a mapping of episode lengths to avoid
    recomputing them on the fly.
    """

    num_phases: int = 4
    use_quality_weight: bool = True
    episode_lengths: Optional[Dict[int, int]] = None


class PhaseQFlowProcessor(Processor):
    """Processor to augment dataset samples with phase IDs and quality weights."""

    def __init__(self, config: ProcessorConfig) -> None:
        super().__init__()
        self.config = config

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Process a batch of raw samples into model inputs.

        Each sample in the batch is expected to contain at least
        ``observation.images`` and ``observation.state`` keys along with
        ``frame_index`` and ``episode_index``.  Adjust the field names if
        your dataset differs.
        """
        obs_images: List[torch.Tensor] = []
        obs_states: List[torch.Tensor] = []
        phase_ids: List[int] = []
        quality_weights: List[float] = []

        # Precompute length lookup for speed
        lengths = self.config.episode_lengths or {}

        for sample in batch:
            # Images: convert to tensor; assume sample stores a list of images
            images = sample.get("observation.images.image") or sample.get("observation.images")
            states = sample.get("observation.state") or sample.get("state")
            # Convert using LeRobot converters if available
            images_tensor = convert_images_to_tensor(images)
            states_tensor = convert_states_to_tensor(states)
            obs_images.append(images_tensor)
            obs_states.append(states_tensor)

            # Compute phase ID
            epi_idx = int(sample.get("episode_index", 0))
            frame_idx = int(sample.get("frame_index", sample.get("step_index", 0)))
            epi_len = lengths.get(epi_idx)
            if epi_len is None:
                # Fallback: treat last frame as length 1 and avoid division by zero
                epi_len = frame_idx + 1
            ratio = frame_idx / max(epi_len - 1, 1)
            phase_id = min(int(math.floor(ratio * self.config.num_phases)), self.config.num_phases - 1)
            phase_ids.append(phase_id)

            # Compute quality weight (simple heuristic based on jerk)
            if self.config.use_quality_weight:
                actions = sample.get("action")
                weight = self._compute_quality_weight(actions)
            else:
                weight = 1.0
            quality_weights.append(weight)

        # Stack observations
        obs_images = torch.stack(obs_images, dim=0)
        obs_states = torch.stack(obs_states, dim=0)
        phase_ids = torch.tensor(phase_ids, dtype=torch.long)
        quality_weights = torch.tensor(quality_weights, dtype=torch.float32)

        return {
            "obs_images": obs_images,
            "obs_states": obs_states,
            "phase_id": phase_ids,
            "sample_weight": quality_weights,
        }

    def _compute_quality_weight(self, actions: Any) -> float:
        """Compute a simple quality weight from a sequence of actions.

        The quality weight is inversely proportional to the mean jerk
        (second finite difference) of the action sequence.  A
        smoother trajectory yields a higher weight in the range
        ``[0.5, 1.0]``.  If actions are not provided or invalid, a
        default weight of 1.0 is used.
        """
        try:
            import numpy as np
            arr = np.asarray(actions, dtype=float)
            if arr.ndim < 2 or arr.shape[0] < 3:
                return 1.0
            first = np.diff(arr, axis=0)
            second = np.diff(first, axis=0)
            jerk = np.mean(np.linalg.norm(second, axis=-1))
            # Normalize jerk to [0,1] heuristically
            norm = 1.0 / (1.0 + jerk)
            weight = 0.5 + 0.5 * norm
            return float(weight)
        except Exception:
            return 1.0
