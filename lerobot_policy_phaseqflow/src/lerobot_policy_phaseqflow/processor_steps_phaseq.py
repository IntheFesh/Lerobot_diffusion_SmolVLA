"""
processor_steps_phaseq.py
=========================

Utilities for online phase/quality computation during step-based inference.it defines functions for computing phase IDs and
quality weights during online inference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def compute_phase_id(
    frame_index: int,
    episode_length: int,
    num_phases: int,
) -> int:
    """
    Compute discrete phase id from trajectory progress.

    Args:
        frame_index: Current frame index (>=0).
        episode_length: Total episode length (>=1).
        num_phases: Number of phases (>=1).

    Returns:
        phase_id in [0, num_phases - 1].
    """
    if num_phases <= 0:
        raise ValueError("num_phases must be >= 1")
    if episode_length <= 0:
        raise ValueError("episode_length must be >= 1")
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")

    denom = max(episode_length - 1, 1)
    ratio = frame_index / denom
    phase_id = int(np.floor(ratio * num_phases))
    return min(max(phase_id, 0), num_phases - 1)


def compute_quality_weight_from_actions(
    actions: np.ndarray,
    min_weight: float = 0.5,
    max_weight: float = 1.0,
) -> float:
    """
    Compute quality weight from action smoothness (inverse jerk).

    Args:
        actions: shape (T, A) action sequence.
        min_weight: lower clamp bound.
        max_weight: upper clamp bound.

    Returns:
        quality weight in [min_weight, max_weight].
    """
    if min_weight > max_weight:
        raise ValueError("min_weight must be <= max_weight")

    arr = np.asarray(actions, dtype=float)
    if arr.ndim < 2 or arr.shape[0] < 3:
        return float(max_weight)

    first = np.diff(arr, axis=0)
    second = np.diff(first, axis=0)
    jerk = np.mean(np.linalg.norm(second, axis=-1))
    norm = 1.0 / (1.0 + float(jerk))
    raw = min_weight + (max_weight - min_weight) * norm
    return float(np.clip(raw, min_weight, max_weight))


@dataclass
class OnlinePhaseState:
    """
    Lightweight helper to maintain online step state.
    """
    num_phases: int = 4
    episode_length: Optional[int] = None
    min_weight: float = 0.5
    max_weight: float = 1.0
    frame_index: int = 0
    _action_buffer: list[np.ndarray] = field(default_factory=list)

    def reset(self, episode_length: Optional[int] = None) -> None:
        self.episode_length = episode_length
        self.frame_index = 0
        self._action_buffer.clear()

    def step(self, action_t: np.ndarray) -> tuple[int, float]:
        """
        Update state with one action and return (phase_id, quality_weight).
        """
        self._action_buffer.append(np.asarray(action_t))
        ep_len = self.episode_length if self.episode_length is not None else max(self.frame_index + 1, 1)
        phase_id = compute_phase_id(self.frame_index, ep_len, self.num_phases)

        # Compute quality on recent trajectory.
        traj = np.stack(self._action_buffer, axis=0)
        q = compute_quality_weight_from_actions(traj, self.min_weight, self.max_weight)

        self.frame_index += 1
        return phase_id, q
