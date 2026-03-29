"""
processor_steps_phaseq.py
=========================

Online utilities for skill/value signals during step-based inference.
Temporal phase logic has been replaced by latent skill phase utilities.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def infer_skill_id_from_features(
    feature: np.ndarray,
    codebook: np.ndarray,
) -> int:
    """Assign nearest codebook index as a latent skill token."""
    feat = np.asarray(feature, dtype=float).reshape(1, -1)
    codes = np.asarray(codebook, dtype=float)
    if codes.ndim != 2:
        raise ValueError("codebook must have shape (K, D)")
    if feat.shape[-1] != codes.shape[-1]:
        raise ValueError("feature and codebook embedding dimensions must match")

    distances = np.sum((codes - feat) ** 2, axis=-1)
    return int(np.argmin(distances))


def compute_value_weight(
    q_value: float,
    beta: float = 1.0,
    min_weight: float = 1e-3,
) -> float:
    """Compute an exponential value-guided scalar weight."""
    weight = np.exp(beta * float(q_value))
    return float(max(weight, min_weight))


@dataclass
class OnlinePhaseState:
    """Lightweight online helper with bounded action history."""

    num_skills: int = 16
    action_window: int = 64
    beta: float = 1.0
    frame_index: int = 0
    _action_buffer: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=64))

    def reset(self) -> None:
        self.frame_index = 0
        self._action_buffer.clear()

    def step(
        self,
        action_t: np.ndarray,
        skill_id: Optional[int] = None,
        q_value: Optional[float] = None,
    ) -> tuple[int, float]:
        """Update state and return (skill_id, value_weight)."""
        if self._action_buffer.maxlen != self.action_window:
            self._action_buffer = deque(self._action_buffer, maxlen=self.action_window)

        self._action_buffer.append(np.asarray(action_t, dtype=float))

        if skill_id is None:
            # Fallback skill from action statistics if encoder output is absent.
            action_mean = float(np.mean(self._action_buffer[-1]))
            skill_id = int(abs(action_mean * 997)) % max(self.num_skills, 1)

        if q_value is None:
            # Incremental proxy q from recent action variance.
            arr = np.asarray(self._action_buffer)
            q_value = -float(np.var(arr)) if arr.size > 0 else 0.0

        value_weight = compute_value_weight(q_value, beta=self.beta)
        self.frame_index += 1
        return int(skill_id), value_weight
