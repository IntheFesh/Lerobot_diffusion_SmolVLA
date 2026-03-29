"""
processor_steps_phaseq.py
=========================

Online helpers for skill-phase/value-guided inference.

Changes:
- Removes time-based compute_phase_id logic.
- Removes jerk-based quality weighting.
- Uses bounded deque buffers and incremental statistics.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional

import numpy as np


def compute_skill_id_from_logits(skill_logits: np.ndarray) -> int:
    """Return discrete skill token id from skill logits/probabilities."""
    logits = np.asarray(skill_logits, dtype=float)
    if logits.ndim != 1:
        raise ValueError("skill_logits must be a 1D array")
    return int(np.argmax(logits))


def compute_value_weight(
    q_value: float,
    beta: float = 2.0,
    min_weight: float = 1e-3,
) -> float:
    """Convert critic value into positive sample weight."""
    weight = float(np.exp(beta * float(q_value)))
    return float(max(weight, min_weight))


@dataclass
class OnlineSkillState:
    """State helper for online inference with bounded history."""

    num_skills: int = 16
    beta: float = 2.0
    action_buffer_maxlen: int = 128
    _action_buffer: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=128))
    _weight_buffer: Deque[float] = field(default_factory=lambda: deque(maxlen=128))

    def __post_init__(self) -> None:
        self._action_buffer = deque(maxlen=self.action_buffer_maxlen)
        self._weight_buffer = deque(maxlen=self.action_buffer_maxlen)

    def reset(self) -> None:
        self._action_buffer.clear()
        self._weight_buffer.clear()

    def step(self, action_t: np.ndarray, skill_logits_t: np.ndarray, q_value_t: float) -> tuple[int, float]:
        self._action_buffer.append(np.asarray(action_t, dtype=float))

        skill_id = compute_skill_id_from_logits(skill_logits_t)
        raw_weight = compute_value_weight(q_value_t, beta=self.beta)
        self._weight_buffer.append(raw_weight)

        # Incremental normalization on bounded window for stable scaling.
        avg_weight = float(np.mean(self._weight_buffer)) if self._weight_buffer else 1.0
        norm_weight = raw_weight / max(avg_weight, 1e-6)
        return skill_id, float(norm_weight)
