"""
processor_phaseqflow.py
=======================

Data processor for the upgraded PhaseQFlow pipeline:
- optional visual augmentation (rand-augment style + cutout)
- state noise injection
- learned-skill placeholder IDs instead of temporal phase IDs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

try:
    from lerobot.processor import Processor
    from lerobot.processor.converters import (
        convert_images_to_tensor,
        convert_states_to_tensor,
    )
except Exception:
    class Processor:
        pass

    def convert_images_to_tensor(x: Any) -> Any:
        return x

    def convert_states_to_tensor(x: Any) -> Any:
        return x


@dataclass
class ProcessorConfig:
    num_skills: int = 16
    use_value_weight: bool = True
    apply_augmentation: bool = True
    state_noise_std: float = 0.01
    cutout_frac: float = 0.15


class PhaseQFlowProcessor(Processor):
    """Processor to augment samples with skill IDs and sample weights."""

    def __init__(self, config: ProcessorConfig) -> None:
        super().__init__()
        self.config = config

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        obs_images: List[torch.Tensor] = []
        obs_states: List[torch.Tensor] = []
        skill_ids: List[int] = []
        sample_weights: List[float] = []

        for sample in batch:
            images = sample.get("observation.images.image") or sample.get("observation.images")
            states = sample.get("observation.state") or sample.get("state")

            images_tensor = self._to_tensor(convert_images_to_tensor(images), dtype=torch.float32)
            states_tensor = self._to_tensor(convert_states_to_tensor(states), dtype=torch.float32)

            if self.config.apply_augmentation:
                images_tensor = self._augment_images(images_tensor)
            states_tensor = self._augment_states(states_tensor)

            obs_images.append(images_tensor)
            obs_states.append(states_tensor)

            skill_id = sample.get("skill_id")
            if skill_id is None:
                # Lightweight deterministic fallback from state statistics.
                state_mean = float(states_tensor.mean().item()) if states_tensor.numel() > 0 else 0.0
                skill_id = int(abs(state_mean * 997)) % max(self.config.num_skills, 1)
            skill_ids.append(int(skill_id))

            q_value = sample.get("q_value", sample.get("return", 0.0))
            sample_weights.append(self._value_weight(q_value) if self.config.use_value_weight else 1.0)

        return {
            "obs_images": torch.stack(obs_images, dim=0),
            "obs_states": torch.stack(obs_states, dim=0),
            "skill_id": torch.tensor(skill_ids, dtype=torch.long),
            "sample_weight": torch.tensor(sample_weights, dtype=torch.float32),
        }

    def _to_tensor(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        if torch.is_tensor(x):
            return x.to(dtype=dtype)
        return torch.as_tensor(x, dtype=dtype)

    def _augment_images(self, images: torch.Tensor) -> torch.Tensor:
        x = images
        # Ensure image-like scale assumptions remain stable.
        if x.is_floating_point():
            # Random brightness/contrast jitter
            brightness = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 0.2
            contrast = 1.0 + (torch.rand(1, device=x.device) - 0.5) * 0.2
            x_mean = x.mean()
            x = (x - x_mean) * contrast + x_mean
            x = x * brightness

        # Cutout on last two dims (H, W)
        if x.ndim >= 3:
            h = x.shape[-2]
            w = x.shape[-1]
            ch = max(1, int(h * self.config.cutout_frac))
            cw = max(1, int(w * self.config.cutout_frac))
            top = torch.randint(0, max(h - ch + 1, 1), (1,), device=x.device).item()
            left = torch.randint(0, max(w - cw + 1, 1), (1,), device=x.device).item()
            x = x.clone()
            x[..., top : top + ch, left : left + cw] = 0
        return x

    def _augment_states(self, states: torch.Tensor) -> torch.Tensor:
        if self.config.state_noise_std <= 0:
            return states
        noise = torch.randn_like(states) * self.config.state_noise_std
        return states + noise

    def _value_weight(self, q_value: Any, beta: float = 1.0) -> float:
        q = float(q_value)
        return float(torch.exp(torch.tensor(beta * q)).item())
