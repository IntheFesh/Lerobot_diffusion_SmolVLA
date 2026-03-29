"""
processor_phaseqflow.py
=======================

Data processor for upgraded PhaseQFlow.

Changes:
- Removes temporal phase computation from frame index.
- Removes jerk-based quality weighting.
- Adds light image/state augmentation hooks for robustness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch

try:
    import torchvision.transforms as T
except Exception:
    T = None

try:
    from lerobot.processor import Processor
    from lerobot.processor.converters import convert_images_to_tensor, convert_states_to_tensor
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
    use_vq_phase: bool = True
    use_value_guided_weight: bool = True
    state_noise_std: float = 0.01
    image_randaugment_n: int = 2
    image_randaugment_m: int = 9


class PhaseQFlowProcessor(Processor):
    def __init__(self, config: ProcessorConfig) -> None:
        super().__init__()
        self.config = config
        self.image_aug = None
        if T is not None:
            self.image_aug = T.RandAugment(num_ops=config.image_randaugment_n, magnitude=config.image_randaugment_m)

    def _augment_images(self, images_tensor: torch.Tensor) -> torch.Tensor:
        if self.image_aug is None:
            return images_tensor
        if images_tensor.ndim < 4:
            return images_tensor
        out = []
        for img in images_tensor:
            img_in = img
            # RandAugment expects uint8 image in [0,255] for best compatibility.
            if img_in.dtype != torch.uint8:
                img_in = (img_in.clamp(0, 1) * 255.0).to(torch.uint8)
            img_aug = self.image_aug(img_in)
            out.append(img_aug.float() / 255.0)
        return torch.stack(out, dim=0)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        obs_images: List[torch.Tensor] = []
        obs_states: List[torch.Tensor] = []

        for sample in batch:
            images = sample.get("observation.images.image") or sample.get("observation.images")
            states = sample.get("observation.state") or sample.get("state")

            images_tensor = convert_images_to_tensor(images)
            states_tensor = convert_states_to_tensor(states)

            if not isinstance(images_tensor, torch.Tensor):
                images_tensor = torch.as_tensor(images_tensor)
            if not isinstance(states_tensor, torch.Tensor):
                states_tensor = torch.as_tensor(states_tensor)

            obs_images.append(images_tensor.float())
            obs_states.append(states_tensor.float())

        obs_images_t = torch.stack(obs_images, dim=0)
        obs_states_t = torch.stack(obs_states, dim=0)

        obs_images_t = self._augment_images(obs_images_t)
        if self.config.state_noise_std > 0:
            obs_states_t = obs_states_t + torch.randn_like(obs_states_t) * self.config.state_noise_std

        batch_size = obs_images_t.shape[0]
        # skill_id is discovered by model-side VQ encoder; processor provides placeholder.
        skill_id = torch.full((batch_size,), -1, dtype=torch.long)
        # sample_weight is learned from critic/value model; default uniform.
        sample_weight = torch.ones(batch_size, dtype=torch.float32)

        return {
            "obs_images": obs_images_t,
            "obs_states": obs_states_t,
            "skill_id": skill_id,
            "sample_weight": sample_weight,
        }
