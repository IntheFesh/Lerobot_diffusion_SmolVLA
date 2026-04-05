"""Data processor for PhaseQFlow training/inference batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import torchvision.transforms as T


@dataclass
class ProcessorConfig:
    num_skills: int = 16
    use_vq_phase: bool = True
    use_value_guided_weight: bool = True
    state_noise_std: float = 0.01
    image_randaugment_n: int = 2
    image_randaugment_m: int = 9


class PhaseQFlowProcessor:
    """Prepare tensors and lightweight augmentations.

    This processor intentionally does not compute temporal phase ids or
    jerk-based quality weights. Skill ids and value-guided weights are learned
    in the model through the skill encoder and critic.
    """

    def __init__(self, config: ProcessorConfig) -> None:
        self.config = config
        self.image_aug = T.RandAugment(num_ops=config.image_randaugment_n, magnitude=config.image_randaugment_m)

    @staticmethod
    def _to_tensor(x: Any) -> torch.Tensor:
        return x if isinstance(x, torch.Tensor) else torch.as_tensor(x)

    def _augment_images(self, images_tensor: torch.Tensor) -> torch.Tensor:
        if images_tensor.ndim != 4:
            return images_tensor
        out = []
        for img in images_tensor:
            image_in = img
            if image_in.dtype != torch.uint8:
                image_in = (image_in.clamp(0, 1) * 255.0).to(torch.uint8)
            image_out = self.image_aug(image_in)
            out.append(image_out.float() / 255.0)
        return torch.stack(out, dim=0)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        obs_images: List[torch.Tensor] = []
        obs_states: List[torch.Tensor] = []

        for sample in batch:
            images = sample.get("observation.images.image", sample.get("observation.images"))
            states = sample.get("observation.state", sample.get("state"))
            if images is None or states is None:
                raise KeyError("Each sample must include observation images and state")

            obs_images.append(self._to_tensor(images).float())
            obs_states.append(self._to_tensor(states).float())

        obs_images_t = torch.stack(obs_images, dim=0)
        obs_states_t = torch.stack(obs_states, dim=0)

        obs_images_t = self._augment_images(obs_images_t)
        if self.config.state_noise_std > 0:
            obs_states_t = obs_states_t + torch.randn_like(obs_states_t) * self.config.state_noise_std

        batch_size = obs_images_t.shape[0]
        skill_id = torch.full((batch_size,), -1, dtype=torch.long)
        sample_weight = torch.ones(batch_size, dtype=torch.float32)

        return {
            "obs_images": obs_images_t,
            "obs_states": obs_states_t,
            "skill_id": skill_id,
            "sample_weight": sample_weight,
        }
