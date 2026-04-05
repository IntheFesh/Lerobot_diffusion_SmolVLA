"""Configuration for PhaseQFlow policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PhaseQFlowConfig:
    """Lightweight config with backward-compatible fields."""

    # Legacy switches
    num_phases: int = 4
    use_quality_weight: bool = True
    use_flow_matching: bool = False
    phase_embedding_dim: int = 32
    quality_weight_min: float = 0.5
    quality_weight_max: float = 1.0

    # Learned skills (VQ/Gumbel)
    num_skills: int = 16
    use_vq_phase: bool = True
    skill_embedding_dim: int = 32
    gumbel_temperature: float = 1.0

    # Value-guided regression
    use_value_guided_weight: bool = True
    value_weight_beta: float = 2.0
    critic_hidden_dim: int = 256

    # Latent flow
    latent_dim: int = 32
    use_latent_flow: bool = True

    # Model size / lightness knobs
    action_dim: int = 16
    max_timestep: int = 2048
    base_loss_weight: float = 0.25

    # DiT backbone
    backbone_type: str = "dit"
    dit_hidden_dim: int = 256
    dit_num_layers: int = 4
    dit_num_heads: int = 8

    # Runtime
    action_buffer_maxlen: int = 128
    observation_horizon: Optional[int] = None
    action_horizon: Optional[int] = None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str) -> "PhaseQFlowConfig":
        """Load config from a directory or JSON file path."""
        path = Path(pretrained_model_name_or_path)
        if path.is_dir():
            path = path / "config.json"
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if "phaseqflow" in payload and isinstance(payload["phaseqflow"], dict):
            payload = payload["phaseqflow"]
        valid = {k: v for k, v in payload.items() if k in cls.__dataclass_fields__}
        return cls(**valid)

    def save_pretrained(self, save_directory: str) -> str:
        """Save config to `config.json` in the target directory."""
        path = Path(save_directory)
        path.mkdir(parents=True, exist_ok=True)
        out_file = path / "config.json"
        with out_file.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)
        return str(out_file)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
