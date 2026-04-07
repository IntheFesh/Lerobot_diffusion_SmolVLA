"""Configuration for PhaseQFlow policy."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class PhaseQFlowConfig:
    """Config for four-layer PhaseQFlow++ architecture.

    The defaults remain lightweight for single-GPU experimentation.
    """

    # Legacy switches kept for compatibility
    num_phases: int = 4
    phase_embedding_dim: int = 32

    # Multimodal tokenizer
    vision_token_dim: int = 256
    state_token_dim: int = 128
    language_token_dim: int = 128
    history_token_dim: int = 128
    fusion_hidden_dim: int = 256
    freeze_vision_encoder: bool = True
    use_vision_adapter: bool = True

    # Asymmetric cross attention
    cross_attn_heads: int = 8
    cross_attn_dropout: float = 0.0

    # Hierarchical latent planner
    num_skills: int = 16
    use_vq_phase: bool = True
    skill_embedding_dim: int = 32
    gumbel_temperature: float = 1.0
    continuous_skill_dim: int = 32
    weak_phase_supervision_mode: str = "hybrid"  # manual|latent|hybrid

    # Value-guided weighting
    use_value_guided_weight: bool = True
    value_weight_beta: float = 2.0
    critic_hidden_dim: int = 256

    # Conditional flow
    latent_dim: int = 32
    use_latent_flow: bool = True
    flow_steps: int = 4

    # Closed-loop verifier
    verifier_hidden_dim: int = 128
    replan_confidence_threshold: float = 0.5

    # Model size / lightness knobs
    action_dim: int = 16
    max_timestep: int = 2048
    base_loss_weight: float = 0.25

    # Transformer backbone
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
