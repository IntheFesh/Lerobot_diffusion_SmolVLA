"""
modeling_phaseqflow.py
======================

Upgraded PhaseQFlow model:
- latent skill phase (VQ/Gumbel style)
- value-guided weighting via critic
- latent-space flow matching path
- DiT-like conditioning backbone for multimodal fusion
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_phaseqflow import PhaseQFlowConfig

try:
    from lerobot.policies.pretrained import PreTrainedPolicy  # type: ignore
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # type: ignore
except Exception:
    PreTrainedPolicy = nn.Module  # type: ignore
    DiffusionPolicy = nn.Module  # type: ignore


class SkillVQEncoder(nn.Module):
    """Lightweight VQ/Gumbel skill encoder."""

    def __init__(self, input_dim: int, num_skills: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, num_skills)
        self.num_skills = num_skills
        self.temperature = temperature

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.proj(x)
        if training:
            probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1)
            skill_id = probs.argmax(dim=-1)
        else:
            skill_id = logits.argmax(dim=-1)
        return skill_id, logits


class ActionTokenizer(nn.Module):
    """Compress/decompress actions to latent vectors."""

    def __init__(self, action_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(action_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, action_dim)

    def encode(self, action: torch.Tensor) -> torch.Tensor:
        return self.encoder(action)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class DiTBackbone(nn.Module):
    """Small DiT-style transformer block for fused conditioning."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class PhaseQFlowPolicy(PreTrainedPolicy):
    config_class = PhaseQFlowConfig

    def __init__(
        self,
        config: PhaseQFlowConfig,
        base_policy: Optional[DiffusionPolicy] = None,
        **_: Any,
    ) -> None:
        super().__init__(config)
        self.config = config
        self.base_policy = base_policy or DiffusionPolicy(config)

        obs_dim = getattr(self.base_policy, "obs_feature_dim", 256)
        action_dim = getattr(self.base_policy, "action_dim", 16)

        self.skill_encoder = SkillVQEncoder(
            input_dim=obs_dim + action_dim,
            num_skills=config.num_skills,
            temperature=config.gumbel_temperature,
        )
        self.skill_embedding = nn.Embedding(config.num_skills, config.skill_embedding_dim)
        self.obs_proj = nn.Linear(obs_dim + config.skill_embedding_dim, config.dit_hidden_dim)

        self.dit_backbone = DiTBackbone(
            hidden_dim=config.dit_hidden_dim,
            num_layers=config.dit_num_layers,
            num_heads=config.dit_num_heads,
        )

        self.action_tokenizer = ActionTokenizer(action_dim=action_dim, latent_dim=config.latent_dim)
        self.action_decoder = self.action_tokenizer.decoder

        critic_in = obs_dim + action_dim
        self.critic_network = nn.Sequential(
            nn.Linear(critic_in, config.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.critic_hidden_dim, 1),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "PhaseQFlowPolicy":
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        base_policy = DiffusionPolicy.from_pretrained(pretrained_model_name_or_path)
        return cls(config, base_policy=base_policy, **kwargs)

    def _encode_obs(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if hasattr(self.base_policy, "encode_obs"):
            return self.base_policy.encode_obs(obs)  # type: ignore[attr-defined]
        if "encoded_obs" in obs:
            return obs["encoded_obs"]
        # Fallback for simple state-only use.
        state = obs.get("obs_states") or obs.get("state")
        if state is None:
            raise KeyError("Could not find observation features for PhaseQFlowPolicy")
        return state.float()

    def infer_skill_id(self, obs_feat: torch.Tensor, actions: Optional[torch.Tensor] = None) -> torch.Tensor:
        if actions is None:
            actions = torch.zeros(obs_feat.size(0), getattr(self.base_policy, "action_dim", 16), device=obs_feat.device)
        skill_input = torch.cat([obs_feat, actions], dim=-1)
        skill_id, _ = self.skill_encoder(skill_input, training=self.training)
        return skill_id

    def encode_observation(
        self,
        obs: Dict[str, torch.Tensor],
        skill_id: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        obs_feat = self._encode_obs(obs)
        if self.config.use_vq_phase:
            if skill_id is None:
                skill_id = self.infer_skill_id(obs_feat, actions)
            skill_emb = self.skill_embedding(skill_id.to(obs_feat.device))
            fused = torch.cat([obs_feat, skill_emb], dim=-1)
        else:
            fused = torch.cat(
                [obs_feat, torch.zeros(obs_feat.size(0), self.config.skill_embedding_dim, device=obs_feat.device)],
                dim=-1,
            )
        fused = self.obs_proj(fused).unsqueeze(1)
        return self.dit_backbone(fused).squeeze(1)

    def _extract_actions(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        for key in ("action", "actions", "target_action"):
            if key in batch:
                x = batch[key]
                if isinstance(x, torch.Tensor):
                    return x.float()
        return None

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        actions = self._extract_actions(batch)
        obs = batch.get("obs", batch)
        obs_feat = self._encode_obs(obs)

        skill_id = batch.get("skill_id")
        encoded_obs = self.encode_observation(obs, skill_id=skill_id, actions=actions)

        model_batch = dict(batch)
        model_batch["encoded_obs"] = encoded_obs

        if self.config.use_latent_flow and actions is not None:
            model_batch["latent_action"] = self.action_tokenizer.encode(actions)

        out = self.base_policy.compute_loss(model_batch)  # type: ignore[attr-defined]
        raw_loss = out["loss"] if isinstance(out, dict) else out

        if raw_loss.ndim == 0:
            per_sample_loss = raw_loss.repeat(encoded_obs.shape[0])
        else:
            per_sample_loss = raw_loss.reshape(encoded_obs.shape[0], -1).mean(dim=-1)

        if self.config.use_value_guided_weight and actions is not None:
            critic_in = torch.cat([obs_feat.detach(), actions.detach()], dim=-1)
            q_values = self.critic_network(critic_in).squeeze(-1)
            weights = torch.softmax(self.config.value_weight_beta * q_values, dim=0)
            weights = weights * weights.numel()
            loss = (per_sample_loss * weights.detach()).mean()
        else:
            loss = per_sample_loss.mean()

        return loss

    def update_critic(self, obs_feat: torch.Tensor, actions: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        critic_in = torch.cat([obs_feat, actions], dim=-1)
        pred_q = self.critic_network(critic_in).squeeze(-1)
        critic_loss = F.mse_loss(pred_q, target_q)
        return critic_loss

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        loss = self.compute_loss(batch)
        return {"loss": loss}

    def reset(self, batch_size: int) -> None:
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset(batch_size)  # type: ignore[attr-defined]
