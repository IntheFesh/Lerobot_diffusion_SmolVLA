"""Core PhaseQFlow policy components.

This module provides a lightweight implementation of:
1) learned discrete skills (Gumbel/VQ-style),
2) value-guided sample reweighting via a critic,
3) latent-action flow-style prediction with decode-to-action.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_phaseqflow import PhaseQFlowConfig


class SkillVQEncoder(nn.Module):
    """Lightweight Gumbel-Softmax skill encoder."""

    def __init__(self, input_dim: int, num_skills: int, temperature: float = 1.0) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, num_skills)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        logits = self.proj(x)
        probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1) if training else logits.softmax(dim=-1)
        skill_id = probs.argmax(dim=-1)
        return skill_id, probs, logits


class ActionTokenizer(nn.Module):
    """Action compression/decompression for latent flow path."""

    def __init__(self, action_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(action_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, action_dim)

    def encode(self, action: torch.Tensor) -> torch.Tensor:
        return self.encoder(action)

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)


class DiTBackbone(nn.Module):
    """Small DiT-style transformer encoder."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class LatentFlowHead(nn.Module):
    """Predict latent action vectors from encoded observations."""

    def __init__(self, obs_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.SiLU(),
            nn.Linear(obs_dim, latent_dim),
        )

    def forward(self, encoded_obs: torch.Tensor) -> torch.Tensor:
        return self.net(encoded_obs)


class PhaseQFlowPolicy(nn.Module):
    """Lightweight, self-contained PhaseQFlow policy.

    `base_policy` is optional. If provided and it exposes `compute_loss`, this
    class will still use its loss, while all upgraded components remain active
    (skills/critic/latent flow outputs and auxiliary losses).
    """

    config_class = PhaseQFlowConfig

    def __init__(self, config: PhaseQFlowConfig, base_policy: Optional[nn.Module] = None, **_: Any) -> None:
        super().__init__()
        self.config = config
        self.base_policy = base_policy

        obs_dim = config.dit_hidden_dim
        action_dim = config.action_dim

        self.obs_encoder = nn.LazyLinear(config.dit_hidden_dim)
        self.skill_encoder = SkillVQEncoder(
            input_dim=config.dit_hidden_dim + action_dim,
            num_skills=config.num_skills,
            temperature=config.gumbel_temperature,
        )
        self.skill_embedding = nn.Embedding(config.num_skills, config.skill_embedding_dim)
        self.obs_proj = nn.Linear(config.dit_hidden_dim + config.skill_embedding_dim, config.dit_hidden_dim)
        self.timestep_embedding = nn.Embedding(config.max_timestep, config.dit_hidden_dim)

        self.dit_backbone = DiTBackbone(
            hidden_dim=config.dit_hidden_dim,
            num_layers=config.dit_num_layers,
            num_heads=config.dit_num_heads,
        )

        self.action_tokenizer = ActionTokenizer(action_dim=action_dim, latent_dim=config.latent_dim)
        self.latent_flow_head = LatentFlowHead(obs_dim=config.dit_hidden_dim, latent_dim=config.latent_dim)
        self.action_decoder = self.action_tokenizer.decoder

        critic_in = config.dit_hidden_dim + action_dim
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
        _ = args
        return cls(config=config, **kwargs)

    def to_config_dict(self) -> Dict[str, Any]:
        return asdict(self.config)

    def _extract_obs_tensor(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "encoded_obs" in obs:
            return obs["encoded_obs"].float()
        if "obs_states" in obs:
            return obs["obs_states"].float()
        if "state" in obs:
            return obs["state"].float()
        raise KeyError("Expected one of: encoded_obs, obs_states, state")

    def _extract_actions(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        for key in ("action", "actions", "target_action"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value.float()
        return None

    def _compute_skill_id(
        self,
        obs_feat: torch.Tensor,
        actions: Optional[torch.Tensor],
        skill_id: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if skill_id is not None and torch.all(skill_id >= 0):
            return skill_id.long(), torch.empty(0, device=obs_feat.device)

        if actions is None:
            actions = torch.zeros(obs_feat.size(0), self.config.action_dim, device=obs_feat.device)
        encoder_in = torch.cat([obs_feat, actions], dim=-1)
        inferred_id, _, logits = self.skill_encoder(encoder_in, training=self.training)
        return inferred_id, logits

    def encode_observation(
        self,
        obs: Dict[str, torch.Tensor],
        skill_id: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        raw_obs = self._extract_obs_tensor(obs)
        obs_feat = self.obs_encoder(raw_obs)

        selected_skill, skill_logits = self._compute_skill_id(obs_feat, actions, skill_id)
        if self.config.use_vq_phase:
            skill_emb = self.skill_embedding(selected_skill.to(obs_feat.device))
        else:
            skill_emb = torch.zeros(obs_feat.size(0), self.config.skill_embedding_dim, device=obs_feat.device)

        fused = self.obs_proj(torch.cat([obs_feat, skill_emb], dim=-1))

        if timestep is None:
            timestep = torch.zeros(obs_feat.size(0), dtype=torch.long, device=obs_feat.device)
        timestep = torch.clamp(timestep.long(), 0, self.config.max_timestep - 1)
        t_emb = self.timestep_embedding(timestep)

        token_seq = torch.stack([fused, t_emb], dim=1)
        encoded = self.dit_backbone(token_seq)[:, 0, :]
        return encoded, skill_logits

    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        obs = batch.get("obs", batch)
        actions = self._extract_actions(batch)
        timestep = batch.get("timestep")
        skill_id = batch.get("skill_id")

        encoded_obs, skill_logits = self.encode_observation(obs, skill_id=skill_id, actions=actions, timestep=timestep)

        latent_pred = self.latent_flow_head(encoded_obs)
        action_pred = self.action_decoder(latent_pred)

        return {
            "encoded_obs": encoded_obs,
            "latent_action_pred": latent_pred,
            "action_pred": action_pred,
            "skill_logits": skill_logits,
        }

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        preds = self.predict_action(batch)
        actions = self._extract_actions(batch)
        if actions is None:
            raise KeyError("compute_loss requires action/actions/target_action in batch")

        latent_target = self.action_tokenizer.encode(actions) if self.config.use_latent_flow else actions
        latent_pred = preds["latent_action_pred"] if self.config.use_latent_flow else preds["action_pred"]

        per_sample_loss = F.mse_loss(latent_pred, latent_target, reduction="none").mean(dim=-1)

        base_loss = None
        if self.base_policy is not None and hasattr(self.base_policy, "compute_loss"):
            base_out = self.base_policy.compute_loss(batch)  # type: ignore[attr-defined]
            if isinstance(base_out, dict) and "loss" in base_out:
                base_loss = base_out["loss"]
            elif isinstance(base_out, torch.Tensor):
                base_loss = base_out

        obs = batch.get("obs", batch)
        raw_obs = self._extract_obs_tensor(obs)
        obs_feat = self.obs_encoder(raw_obs)
        q_values = self.critic_network(torch.cat([obs_feat.detach(), actions.detach()], dim=-1)).squeeze(-1)

        if self.config.use_value_guided_weight:
            weights = torch.softmax(self.config.value_weight_beta * q_values, dim=0)
            weights = weights * weights.numel()
            weighted_loss = (per_sample_loss * weights.detach()).mean()
        else:
            weighted_loss = per_sample_loss.mean()

        if base_loss is not None:
            if base_loss.ndim > 0:
                base_loss = base_loss.mean()
            loss = weighted_loss + self.config.base_loss_weight * base_loss
        else:
            loss = weighted_loss

        return loss

    def update_critic(self, obs_feat: torch.Tensor, actions: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        critic_in = torch.cat([obs_feat, actions], dim=-1)
        pred_q = self.critic_network(critic_in).squeeze(-1)
        return F.mse_loss(pred_q, target_q)

    def forward(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        loss = self.compute_loss(batch)
        preds = self.predict_action(batch)
        preds["loss"] = loss
        return preds

    def reset(self, batch_size: int) -> None:
        _ = batch_size
