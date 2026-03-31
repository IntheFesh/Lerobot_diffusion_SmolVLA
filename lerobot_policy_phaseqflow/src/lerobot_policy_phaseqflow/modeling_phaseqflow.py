"""
modeling_phaseqflow.py
======================

PhaseQFlow policy upgraded with:
1) Learned skill tokens (VQ-style latent skill phase)
2) Value-guided sample weighting (critic network)
3) Optional latent-space flow matching path

Implementation remains API-compatible with lightweight LeRobot integrations.
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
    PreTrainedPolicy = object  # type: ignore
    DiffusionPolicy = object  # type: ignore


class SimpleDiTBackbone(nn.Module):
    """Compact Transformer backbone used as a DiT-style conditioner."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.encoder(tokens)


class VectorQuantizer(nn.Module):
    """Minimal VQ module for skill token discovery."""

    def __init__(self, num_embeddings: int, embedding_dim: int, commitment_cost: float) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.commitment_cost = commitment_cost
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # z_e: [B, D]
        z = z_e.reshape(-1, z_e.shape[-1])
        distances = (
            z.pow(2).sum(dim=1, keepdim=True)
            + self.embedding.weight.pow(2).sum(dim=1)
            - 2.0 * z @ self.embedding.weight.t()
        )
        indices = torch.argmin(distances, dim=1)
        z_q = self.embedding(indices).view_as(z_e)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        codebook_loss = F.mse_loss(z_q, z_e.detach())
        commitment_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        return z_q_st, indices.view(z_e.shape[0]), vq_loss


class PhaseQFlowPolicy(PreTrainedPolicy):
    """Skill/value/latent-flow upgraded policy."""

    config_class = PhaseQFlowConfig

    def __init__(
        self,
        config: PhaseQFlowConfig,
        base_policy: Optional[DiffusionPolicy] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(config)
        self.config: PhaseQFlowConfig = config
        self.base_policy = base_policy or DiffusionPolicy(config)

        obs_dim = getattr(self.base_policy, "obs_feature_dim", config.dit_hidden_dim)
        self.obs_proj = nn.Linear(obs_dim, config.dit_hidden_dim)

        # 1) Learned skills
        self.skill_encoder = nn.Sequential(
            nn.Linear(config.dit_hidden_dim, config.dit_hidden_dim),
            nn.GELU(),
            nn.Linear(config.dit_hidden_dim, config.skill_embedding_dim),
        )
        self.skill_quantizer = VectorQuantizer(
            num_embeddings=config.num_skills,
            embedding_dim=config.skill_embedding_dim,
            commitment_cost=config.vq_commitment_cost,
        )
        self.skill_embedding = nn.Embedding(config.num_skills, config.skill_embedding_dim)
        self.skill_fuse = nn.Linear(config.dit_hidden_dim + config.skill_embedding_dim, config.dit_hidden_dim)

        # 2) Critic for value-guided weighting
        self.critic_network = nn.Sequential(
            nn.Linear(config.dit_hidden_dim * 2, config.dit_hidden_dim),
            nn.GELU(),
            nn.Linear(config.dit_hidden_dim, 1),
        )

        # 3) Latent flow components
        action_dim = getattr(self.base_policy, "action_dim", config.latent_dim)
        self.action_encoder = nn.Linear(action_dim, config.latent_dim)
        self.action_decoder = nn.Linear(config.latent_dim, action_dim)

        self.timestep_embedding = nn.Embedding(1024, config.dit_hidden_dim)
        self.dit_backbone = SimpleDiTBackbone(
            hidden_dim=config.dit_hidden_dim,
            num_layers=config.dit_num_layers,
            num_heads=config.dit_num_heads,
        )

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *model_args: Any,
        **kwargs: Any,
    ) -> "PhaseQFlowPolicy":
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        base_policy = DiffusionPolicy.from_pretrained(pretrained_model_name_or_path)
        return cls(config, base_policy)

    def _extract_actions(self, batch: Dict[str, Any], device: torch.device) -> torch.Tensor:
        actions = batch.get("action", batch.get("actions"))
        if actions is None:
            bsz = self._infer_batch_size(batch)
            action_dim = getattr(self.base_policy, "action_dim", self.config.latent_dim)
            return torch.zeros(bsz, action_dim, device=device)
        if not torch.is_tensor(actions):
            actions = torch.as_tensor(actions, dtype=torch.float32, device=device)
        return actions.to(device=device, dtype=torch.float32)

    def _infer_batch_size(self, batch: Dict[str, Any]) -> int:
        for value in batch.values():
            if torch.is_tensor(value) and value.ndim > 0:
                return int(value.shape[0])
        return 1

    def _compute_skill_token(self, obs_feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_e = self.skill_encoder(obs_feat)
        z_q, skill_id, vq_loss = self.skill_quantizer(z_e)
        return z_q, skill_id, vq_loss

    def encode_observation(
        self,
        obs: Dict[str, torch.Tensor],
        skill_id: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Encode observation and fuse learned (or provided) skill embedding."""
        base_encoder = getattr(self.base_policy, "encode_obs", None)
        if base_encoder is None:
            raise AttributeError("base_policy must expose encode_obs for PhaseQFlowPolicy")

        raw_obs_feat = base_encoder(obs)
        obs_feat = self.obs_proj(raw_obs_feat)

        vq_loss = torch.zeros((), device=obs_feat.device)
        if self.config.use_vq_phase:
            z_q, learned_skill_id, vq_loss = self._compute_skill_token(obs_feat)
            if skill_id is None:
                skill_id = learned_skill_id
            skill_emb = self.skill_embedding(skill_id.to(obs_feat.device))
            # Blend codebook vector with embedding lookup for stability
            skill_vec = 0.5 * (z_q + skill_emb)
            fused = torch.cat([obs_feat, skill_vec], dim=-1)
            obs_feat = self.skill_fuse(fused)
        return obs_feat, skill_id, vq_loss

    def _latent_action(self, actions: torch.Tensor) -> torch.Tensor:
        if not self.config.use_latent_flow:
            return actions
        return self.action_encoder(actions)

    def _decode_action(self, latent_actions: torch.Tensor) -> torch.Tensor:
        if not self.config.use_latent_flow:
            return latent_actions
        return self.action_decoder(latent_actions)

    def _critic_value(self, obs_feat: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        action_latent = self._latent_action(actions)
        if action_latent.shape[-1] != obs_feat.shape[-1]:
            action_latent = F.pad(action_latent, (0, max(0, obs_feat.shape[-1] - action_latent.shape[-1])))
            action_latent = action_latent[..., : obs_feat.shape[-1]]
        critic_in = torch.cat([obs_feat, action_latent], dim=-1)
        return self.critic_network(critic_in).squeeze(-1)

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        out = self.base_policy.compute_loss(batch)  # type: ignore
        base_loss = out["loss"] if isinstance(out, dict) else out

        obs = batch.get("obs", batch.get("observation", batch))
        obs_feat, skill_id, vq_loss = self.encode_observation(obs, batch.get("skill_id"))
        actions = self._extract_actions(batch, obs_feat.device)

        # Value-guided weighting
        if self.config.use_value_guided_weight:
            q_values = self._critic_value(obs_feat, actions)
            sample_weight = F.softmax(self.config.value_weight_beta * q_values, dim=0)
            if base_loss.ndim == 0:
                weighted_loss = base_loss
            else:
                while sample_weight.ndim < base_loss.ndim:
                    sample_weight = sample_weight.unsqueeze(-1)
                weighted_loss = (base_loss * sample_weight).sum() / sample_weight.sum().clamp_min(1e-6)
        elif self.config.use_quality_weight and "sample_weight" in batch:
            sample_weight = batch["sample_weight"].to(base_loss.device)
            sample_weight = torch.clamp(sample_weight, self.config.quality_weight_min, self.config.quality_weight_max)
            weighted_loss = (base_loss * sample_weight).mean() if base_loss.ndim > 0 else base_loss
        else:
            weighted_loss = base_loss.mean() if base_loss.ndim > 0 else base_loss

        total_loss = weighted_loss + vq_loss
        return total_loss

    def update_critic(self, batch: Dict[str, Any], gamma: float = 0.99) -> torch.Tensor:
        """Online/offline critic update entry-point.

        Expects `reward` and optionally `next_value` in batch.
        """
        obs = batch.get("obs", batch.get("observation", batch))
        obs_feat, _, _ = self.encode_observation(obs, batch.get("skill_id"))
        actions = self._extract_actions(batch, obs_feat.device)

        q_pred = self._critic_value(obs_feat, actions)
        reward = batch.get("reward")
        if reward is None:
            target = torch.zeros_like(q_pred)
        else:
            if not torch.is_tensor(reward):
                reward = torch.as_tensor(reward, dtype=torch.float32, device=q_pred.device)
            reward = reward.to(q_pred.device, dtype=torch.float32)
            next_value = batch.get("next_value")
            if next_value is None:
                target = reward
            else:
                if not torch.is_tensor(next_value):
                    next_value = torch.as_tensor(next_value, dtype=torch.float32, device=q_pred.device)
                target = reward + gamma * next_value.to(q_pred.device, dtype=torch.float32)

        return F.mse_loss(q_pred, target.detach())

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        phase_id = batch.get("phase_id")
        skill_id = batch.get("skill_id", phase_id)
        obs = batch.get("obs", batch.get("observation", batch))

        obs_feat, skill_id, vq_loss = self.encode_observation(obs, skill_id)

        batch = dict(batch)
        batch["encoded_obs"] = obs_feat
        batch["skill_id"] = skill_id
        batch["vq_loss"] = vq_loss

        # latent flow path: pre-encode action target into latent space
        if self.config.use_latent_flow and ("action" in batch or "actions" in batch):
            action_key = "action" if "action" in batch else "actions"
            actions = self._extract_actions(batch, obs_feat.device)
            batch[f"{action_key}_latent"] = self._latent_action(actions)

        output = self.base_policy(batch)  # type: ignore

        # Optional decode predicted latent action
        if self.config.use_latent_flow and isinstance(output, dict):
            if "pred_action_latent" in output:
                output["pred_action"] = self._decode_action(output["pred_action_latent"])
        return output

    def reset(self, batch_size: int) -> None:
        if hasattr(self.base_policy, "reset"):
            self.base_policy.reset(batch_size)  # type: ignore
