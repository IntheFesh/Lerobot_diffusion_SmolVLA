"""Core PhaseQFlow policy components with four-layer architecture."""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .configuration_phaseqflow import PhaseQFlowConfig


class SkillVQEncoder(nn.Module):
    """Gumbel-Softmax discrete phase encoder."""

    def __init__(self, input_dim: int, num_skills: int, temperature: float = 1.0) -> None:
        """Initialize the linear projection used for discrete phase inference."""
        super().__init__()
        self.proj = nn.Linear(input_dim, num_skills)
        self.temperature = temperature

    def forward(self, x: torch.Tensor, training: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Infer discrete skill ids and return `(id, probs, logits)`."""
        logits = self.proj(x)
        probs = F.gumbel_softmax(logits, tau=self.temperature, hard=True, dim=-1) if training else logits.softmax(dim=-1)
        skill_id = probs.argmax(dim=-1)
        return skill_id, probs, logits


'''
resnet18 - VisionTokenizer
from torchvision.models import resnet18
from typing import Dict, Optional

class VisionTokenizer(nn.Module):
    def __init__(self, config: object) -> None:
        super().__init__()
        self.config = config
        self.spatial_grid = getattr(config, "spatial_grid_size", 4)
        resnet = resnet18(weights=None)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-2])
        self.vision_pool = nn.AdaptiveAvgPool2d((self.spatial_grid, self.spatial_grid))
        self.vision_proj = nn.Sequential(
            nn.Conv2d(512, config.vision_token_dim, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=32, num_channels=config.vision_token_dim),
            nn.GELU()
        )

        self.vision_adapter = nn.Linear(config.vision_token_dim, config.fusion_hidden_dim)
        self.state_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.language_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.history_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
            batch_first=True,
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

    def maybe_freeze_vision(self) -> None:
        if not getattr(self.config, "freeze_vision_encoder", False):
            return
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.vision_pool.parameters():
            p.requires_grad = False
        for p in self.vision_proj.parameters():
            p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        del masks
        if states.ndim > 2:
            states = states.flatten(start_dim=1)

        B = images.shape[0]
        feat = self.vision_encoder(images)                            # [B, 512, H/32, W/32]
        feat = self.vision_pool(feat)                                 # [B, 512, G, G]
        feat = self.vision_proj(feat)                                 # [B, token_dim, G, G]
        feat = feat.permute(0, 2, 3, 1).reshape(B, -1, self.config.vision_token_dim) # [B, G², token_dim]
        vision_tokens = self.vision_adapter(feat)                     # [B, G², fusion_hidden_dim]

        state_tokens = self.state_tokenizer(states).unsqueeze(1)
        if language is None:
            language = torch.zeros_like(states[:, :1])
        if language.ndim > 2:
            language = language.flatten(start_dim=1)
        language_tokens = self.language_tokenizer(language).unsqueeze(1)
        if history is None:
            history = states
        if history.ndim > 2:
            history = history.flatten(start_dim=1)
        history_tokens = self.history_tokenizer(history).unsqueeze(1)
        query_tokens = torch.cat([state_tokens, history_tokens], dim=1)
        attended, _ = self.cross_attn(query=query_tokens, key=vision_tokens, value=vision_tokens)
        attended_summary = attended.mean(dim=1)
        proprio_summary = torch.cat([state_tokens, history_tokens], dim=1).mean(dim=1)

        gate = torch.sigmoid(self.uncertainty_gate(torch.cat([attended_summary, proprio_summary], dim=-1)))
        fused = gate * attended_summary + (1.0 - gate) * proprio_summary
        vision_tokens_pooled = vision_tokens.mean(dim=1, keepdim=True)
        context_tokens = torch.cat([state_tokens, history_tokens, language_tokens, vision_tokens_pooled], dim=1)

        return {
            "fused": fused,
            "context_tokens": context_tokens,
            "vision_tokens": vision_tokens,
            "state_tokens": state_tokens,
            "language_tokens": language_tokens,
            "history_tokens": history_tokens,
            "uncertainty_gate": gate.squeeze(-1),
        }
        '''


class VisionTokenizer(nn.Module):
    """Multimodal tokenization with asymmetric cross-attention + uncertainty gate."""

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Build multimodal tokenizers, cross-attention, and uncertainty gate."""
        super().__init__()
        self.config = config

        self.vision_backbone = nn.LazyLinear(config.vision_token_dim)
        self.vision_adapter = nn.Linear(config.vision_token_dim, config.fusion_hidden_dim)
        self.state_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.language_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)
        self.history_tokenizer = nn.LazyLinear(config.fusion_hidden_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=config.fusion_hidden_dim,
            num_heads=config.cross_attn_heads,
            dropout=config.cross_attn_dropout,
            batch_first=True,
        )

        self.uncertainty_gate = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim * 2, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, 1),
        )

    def maybe_freeze_vision(self) -> None:
        """Freeze the vision backbone when adapter-style training is requested."""
        if not self.config.freeze_vision_encoder:
            return
        for p in self.vision_backbone.parameters():
            p.requires_grad = False

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Tokenize multimodal inputs and fuse visual/proprioceptive context."""
        del masks
        if images.ndim > 2:
            images = images.flatten(start_dim=1)
        if states.ndim > 2:
            states = states.flatten(start_dim=1)

        vision_tokens = self.vision_backbone(images).unsqueeze(1)
        vision_tokens = self.vision_adapter(vision_tokens)

        state_tokens = self.state_tokenizer(states).unsqueeze(1)

        if language is None:
            language = torch.zeros_like(states[:, :1])
        if language.ndim > 2:
            language = language.flatten(start_dim=1)
        language_tokens = self.language_tokenizer(language).unsqueeze(1)

        if history is None:
            history = states
        if history.ndim > 2:
            history = history.flatten(start_dim=1)
        history_tokens = self.history_tokenizer(history).unsqueeze(1)

        query_tokens = torch.cat([state_tokens, history_tokens], dim=1)
        attended, _ = self.cross_attn(query=query_tokens, key=vision_tokens, value=vision_tokens)
        attended_summary = attended.mean(dim=1)
        proprio_summary = torch.cat([state_tokens, history_tokens], dim=1).mean(dim=1)

        gate = torch.sigmoid(self.uncertainty_gate(torch.cat([attended_summary, proprio_summary], dim=-1)))
        fused = gate * attended_summary + (1.0 - gate) * proprio_summary

        context_tokens = torch.cat([state_tokens, history_tokens, language_tokens, vision_tokens], dim=1)
        return {
            "fused": fused,
            "context_tokens": context_tokens,
            "vision_tokens": vision_tokens,
            "state_tokens": state_tokens,
            "language_tokens": language_tokens,
            "history_tokens": history_tokens,
            "uncertainty_gate": gate.squeeze(-1),
        }


class HierarchicalPlanner(nn.Module):
    """Discrete phase + continuous skill latent planner."""

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Initialize discrete phase encoder and continuous skill latent head."""
        super().__init__()
        self.config = config
        self.phase_encoder = SkillVQEncoder(config.fusion_hidden_dim, config.num_skills, config.gumbel_temperature)
        self.skill_continuous = nn.Sequential(
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, config.continuous_skill_dim),
        )
        self.phase_embedding = nn.Embedding(config.num_skills, config.skill_embedding_dim)

    def forward(
        self,
        fused_obs: torch.Tensor,
        phase_labels: Optional[torch.Tensor] = None,
        phase_mode: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """Infer phase/skill latents under manual, latent, or hybrid supervision modes."""
        mode = phase_mode or self.config.weak_phase_supervision_mode
        inferred_phase_id, _, phase_logits = self.phase_encoder(fused_obs, training=self.training)

        if mode == "manual" and phase_labels is not None and torch.all(phase_labels >= 0):
            phase_id = phase_labels.long()
        else:
            phase_id = inferred_phase_id

        phase_embed = self.phase_embedding(phase_id)
        skill_latent = self.skill_continuous(fused_obs)

        return {
            "phase_id": phase_id,
            "phase_logits": phase_logits,
            "phase_embed": phase_embed,
            "skill_latent": skill_latent,
        }


class FlowActionHead(nn.Module):
    """Phase-conditioned continuous action chunk generator."""

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Create conditional flow field and action decoder."""
        super().__init__()
        self.config = config
        cond_dim = config.fusion_hidden_dim + config.skill_embedding_dim + config.continuous_skill_dim
        self.latent_dim = config.latent_dim
        self.conditioner = nn.Linear(cond_dim, config.fusion_hidden_dim)
        self.flow_field = nn.Sequential(
            nn.Linear(config.latent_dim + config.fusion_hidden_dim + 1, config.fusion_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.fusion_hidden_dim, config.latent_dim),
        )
        self.action_decoder = nn.Linear(config.latent_dim, config.action_dim)

    def forward(self, fused_obs: torch.Tensor, phase_embed: torch.Tensor, skill_latent: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Integrate a conditional flow from noise and decode to actions."""
        cond = self.conditioner(torch.cat([fused_obs, phase_embed, skill_latent], dim=-1))
        u = torch.randn(fused_obs.size(0), self.latent_dim, device=fused_obs.device)
        dt = 1.0 / max(self.config.flow_steps, 1)
        for i in range(self.config.flow_steps):
            tau = torch.full((u.size(0), 1), i * dt, device=u.device)
            du = self.flow_field(torch.cat([u, cond, tau], dim=-1))
            u = u + dt * du
        action_pred = self.action_decoder(u)
        return {"latent_action_pred": u, "action_pred": action_pred}


class ChunkVerifier(nn.Module):
    """Closed-loop chunk confidence + phase drift estimator."""

    def __init__(self, config: PhaseQFlowConfig) -> None:
        """Build chunk confidence and phase-drift prediction head."""
        super().__init__()
        in_dim = config.fusion_hidden_dim + config.action_dim + config.skill_embedding_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.verifier_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.verifier_hidden_dim, 2),
        )

    def forward(self, fused_obs: torch.Tensor, predicted_action: torch.Tensor, phase_embed: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Estimate online execution confidence and replanning flags."""
        out = self.net(torch.cat([fused_obs, predicted_action, phase_embed], dim=-1))
        confidence = torch.sigmoid(out[:, 0])
        phase_drift = torch.sigmoid(out[:, 1])
        return {
            "chunk_confidence": confidence,
            "phase_drift": phase_drift,
            "should_replan": (confidence < 0.5) | (phase_drift > 0.5),
        }


class DiTBackbone(nn.Module):
    """Transformer context encoder."""

    def __init__(self, hidden_dim: int, num_layers: int, num_heads: int) -> None:
        """Construct a lightweight Transformer encoder for context tokens."""
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
        """Encode token sequences into contextualized representations."""
        return self.encoder(x)


class PhaseQFlowPolicy(nn.Module):
    """Four-module PhaseQFlow policy."""

    config_class = PhaseQFlowConfig

    def __init__(self, config: PhaseQFlowConfig, base_policy: Optional[nn.Module] = None, **_: Any) -> None:
        """Initialize all four policy modules and auxiliary loss networks."""
        super().__init__()
        self.config = config
        self.base_policy = base_policy

        self.vision_tokenizer = VisionTokenizer(config)
        self.vision_tokenizer.maybe_freeze_vision()
        self.context_backbone = DiTBackbone(config.fusion_hidden_dim, config.dit_num_layers, config.dit_num_heads)
        self.hierarchical_planner = HierarchicalPlanner(config)
        self.flow_action_head = FlowActionHead(config)
        self.chunk_verifier = ChunkVerifier(config)

        self.timestep_embedding = nn.Embedding(config.max_timestep, config.fusion_hidden_dim)
        critic_in = config.fusion_hidden_dim + config.action_dim
        self.critic_network = nn.Sequential(
            nn.Linear(critic_in, config.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.critic_hidden_dim, config.critic_hidden_dim),
            nn.SiLU(),
            nn.Linear(config.critic_hidden_dim, 1),
        )

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, *args: Any, **kwargs: Any) -> "PhaseQFlowPolicy":
        """Instantiate policy from a serialized configuration location."""
        config = cls.config_class.from_pretrained(pretrained_model_name_or_path)
        _ = args
        return cls(config=config, **kwargs)

    def to_config_dict(self) -> Dict[str, Any]:
        """Return configuration as a plain dictionary."""
        return asdict(self.config)

    def _extract_actions(self, batch: Dict[str, Any]) -> Optional[torch.Tensor]:
        """Extract supervision actions from common batch key aliases."""
        for key in ("action", "actions", "target_action"):
            value = batch.get(key)
            if isinstance(value, torch.Tensor):
                return value.float()
        return None

    def _extract_inputs(self, batch: Dict[str, Any]) -> Dict[str, Optional[torch.Tensor]]:
        """Extract explicit multimodal inputs from batch/obs dictionaries."""
        obs = batch.get("obs", batch)
        images = obs.get("images", obs.get("obs_images", obs.get("observation.images")))
        states = obs.get("states", obs.get("obs_states", obs.get("observation.state")))
        language = obs.get("language", obs.get("task_descriptor"))
        history = obs.get("history")
        masks = obs.get("masks", batch.get("masks"))
        if images is None or states is None:
            raise KeyError("Expected explicit multimodal inputs: images, states, language, history, masks")
        return {
            "images": images.float(),
            "states": states.float(),
            "language": None if language is None else language.float(),
            "history": None if history is None else history.float(),
            "masks": None if masks is None else masks.float(),
        }

    def forward(
        self,
        images: torch.Tensor,
        states: torch.Tensor,
        language: Optional[torch.Tensor] = None,
        history: Optional[torch.Tensor] = None,
        masks: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        phase_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Run four-layer forward pass and return intermediate diagnostics."""
        tok = self.vision_tokenizer(images=images, states=states, language=language, history=history, masks=masks)
        context = self.context_backbone(tok["context_tokens"])
        fused_obs = context.mean(dim=1) + tok["fused"]

        if timestep is None:
            timestep = torch.zeros(fused_obs.size(0), dtype=torch.long, device=fused_obs.device)
        timestep = torch.clamp(timestep.long(), 0, self.config.max_timestep - 1)
        fused_obs = fused_obs + self.timestep_embedding(timestep)

        plan = self.hierarchical_planner(fused_obs=fused_obs, phase_labels=phase_labels)
        flow = self.flow_action_head(fused_obs=fused_obs, phase_embed=plan["phase_embed"], skill_latent=plan["skill_latent"])
        verify = self.chunk_verifier(fused_obs=fused_obs, predicted_action=flow["action_pred"], phase_embed=plan["phase_embed"])

        return {
            **tok,
            **plan,
            **flow,
            **verify,
            "encoded_obs": fused_obs,
        }

    def predict_action(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """Predict actions from a training/evaluation batch dictionary."""
        inputs = self._extract_inputs(batch)
        timestep = batch.get("timestep")
        phase_labels = batch.get("phase_id")
        return self.forward(**inputs, timestep=timestep, phase_labels=phase_labels)

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute imitation, flow, phase, and verifier training objectives."""
        preds = self.predict_action(batch)
        actions = self._extract_actions(batch)
        if actions is None:
            raise KeyError("compute_loss requires action/actions/target_action in batch")

        flow_loss = F.mse_loss(preds["action_pred"], actions)
        smoothness = (preds["action_pred"][1:] - preds["action_pred"][:-1]).pow(2).mean() if preds["action_pred"].shape[0] > 1 else 0.0

        obs_feat = preds["encoded_obs"].detach()
        q_values = self.critic_network(torch.cat([obs_feat, actions.detach()], dim=-1)).squeeze(-1)
        per_sample = F.mse_loss(preds["action_pred"], actions, reduction="none").mean(dim=-1)

        if self.config.use_value_guided_weight:
            weights = torch.softmax(self.config.value_weight_beta * q_values, dim=0)
            weights = weights * weights.numel()
            imitation_loss = (per_sample * weights.detach()).mean()
        else:
            imitation_loss = per_sample.mean()

        verifier_targets = torch.ones_like(preds["chunk_confidence"])
        verifier_loss = F.binary_cross_entropy(preds["chunk_confidence"], verifier_targets)

        phase_labels = batch.get("phase_id")
        if isinstance(phase_labels, torch.Tensor) and torch.all(phase_labels >= 0):
            phase_loss = F.cross_entropy(preds["phase_logits"], phase_labels.long())
        else:
            phase_loss = torch.zeros((), device=actions.device)

        loss = imitation_loss + 0.5 * flow_loss + 0.05 * smoothness + 0.1 * verifier_loss + 0.1 * phase_loss

        base_loss = None
        if self.base_policy is not None and hasattr(self.base_policy, "compute_loss"):
            base_out = self.base_policy.compute_loss(batch)  # type: ignore[attr-defined]
            if isinstance(base_out, dict) and "loss" in base_out:
                base_loss = base_out["loss"]
            elif isinstance(base_out, torch.Tensor):
                base_loss = base_out
        if base_loss is not None:
            if base_loss.ndim > 0:
                base_loss = base_loss.mean()
            loss = loss + self.config.base_loss_weight * base_loss

        return loss

    def update_critic(self, obs_feat: torch.Tensor, actions: torch.Tensor, target_q: torch.Tensor) -> torch.Tensor:
        """Update critic via MSE regression against target Q values."""
        critic_in = torch.cat([obs_feat, actions], dim=-1)
        pred_q = self.critic_network(critic_in).squeeze(-1)
        return F.mse_loss(pred_q, target_q)
