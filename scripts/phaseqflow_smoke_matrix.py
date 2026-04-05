"""Executable PhaseQFlow smoke/regression matrix.

Covers:
1) minimal training smoke,
2) forward/backward,
3) shape contract,
4) config save/load regression.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_dir = repo_root / "lerobot_policy_phaseqflow" / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def main() -> int:
    _ensure_src_on_path()

    try:
        import torch
    except ModuleNotFoundError:
        print("[BLOCKED] torch is not installed; cannot execute runtime smoke matrix.")
        return 2

    try:
        import torchvision  # noqa: F401
    except ModuleNotFoundError:
        print("[BLOCKED] torchvision is not installed; cannot execute processor/runtime matrix.")
        return 2

    from lerobot_policy_phaseqflow.configuration_phaseqflow import PhaseQFlowConfig
    from lerobot_policy_phaseqflow.modeling_phaseqflow import PhaseQFlowPolicy

    results: list[tuple[str, str, str]] = []

    torch.manual_seed(0)
    config = PhaseQFlowConfig(
        dit_hidden_dim=64,
        dit_num_layers=2,
        dit_num_heads=4,
        action_dim=8,
        latent_dim=4,
        num_skills=8,
    )
    model = PhaseQFlowPolicy(config)
    model.train()

    batch_size = 6
    obs_dim = 12
    batch = {
        "obs_states": torch.randn(batch_size, obs_dim),
        "actions": torch.randn(batch_size, config.action_dim),
        "timestep": torch.randint(0, 128, (batch_size,)),
    }

    # 1) forward pass / shape contract
    try:
        out = model.predict_action(batch)
        assert out["action_pred"].shape == (batch_size, config.action_dim)
        assert out["latent_action_pred"].shape == (batch_size, config.latent_dim)
        assert out["encoded_obs"].shape == (batch_size, config.dit_hidden_dim)
        results.append(("shape_contract", "PASS", "action/latent/encoded shapes match config"))
    except Exception as exc:  # pragma: no cover - smoke script
        results.append(("shape_contract", "FAIL", repr(exc)))

    # 2) forward/backward and minimal training smoke
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
        opt.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        loss.backward()
        grad_ok = any(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
        assert grad_ok, "no finite gradients found"
        opt.step()
        results.append(("forward_backward", "PASS", f"loss={float(loss.detach().cpu()):.6f}"))
        results.append(("training_smoke", "PASS", "one optimizer step finished"))
    except Exception as exc:  # pragma: no cover - smoke script
        results.append(("forward_backward", "FAIL", repr(exc)))
        results.append(("training_smoke", "FAIL", repr(exc)))

    # 3) save/load regression (config + state dict)
    try:
        with tempfile.TemporaryDirectory() as td:
            td_path = Path(td)
            cfg_path = Path(config.save_pretrained(str(td_path)))
            sd_path = td_path / "model.pt"
            torch.save(model.state_dict(), sd_path)

            reloaded_cfg = PhaseQFlowConfig.from_pretrained(str(cfg_path))
            reloaded_model = PhaseQFlowPolicy(reloaded_cfg)
            reloaded_model.load_state_dict(torch.load(sd_path, map_location="cpu"))

            # regression sanity: config survives round trip
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
            assert raw["latent_dim"] == config.latent_dim
            assert reloaded_cfg.num_skills == config.num_skills
        results.append(("save_load_regression", "PASS", "config/state_dict round-trip succeeded"))
    except Exception as exc:  # pragma: no cover - smoke script
        results.append(("save_load_regression", "FAIL", repr(exc)))

    print("\n=== PhaseQFlow Smoke Matrix ===")
    width = max(len(name) for name, _, _ in results)
    for name, status, detail in results:
        print(f"{name.ljust(width)} | {status:<4} | {detail}")

    has_fail = any(status == "FAIL" for _, status, _ in results)
    return 1 if has_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
