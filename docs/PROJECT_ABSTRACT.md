# Project Abstract: PhaseQFlow for Long-Horizon LIBERO Manipulation

## Summary

This project builds a simulation-first pipeline for embodied policy research on
LeRobot and LIBERO. Starting from a diffusion-policy reproduction baseline, it
introduces **PhaseQFlow**, a generative action-chunk policy that combines:

1. **Phase-aware conditioning** for trajectory progress awareness.
2. **Quality-weighted imitation learning** for robust low-data training.
3. **Flow-matching objectives** for efficient generation and scalable inference.

The implementation follows LeRobot plugin conventions so the policy can be
trained and evaluated through standard `lerobot-train` and `lerobot-eval`
workflows.

## Technical contributions

### 1) Phase-aware conditioning

Trajectory progress is represented as a normalized ratio:

`phase_progress = frame_index / episode_length`

Progress values are discretized into `K` phase bins and embedded with a
learnable phase embedding table. The model uses this embedding as an additional
conditioning signal, enabling different behavior modes for early/mid/late task
stages.

### 2) Quality-weighted imitation learning

Each training sample receives a quality weight derived from motion smoothness
(e.g., jerk-based metrics). The loss is scaled by this weight so cleaner,
consistent expert segments contribute more strongly to optimization.

This design aims to improve convergence stability and final success rates,
especially when demonstrations are limited or heterogeneous.

### 3) Flow matching policy objective

Instead of relying only on diffusion denoising, PhaseQFlow uses a flow-matching
objective for action-chunk generation. This supports faster and more stable
sampling at inference time and aligns with recent trends in generative robotics.

## Engineering outcomes

- Reproducible repository structure with local and cloud launch scripts.
- Clear separation between policy package and experiment operations.
- Utility scripts for dataset diagnostics, checkpoint export, evaluation, and
  inference latency benchmarking.
- Extension path toward SmolVLA LoRA adaptation and future online RL integration.

## Evaluation focus

The project emphasizes practical research metrics:

- Task success rate on LIBERO suites.
- Average reward and rollout robustness.
- Inference latency for deployment readiness.
