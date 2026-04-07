# Project Abstract: PhaseQFlow++ for Long-Horizon LIBERO Manipulation

## Summary

This project builds a simulation-first pipeline for embodied policy research on
LeRobot and LIBERO. Starting from a diffusion-policy reproduction baseline, it
introduces **PhaseQFlow++**, a four-layer multimodal generative chunk policy
that combines:

1. **Multimodal tokenization with asymmetric cross-attention**.
2. **Hierarchical latent phase/skill planning**.
3. **Phase-conditioned flow action generation**.
4. **Closed-loop verification and replanning**.

The implementation follows LeRobot plugin conventions so the policy can be
trained and evaluated through standard `lerobot-train` and `lerobot-eval`
workflows.

## Technical contributions

### 1) Multimodal tokenization with control-centric visual fusion

At each control step, observations are modeled as:

`x_t = {V_t, S_t, L, H_t}`

where `V_t` is visual tokens, `S_t` is robot state, `L` is language/task token,
and `H_t` is history. Instead of early-fusion concatenation, the policy uses
asymmetric cross-attention with state/history queries over vision keys/values,
plus an uncertainty gate that adaptively weighs vision versus proprioception.

### 2) Hierarchical latent planner

Phase is represented as mixed latent variables:

- Discrete phase: `z_t^(p) in {1, ..., K}`
- Continuous skill style/subgoal: `z_t^(s) in R^d`

This turns phase into a true high-level control variable for chunk structure,
action mode, and replanning timing, instead of a passive appended feature.

### 3) Conditional flow action generation

Given `x_<=t` and latent planner outputs, the action head integrates a
phase-conditioned continuous flow field and decodes final latent state into an
action chunk. This keeps multimodal generation benefits while remaining lighter
than large diffusion stacks.

### 4) Closed-loop chunk verification

A lightweight verifier predicts chunk confidence and phase drift during
execution. If mismatch is detected, it triggers early chunk truncation and
replanning, upgrading the method from open-loop chunk prediction to closed-loop
planning-and-verification.

## Training strategy

The project now uses a four-stage curriculum:

1. Multimodal alignment pretraining.
2. Phase/skill latent learning (manual, latent-only, or hybrid weak supervision).
3. Flow action imitation with structure-aware regularization.
4. Closed-loop correction fine-tuning with rollout mismatch handling.

## Engineering outcomes

- Reproducible repository structure with local and cloud launch scripts.
- Clear separation between policy package and experiment operations.
- Explicit multimodal processor-to-policy data path (`images/states/language/history/masks`).
- Utility scripts for dataset diagnostics, checkpoint export, evaluation, and
  inference latency benchmarking.
- Extension path toward SmolVLA LoRA adaptation and future online RL integration.

## Evaluation focus

The project emphasizes practical research metrics:

- Task success rate on LIBERO suites.
- Average reward and rollout robustness.
- Inference latency for deployment readiness.
