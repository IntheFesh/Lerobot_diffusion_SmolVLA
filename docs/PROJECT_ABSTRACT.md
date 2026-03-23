# Project Abstract: PhaseQFlow—Phase‑Aware and Quality‑Weighted Generative Imitation Learning for LIBERO

This project builds a simulation‑first research pipeline on the LeRobot and LIBERO benchmarks, designed for PhD applications requiring reproducible, extensible, and engineerable deliverables.  After reproducing the Diffusion Policy imitation baseline, we propose **PhaseQFlow**, a phase‑aware and quality‑weighted generative action‑chunk policy.  PhaseQFlow injects two forms of structured priors into the generative model: discrete progress phases and per‑sample quality weights, enabling more stable training and improved performance on long‑horizon tasks under limited data.

Key contributions:

1. **Phase‑Aware Conditioning**: Trajectory progress `(frame_index / episode_length)` is discretized into K phases and embedded via a learnable phase embedding.  The embedding conditions the generative model so that different behaviour modes can emerge for early versus late stages of a task.
2. **Quality‑Weighted Imitation**: Sample quality weights are computed from action smoothness (jerk).  By scaling the imitation loss with these weights, smoother and more consistent expert segments contribute more to learning, improving convergence in low‑data regimes.
3. **Flow Matching Objective**: Replacing traditional diffusion denoising with a flow matching / rectified flow target yields faster sampling at inference time and aligns with 2025–2026 trends in generative robotic policies.  The design leaves room for Real‑Time Chunking and streaming inference stacks.

From an engineering perspective, the project conforms to LeRobot’s policy plug‑in and processor pipeline conventions.  It offers turnkey training and evaluation scripts—both for local debugging on the Smol‑LIBERO subset and for full‑scale cloud training.  We report success rate, average reward, and inference latency.  Beyond the main line, we provide a SmolVLA PEFT/LoRA fine‑tuning and a phase prompt/adapter prototype as an extension path toward vision‑language action (VLA) and reinforcement learning.
