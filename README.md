# LeRobot + LIBERO PhaseQFlow

This repository is a research-and-engineering template for training and evaluating
**generative imitation learning** policies on
[LeRobot](https://huggingface.co/docs/lerobot/index), with a focus on
[LIBERO](https://libero-project.github.io/) long-horizon manipulation tasks.

The core contribution is **PhaseQFlow++**: a four-layer multimodal hierarchical
policy that extends the LeRobot policy plugin architecture.
The repository also includes baseline diffusion configuration, SmolVLA LoRA
fine-tuning examples, and utility scripts for dataset inspection, evaluation,
latency benchmarking, and checkpoint export.

## Project goals

- Provide a reproducible LeRobot-compatible codebase for LIBERO experiments.
- Upgrade monolithic policies to explicit multimodal, hierarchical, and
  closed-loop control modules.
- Improve long-horizon policy learning with discrete-continuous phase/skill
  latents instead of phase-as-feature shortcuts.
- Improve sample efficiency with value-guided imitation and stage-wise training.
- Offer practical scripts for local debugging and cloud-scale training.

## Latest model architecture (PhaseQFlow++)

The policy is now organized into **four explicit layers**:

1. **Multimodal tokenization layer** (`VisionTokenizer`):
   - Explicit inputs: `images, states, language, history, masks`.
   - Asymmetric cross-attention (`state/history` as query; `vision` as key/value).
   - Uncertainty gate to balance proprioception and visual evidence.
2. **Hierarchical latent planner** (`HierarchicalPlanner`):
   - Discrete phase latent `z^(p)` and continuous skill latent `z^(s)`.
   - Weak-supervision modes: `manual | latent | hybrid`.
3. **Conditional flow action generator** (`FlowActionHead`):
   - Phase-conditioned continuous flow field for action chunk generation.
4. **Closed-loop verifier + replanning trigger** (`ChunkVerifier`):
   - Estimates chunk confidence and phase drift.
   - Emits `should_replan` for online truncation/regeneration.

## Latest training strategy (4-stage curriculum)

Training is now split into stage-wise configs:

- `configs/pretrain_multimodal.yaml`: multimodal representation alignment.
- `configs/train_latent.yaml`: phase/skill latent learning.
- `configs/train_flow.yaml`: conditional flow imitation.
- `configs/finetune_closedloop.yaml`: closed-loop verifier and correction tuning.

This avoids single-stage BC collapse and makes ablation studies explicit.

## Latest data processing strategy

The processor now emits an explicit multimodal observation dictionary:

- `obs.images`
- `obs.states`
- `obs.language`
- `obs.history`
- `obs.masks`

Top-level compatibility keys are retained for legacy integrations, but policy
forwarding now prioritizes explicit multimodal fields for clarity and
reproducibility.

## Repository structure

| Path | Purpose |
|---|---|
| `configs/local/` | Local launch scripts for quick experiments and debugging. |
| `configs/cloud/` | Cloud/multi-GPU launch scripts using `accelerate`. |
| `scripts/` | Utilities for evaluation, dataset analysis, checkpoint export, and latency benchmarking. |
| `docs/` | Project abstract, CV summary bullets, and operation-focused documentation. |
| `lerobot_policy_phaseqflow/` | Installable LeRobot policy package implementing PhaseQFlow. |
| `environment.yml` / `requirements.txt` | Environment and dependency definitions. |

## Installation

> Assumption: you already installed a compatible Python environment and LeRobot.

```bash
# Optional: create conda environment
conda env create -f environment.yml
conda activate lerobot_env

# Install Python dependencies
pip install -r requirements.txt

# Install PhaseQFlow policy package in editable mode
pip install -e ./lerobot_policy_phaseqflow
```

## Typical workflow

### 1) Inspect dataset quality and length distribution

```bash
python scripts/inspect_dataset.py --dataset HuggingFaceVLA/smol-libero --n 5
python scripts/compute_episode_lengths.py \
  --dataset HuggingFaceVLA/smol-libero \
  --out artifacts/episode_lengths/smol_libero_lengths.json
```

### 2) Run local PhaseQFlow training

```bash
bash configs/local/phaseqflow_local.sh
```

### 3) Evaluate the trained checkpoint

```bash
bash scripts/run_eval_libero.sh \
  outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model
```

### 4) Export model artifacts for sharing/deployment

```bash
python scripts/export_checkpoint.py \
  --src outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \
  --dst exports/phaseqflow_smol_local
```

## Baselines and extensions

- Diffusion baseline (local): `configs/local/diffusion_baseline_local.sh`
- SmolVLA LoRA fine-tuning (local): `configs/local/smolvla_lora_local.sh`
- PhaseQFlow cloud training: `configs/cloud/phaseqflow_cloud_accelerate.sh`
- SmolVLA LoRA cloud training: `configs/cloud/smolvla_lora_cloud_accelerate.sh`

## Documentation map

- Project abstract: `docs/PROJECT_ABSTRACT.md`
- CV-ready highlights: `docs/CV_BULLETS.md`
- Operations and usage guide: `docs/OPERATIONS_GUIDE.md`

## License

This repository is released under the Apache 2.0 License. See `LICENSE` for details.
