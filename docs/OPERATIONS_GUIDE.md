# Operations and Usage Guide

This document explains how to run the repository end-to-end and how to choose
between local debugging and cloud-scale training.

## 1. Environment setup

```bash
conda env create -f environment.yml
conda activate lerobot_env
pip install -r requirements.txt
pip install -e ./lerobot_policy_phaseqflow
```

If you already manage environments with `venv` or `uv`, keep the same install
order: dependencies first, then editable policy package.

## 2. Data inspection (recommended before training)

```bash
python scripts/inspect_dataset.py --dataset HuggingFaceVLA/smol-libero --n 5
python scripts/compute_episode_lengths.py \
  --dataset HuggingFaceVLA/smol-libero \
  --out artifacts/episode_lengths/smol_libero_lengths.json
```

Why this matters:
- Catches schema mismatches early.
- Helps tune action chunk length and max sequence horizon.
- Provides evidence for data-quality assumptions.

## 3. Local training recipes

### PhaseQFlow local run

```bash
bash configs/local/phaseqflow_local.sh
```

### Diffusion baseline local run

```bash
bash configs/local/diffusion_baseline_local.sh
```

### SmolVLA LoRA local run

```bash
bash configs/local/smolvla_lora_local.sh
```

Use local runs to validate config changes and avoid expensive cloud cycles.

## 4. Cloud training recipes

### PhaseQFlow cloud (accelerate)

```bash
bash configs/cloud/phaseqflow_cloud_accelerate.sh
```

### SmolVLA LoRA cloud (accelerate)

```bash
bash configs/cloud/smolvla_lora_cloud_accelerate.sh
```

Before launching, verify:
- Correct `CUDA_VISIBLE_DEVICES` and world size.
- Batch size per device and gradient accumulation settings.
- Output directory naming for reproducibility.

## 5. Evaluation and benchmarking

### Run LIBERO evaluation

```bash
bash scripts/run_eval_libero.sh \
  outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model
```

### PushT sanity evaluation

```bash
bash scripts/run_eval_pusht_sanity.sh \
  outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model
```

### Latency benchmark

```bash
python scripts/benchmark_latency.py --help
python scripts/edited_benchmark_latency.py --help
```

Track both policy quality metrics (success/reward) and runtime metrics
(latency/throughput) to support deployment-oriented decisions.

## 6. Checkpoint export

```bash
python scripts/export_checkpoint.py \
  --src outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \
  --dst exports/phaseqflow_smol_local
```

Use exported artifacts for model sharing, inference serving experiments, or
further fine-tuning in downstream pipelines.

## 7. Troubleshooting checklist

- Confirm the policy package is installed with `pip show lerobot_policy_phaseqflow`.
- Confirm required dataset paths and network access for Hugging Face resources.
- Start with local configs before cloud configs after any major code change.
- Keep one run = one output directory to simplify experiment tracking.
