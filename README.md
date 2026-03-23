# LeRobot + LIBERO PhaseQFlow

This repository provides a reproducible research template for developing
generative imitation learning policies on the [LeRobot](https://huggingface.co/docs/lerobot/index) platform.  It
focuses on LIBERO, a collection of long‑horizon simulation tasks for
robot manipulation, and introduces **PhaseQFlow**, a phase‑aware and
quality‑weighted policy that builds on the diffusion and flow matching
paradigms.

## Repository overview

The project is divided into two parts: a top‑level directory with
configuration, scripts, documentation, and installation files, and a
`lerobot_policy_phaseqflow` subpackage containing the actual policy
implementation.  The policy package registers a new `phaseqflow`
policy with LeRobot via an entry point so that you can invoke it
directly from `lerobot-train` and `lerobot-eval`.

Key directories:

| Directory/file | Description |
|---|---|
| `environment.yml` | Conda environment specification for installing basic dependencies. |
| `requirements.txt` | Additional Python packages used in scripts and the policy implementation. |
| `configs/` | Shell scripts for launching local and cloud training jobs. |
| `scripts/` | Utilities for inspecting datasets, computing episode lengths, benchmarking latency, exporting checkpoints, and running evaluations. |
| `docs/` | Project abstracts, CV bullets.|
| `lerobot_policy_phaseqflow/` | The Python package implementing the PhaseQFlow policy plug‑in.  See its README for details. |

## Quickstart (local debugging)

The following steps assume you already have LeRobot and its
dependencies installed in your Python environment (e.g. via
`pip install lerobot`).  If not, install LeRobot according to the
[official instructions](https://huggingface.co/docs/lerobot/installation).

1. **Create the conda environment** (optional)::

   ```bash
   conda env create -f environment.yml
   conda activate lerobot_env
   ```

2. **Install requirements and the PhaseQFlow policy**::

   ```bash
   pip install -r requirements.txt
   pip install -e ./lerobot_policy_phaseqflow
   ```

3. **Inspect the dataset** (optional)::

   ```bash
   python scripts/inspect_dataset.py --dataset HuggingFaceVLA/smol-libero --n 5
   python scripts/compute_episode_lengths.py \
     --dataset HuggingFaceVLA/smol-libero \
     --out artifacts/episode_lengths/smol_libero_lengths.json
   ```

4. **Run a small PhaseQFlow training job**::

   ```bash
   bash configs/local/phaseqflow_local.sh
   ```

5. **Evaluate the trained policy on LIBERO**::

   ```bash
   bash scripts/run_eval_libero.sh outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model
   ```

6. **Export the checkpoint** (optional)::

   ```bash
   python scripts/export_checkpoint.py \
     --src outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model \
     --dst exports/phaseqflow_smol_local
   ```

Refer to `configs/local/diffusion_baseline_local.sh` for a baseline diffusion run
and `configs/local/smolvla_lora_local.sh` for a lightweight SmolVLA
PEFT/LoRA fine‑tuning example.

## Cloud training

For full LIBERO training, you can launch a multi‑GPU job using the
`accelerate` library.  See `configs/cloud/phaseqflow_cloud_accelerate.sh`
for an example command.  Adjust the batch size, number of steps, and
`accelerate` settings to your hardware.  See the LeRobot
documentation on [multi‑GPU training](https://huggingface.co/docs/lerobot/multi_gpu_training)
for details.

## Documentation and application materials

In the `docs/` directory you will find:

* **Project abstracts** (in Chinese and English) summarizing the
  research contributions of PhaseQFlow.
* **CV bullet points** describing the project in a concise manner.
* **Personal statement paragraphs** explaining your research goals and
  how this project fits into your PhD aspirations.
* **Agent prompts** for automating repository creation in your IDE.

Use these materials to prepare application documents or to explain
your work to collaborators.

## License

This repository is released under the Apache 2.0 license.  See the
`LICENSE` file for details.  Portions of the code are inspired by
examples and documentation from the LeRobot project.
