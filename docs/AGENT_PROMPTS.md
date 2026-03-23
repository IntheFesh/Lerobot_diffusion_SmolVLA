# Agent prompt for automatic repository construction

You are a code generation and engineering execution agent working inside PyCharm on WSL2.  Your tasks are:

1. **Create the repository structure**: Make a directory named `lerobot-libero-phaseqflow` and create all files and subdirectories as specified in the provided file tree.  Preserve indentation and line endings exactly as given.

2. **Write file contents verbatim**: For each file, insert the provided content.  Where the specification includes instructions to adapt to a specific LeRobot version, you may need to adjust field names or shapes accordingly.  Comments labelled “可能需按 LeRobot 版本调整” mark places where adjustments may be required.

3. **Install dependencies**: Reuse the currently installed LeRobot environment if available.  Then run:
   ```bash
   pip install -r requirements.txt
   pip install -e ./lerobot_policy_phaseqflow
   ```

4. **Run sanity checks**: Execute the following commands to verify basic functionality:
   ```bash
   python scripts/inspect_dataset.py --dataset HuggingFaceVLA/smol-libero --n 5
   python scripts/compute_episode_lengths.py --dataset HuggingFaceVLA/smol-libero --out artifacts/episode_lengths/smol_libero_lengths.json
   ```

5. **Run local training and evaluation**: Invoke:
   ```bash
   bash configs/local/phaseqflow_local.sh
   bash scripts/run_eval_libero.sh outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model
   ```

6. **Handle errors**: If any command fails, print the stack trace.  Investigate whether the error is due to a field name mismatch (e.g. `observation.images.*` vs `observation.image`) and modify the code at the indicated adjustment points.  Retry until the training and evaluation commands succeed.

7. **Export the checkpoint**: Finally, run:
   ```bash
   python scripts/export_checkpoint.py --src outputs/train/phaseqflow_smol_local/checkpoints/last/pretrained_model --dst exports/phaseqflow_smol_local
   ```

8. **Report**: Summarize whether the training ran successfully, whether evaluation produced logs, and whether the export directory exists.

This prompt should be passed to your PyCharm agent to automate the creation and execution of the PhaseQFlow project.
