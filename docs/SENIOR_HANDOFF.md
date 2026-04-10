# Senior Handoff Checklist

This note is for continuing the current project on a stronger machine.

## Scope

- Current project scope: Questions 1 to 3 only.
- Immediate priorities:
  - refine Question 2 with the MPC baseline in `q2_mpc.py`
  - continue training Question 3 with the hierarchical PPO pipeline in `q3_sb3.py`
- Question 1 is already stable enough and does not need repeated reruns.

## Repository Setup

Clone and install:

```bash
git clone https://github.com/lixiang-A/5G_Optimization.git
cd 5G_Optimization
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Optional environment checks:

```bash
./.venv/bin/python -c "import torch; print('cuda', torch.cuda.is_available())"
./.venv/bin/python q2_mpc.py --help
./.venv/bin/python q3_sb3.py train-slice --help
```

If the machine has an NVIDIA GPU, use `--device cuda` for Question 3 training. Otherwise use `--device cpu`.

## Files That May Need To Be Sent Separately

The repository ignores `outputs/`, so current checkpoints are not in git.

If continuing from the latest local training state is desired, ask Liz to send:

- `outputs/q3_sb3/q3_slice_ppo.zip`
- `outputs/q3_sb3/q3_power_ppo.zip`

If these files are not available, Question 3 can still be retrained from scratch.

## Task A: Refine Question 2 MPC Baseline

Main script:

- `q2_mpc.py`

Main goal:

- compare different MPC lookahead depths
- record objective, delay, drops, and runtime
- keep the current model structure unchanged

Recommended runs:

```bash
./.venv/bin/python q2_mpc.py --lookahead 1 --json-out outputs/q2_mpc/lookahead1.json
./.venv/bin/python q2_mpc.py --lookahead 2 --json-out outputs/q2_mpc/lookahead2.json
./.venv/bin/python q2_mpc.py --lookahead 3 --json-out outputs/q2_mpc/lookahead3.json
```

Current reference:

- `lookahead=2` has been verified locally and is feasible on CPU, but already non-trivial in runtime.

Please return:

- the three JSON files under `outputs/q2_mpc/`
- the terminal runtime summary for each run
- a short note if `lookahead=3` is still slow, and whether `lookahead=4` seems realistic on the stronger machine

## Task B: Continue Question 3 Hierarchical PPO Training

Main scripts:

- `q3_hierarchical_rl.py`
- `q3_sb3.py`

Main goal:

- continue improving the current hierarchical PPO result
- keep the training curves and metrics files
- preserve old checkpoints by writing new stage files instead of overwriting immediately

### Option 1: Continue From Existing Checkpoints

If Liz sends the two checkpoint `.zip` files, use these commands first.

Continue slice-layer PPO:

```bash
./.venv/bin/python q3_sb3.py train-slice \
  --device cuda \
  --init-model outputs/q3_sb3/q3_slice_ppo.zip \
  --total-timesteps 20000 \
  --eval-freq 1000 \
  --eval-episodes 2 \
  --model-out outputs/q3_sb3/q3_slice_ppo_continue_20k.zip \
  --metrics-out outputs/q3_sb3/q3_slice_ppo_continue_20k_metrics.json \
  --plot-out outputs/q3_sb3/q3_slice_ppo_continue_20k_curves.png
```

Continue power-layer PPO:

```bash
./.venv/bin/python q3_sb3.py train-power \
  --device cuda \
  --slice-mode model \
  --slice-model-in outputs/q3_sb3/q3_slice_ppo_continue_20k.zip \
  --init-model outputs/q3_sb3/q3_power_ppo.zip \
  --total-timesteps 20000 \
  --eval-freq 1000 \
  --eval-episodes 2 \
  --model-out outputs/q3_sb3/q3_power_ppo_continue_20k.zip \
  --metrics-out outputs/q3_sb3/q3_power_ppo_continue_20k_metrics.json \
  --plot-out outputs/q3_sb3/q3_power_ppo_continue_20k_curves.png
```

Evaluate the combined controller:

```bash
./.venv/bin/python q3_sb3.py evaluate \
  --slice-mode model \
  --slice-model-in outputs/q3_sb3/q3_slice_ppo_continue_20k.zip \
  --power-model-in outputs/q3_sb3/q3_power_ppo_continue_20k.zip \
  --eval-episodes 3 \
  --metrics-out outputs/q3_sb3/q3_combined_eval_continue_20k.json
```

### Option 2: Retrain From Scratch

If the checkpoint files are unavailable, run from scratch:

```bash
./.venv/bin/python q3_sb3.py train-slice \
  --device cuda \
  --total-timesteps 10000 \
  --eval-freq 1000 \
  --eval-episodes 2 \
  --model-out outputs/q3_sb3/q3_slice_ppo_fresh_10k.zip \
  --metrics-out outputs/q3_sb3/q3_slice_ppo_fresh_10k_metrics.json \
  --plot-out outputs/q3_sb3/q3_slice_ppo_fresh_10k_curves.png
```

```bash
./.venv/bin/python q3_sb3.py train-power \
  --device cuda \
  --slice-mode model \
  --slice-model-in outputs/q3_sb3/q3_slice_ppo_fresh_10k.zip \
  --total-timesteps 10000 \
  --eval-freq 1000 \
  --eval-episodes 2 \
  --model-out outputs/q3_sb3/q3_power_ppo_fresh_10k.zip \
  --metrics-out outputs/q3_sb3/q3_power_ppo_fresh_10k_metrics.json \
  --plot-out outputs/q3_sb3/q3_power_ppo_fresh_10k_curves.png
```

```bash
./.venv/bin/python q3_sb3.py evaluate \
  --slice-mode model \
  --slice-model-in outputs/q3_sb3/q3_slice_ppo_fresh_10k.zip \
  --power-model-in outputs/q3_sb3/q3_power_ppo_fresh_10k.zip \
  --eval-episodes 3 \
  --metrics-out outputs/q3_sb3/q3_combined_eval_fresh_10k.json
```

## What To Return

Please send back these files after the run:

- `outputs/q2_mpc/lookahead1.json`
- `outputs/q2_mpc/lookahead2.json`
- `outputs/q2_mpc/lookahead3.json`
- latest Question 3 model `.zip` files
- latest Question 3 metrics `.json` files
- latest Question 3 convergence `.png` files

Also please report:

- GPU model or CPU model used
- total runtime of each Question 3 stage
- whether the objective improves over the current tracked value `0.48511000199064225`
- any dependency or runtime errors

## Not In Scope For This Round

- Question 4 and Question 5
- rewriting the Question 2 model from scratch
- replacing the Question 2 MPC baseline with RL as the main answer

The intended interpretation is:

- Question 2: MPC is the main rigorous baseline, RL is only a later supplementary direction
- Question 3: hierarchical RL is the current main optimization target
