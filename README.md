# 5G Network Slicing Optimization

Paper-oriented rebuild of the first three contest questions into a single optimization and reinforcement learning project.

## At A Glance

- Question 1: static single-base-station RB slicing as an exact small-scale integer allocation problem.
- Question 2: dynamic single-base-station scheduling as a finite-horizon MPC pipeline in `q2_mpc.py`.
- Question 3: multi-base-station slicing and power control under inter-cell interference as a hierarchical RL pipeline in `q3_hierarchical_rl.py` and `q3_sb3.py`.
- All experiments use the original repository Excel data in `channel_data等2个文件/`, `channel_data等2个文件(1)/`, and `BS2等5个文件/`.

## Featured Result

- Best tracked Q3 hierarchical PPO run reaches `objective = 0.4995` in combined evaluation with `seed = 17`.
- Three `10k`-timestep Q3 runs with `seed = 7 / 17 / 27` yield mean combined `objective = 0.4960 ± 0.0031`.
- The earlier numpy hierarchical actor-critic baseline reaches `best_eval_objective = 0.4553` in a short run.
- The current Q2 MPC baseline reaches `objective = 0.8986` at `lookahead = 2`, while `lookahead = 3` is much slower without improvement.

![Q3 slice convergence](docs/assets/q3_slice_seed17_convergence.png)

Full metrics, evidence sources, and caveats are summarized in [docs/Q3_RESULTS.md](docs/Q3_RESULTS.md). The paper-writing base is in [docs/PAPER_DRAFT.md](docs/PAPER_DRAFT.md).

## Why This Repository Matters

- It keeps the contest's real constraints instead of replacing them with a toy RL benchmark.
- It separates modeling assumptions, official evaluation logic, and training proxies.
- It shows a clean research progression: exact optimization -> MPC -> hierarchical RL.
- It is reproducible from the project root with explicit scripts, data sources, and tracked showcase assets.

## Repository Map

- `背景信息/`: contest statement, appendix, modeling notes, references, and earlier materials.
- `channel_data等2个文件/`: Question 1 data and original materials.
- `channel_data等2个文件(1)/`: Question 2 data.
- `BS2等5个文件/`: Question 3 data.
- `q2_mpc.py`: finite-horizon MPC implementation for Question 2.
- `q3_hierarchical_rl.py`: Question 3 environment and lightweight numpy RL baseline.
- `q3_sb3.py`: Question 3 SB3/PyTorch hierarchical PPO entry point.
- `docs/Q3_RESULTS.md`: tracked result snapshot for GitHub display.
- `docs/PAPER_DRAFT.md`: current writing base for the paper's modeling and result sections.
- `docs/assets/`: tracked images that should stay visible on GitHub even though `outputs/` is ignored.

## Important Modeling Notes

- This repository currently focuses on Questions 1 to 3 only.
- Question 3 follows the contest's `100 ms` resource reconfiguration cycle and `1 ms` service execution cycle.
- In the current Question 3 implementation, the serving BS is assigned as the nearest micro base station at task arrival time. This is an explicit modeling assumption because access optimization is outside the current scope.
- `outputs/` is intentionally ignored by git. Only selected showcase artifacts are copied into `docs/assets/`.

## Quick Start

Create the environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the current tracked Question 3 hierarchical PPO workflow:

```bash
./.venv/bin/python q3_sb3.py train-slice \
  --device cpu \
  --seed 17 \
  --total-timesteps 10000 \
  --eval-freq 1000 \
  --eval-episodes 1 \
  --model-out outputs/q3_sb3/q3_slice_ppo_seed17_10k.zip \
  --metrics-out outputs/q3_sb3/q3_slice_ppo_seed17_10k_metrics.json \
  --plot-out outputs/q3_sb3/q3_slice_ppo_seed17_10k_curves.png
```

```bash
./.venv/bin/python q3_sb3.py train-power \
  --device cpu \
  --seed 17 \
  --slice-mode model \
  --slice-model-in outputs/q3_sb3/q3_slice_ppo_seed17_10k.zip \
  --total-timesteps 10000 \
  --eval-freq 1000 \
  --eval-episodes 1 \
  --model-out outputs/q3_sb3/q3_power_ppo_seed17_10k.zip \
  --metrics-out outputs/q3_sb3/q3_power_ppo_seed17_10k_metrics.json \
  --plot-out outputs/q3_sb3/q3_power_ppo_seed17_10k_curves.png
```

```bash
./.venv/bin/python q3_sb3.py evaluate \
  --seed 17 \
  --slice-mode model \
  --slice-model-in outputs/q3_sb3/q3_slice_ppo_seed17_10k.zip \
  --power-model-in outputs/q3_sb3/q3_power_ppo_seed17_10k.zip \
  --eval-episodes 1 \
  --metrics-out outputs/q3_sb3/q3_combined_eval_seed17_10k.json
```

To continue from an existing checkpoint, add `--init-model <checkpoint.zip>` to `train-slice` or `train-power`. For a quick smoke run, lower `--total-timesteps`.

## Safe GitHub Publish Flow

This repository is already connected to `origin`. To publish the cleaned showcase version without accidentally committing ignored checkpoints:

```bash
git add .gitignore README.md requirements.txt q2_mpc.py q3_hierarchical_rl.py q3_sb3.py docs/Q3_RESULTS.md docs/PAPER_DRAFT.md docs/assets/q3_slice_seed17_convergence.png
git commit -m "Refresh tracked results and paper draft"
git push -u origin HEAD
```

If you also want another machine to continue training from your current checkpoints, share the `outputs/q3_sb3/` checkpoint files separately or add them with `git add -f`.
