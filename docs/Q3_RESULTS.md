# Question 3 Result Snapshot

This document tracks the current GitHub-visible evidence for the Question 3 hierarchical RL pipeline.

## Data And Scope

- Scope: Questions 1 to 3 only.
- Question 3 data files:
  - `BS2等5个文件/BS1.xlsx`
  - `BS2等5个文件/BS2.xlsx`
  - `BS2等5个文件/BS3.xlsx`
  - `BS2等5个文件/taskflow.xlsx`
- Time scales follow the contest statement and the modeling document in `背景信息/5G环境开发设计文档 (1).md`:
  - slow decision scale: `100 ms`
  - service execution scale: `1 ms`

## Code Paths

- `q3_hierarchical_rl.py`: environment plus lightweight numpy hierarchical actor-critic baseline.
- `q3_sb3.py`: SB3/PyTorch hierarchical PPO training and evaluation entry.

## Current Tracked Results

| Item | Script / command path | Metric | Value | Evidence source |
| --- | --- | --- | --- | --- |
| Lightweight baseline | `q3_hierarchical_rl.py` | `best_eval_objective` | `0.4552520100347934` | `outputs/q3_rl/short_run_metrics.json` |
| Slice-layer PPO | `q3_sb3.py train-slice` | `objective` | `0.47275657040086505` | `outputs/q3_sb3/q3_slice_metrics.json` |
| Slice-layer PPO | `q3_sb3.py train-slice` | training history length | `204` episodes | `outputs/q3_sb3/q3_slice_metrics.json` |
| Slice-layer PPO | `q3_sb3.py train-slice` | evaluation history length | `11` checkpoints | `outputs/q3_sb3/q3_slice_metrics.json` |
| Hierarchical PPO combined evaluation | `q3_sb3.py evaluate` | `objective` | `0.48511000199064225` | `outputs/q3_sb3/q3_combined_eval.json` |
| Hierarchical PPO combined evaluation | `q3_sb3.py evaluate` | `mean_reward` | `7.318471` | `outputs/q3_sb3/q3_combined_eval.json` |

## Combined Evaluation Detail

The current tracked combined evaluation reports:

| KPI | Value |
| --- | --- |
| URLLC completed / dropped | `558 / 1` |
| eMBB completed / dropped | `5406 / 587` |
| mMTC completed / dropped | `10527 / 10556` |
| URLLC average delay | `1.1302 ms` |
| eMBB average delay | `99.8656 ms` |
| mMTC average delay | `437.0022 ms` |
| eMBB average service rate | `117.6738 Mbps` |
| Mean transmit power | `0.1784 W` |
| Mean interference ratio | `0.3600` |

## How To Read These Numbers

- The current SB3 hierarchical PPO result is better than the lightweight numpy baseline in this tracked short-to-medium training setup.
- URLLC delay remains well inside the `5 ms` SLA.
- eMBB delay is very close to the `100 ms` SLA boundary, so longer training and additional tuning are still meaningful.
- mMTC backlog remains the hardest slice in the current formulation, which is a real limitation rather than something hidden by the presentation.

## Tracked Figure

- GitHub-visible convergence figure: `docs/assets/q3_slice_convergence.png`
- Source file copied from: `outputs/q3_sb3/q3_slice_ppo_curves.png`

## Important Caveats

- The current combined evaluation is based on a single evaluation episode, so it is suitable for a GitHub project snapshot but not yet for a paper-grade final claim.
- The tracked convergence figure currently covers the slice-layer PPO run. The power-layer PPO should be rerun with the latest history logging if a second convergence figure is needed for a report or presentation.
- The current environment assigns the serving BS as the nearest micro base station at task arrival time. This is an explicit modeling assumption, not an omitted design choice.
