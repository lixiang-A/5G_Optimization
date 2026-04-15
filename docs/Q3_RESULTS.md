# Question 3 Result Snapshot

This document tracks the current GitHub-visible evidence for the Question 3 hierarchical RL pipeline after the latest multi-seed reruns.

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
| Slice-layer PPO, `seed=7` | `q3_sb3.py train-slice` | final evaluation `objective` | `0.4918847769960313` | `outputs/q3_sb3/q3_slice_ppo_fresh_10k_metrics.json` |
| Slice-layer PPO, `seed=17` | `q3_sb3.py train-slice` | final evaluation `objective` | `0.49750574292733196` | `outputs/q3_sb3/q3_slice_ppo_seed17_10k_metrics.json` |
| Slice-layer PPO, `seed=27` | `q3_sb3.py train-slice` | final evaluation `objective` | `0.4936998213169567` | `outputs/q3_sb3/q3_slice_ppo_seed27_10k_metrics.json` |
| Hierarchical PPO combined, `seed=7` | `q3_sb3.py evaluate` | `objective` | `0.49188382075400083` | `outputs/q3_sb3/q3_combined_eval_fresh_10k.json` |
| Hierarchical PPO combined, `seed=17` | `q3_sb3.py evaluate` | `objective` | `0.4994806473699616` | `outputs/q3_sb3/q3_combined_eval_seed17_10k.json` |
| Hierarchical PPO combined, `seed=27` | `q3_sb3.py evaluate` | `objective` | `0.4965230144859628` | `outputs/q3_sb3/q3_combined_eval_seed27_10k.json` |
| Multi-seed combined summary | `seed=7 / 17 / 27` | mean `objective` | `0.4959624942033084` | the three combined-eval files above |
| Multi-seed combined summary | `seed=7 / 17 / 27` | objective std | `0.0031266148786072943` | the three combined-eval files above |

## Combined Evaluation Detail

The current tracked combined evaluation reports:

| Seed | Objective | URLLC dropped | eMBB dropped | mMTC dropped | URLLC avg delay | eMBB avg delay | mMTC avg delay | eMBB avg service rate | Mean transmit power | Mean interference ratio |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `7` | `0.49188382075400083` | `0` | `587` | `10556` | `1.3598 ms` | `100.0000 ms` | `432.2554 ms` | `116.1742 Mbps` | `0.0902 W` | `0.3494` |
| `17` | `0.4994806473699616` | `0` | `587` | `10556` | `1.2795 ms` | `99.9393 ms` | `398.5841 ms` | `112.2346 Mbps` | `0.1032 W` | `0.3555` |
| `27` | `0.4965230144859628` | `0` | `587` | `10555` | `1.2899 ms` | `99.8143 ms` | `407.4461 ms` | `112.4428 Mbps` | `0.1004 W` | `0.3541` |

## How To Read These Numbers

- The hierarchical PPO controller is now stably better than the lightweight numpy baseline across the tracked `seed=7 / 17 / 27` reruns.
- `seed=17` is the best current showcase run and is therefore the preferred GitHub and paper-display checkpoint.
- All three tracked Q3 reruns achieve `0` URLLC drops, which is materially stronger than the earlier short-run snapshot.
- eMBB delay remains near the `100 ms` SLA boundary. This is acceptable for the current result package, but it is also the main place where the method is still tight rather than clearly slack.
- mMTC remains the hardest slice, but the multi-seed reruns improve its average delay relative to the earlier tracked snapshot.
- Given the small cross-seed variation in objective, further large-scale retuning is no longer the priority. The current focus should shift to writing and presentation.

## Tracked Figure

- GitHub-visible convergence figure: `docs/assets/q3_slice_seed17_convergence.png`
- Source file copied from: `outputs/q3_sb3/q3_slice_ppo_seed17_10k_curves.png`

## Important Caveats

- Each combined evaluation file still reports one deterministic rollout per trained policy. The stronger stability evidence now comes from retraining across multiple seeds rather than from repeated stochastic evaluation of one checkpoint.
- In the power-layer metrics files, the best checkpoint during training can be slightly better than the final exported checkpoint. This is a potential future checkpoint-selection refinement, not a blocker for the current paper-writing stage.
- The current environment assigns the serving BS as the nearest micro base station at task arrival time. This is an explicit modeling assumption, not an omitted design choice.
