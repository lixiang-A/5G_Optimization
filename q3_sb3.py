#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Callable, Dict, List, Protocol

PROJECT_ROOT = Path(__file__).resolve().parent
CACHE_ROOT = PROJECT_ROOT / ".cache"
CACHE_ROOT.mkdir(parents=True, exist_ok=True)
MPL_CACHE = CACHE_ROOT / "matplotlib"
MPL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))

import gymnasium as gym
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

from q3_hierarchical_rl import (
    BS_NAMES,
    POWER_HALF_RANGE_DB,
    POWER_MAX_DBM,
    POWER_MID_DBM,
    POWER_MIN_DBM,
    ROOT,
    SLICE_CONFIGS,
    SLICE_INDEX,
    SLICE_KEYS,
    MultiBSQ3Env,
    build_slice_action_space,
    load_q3_data,
)

OUTPUT_DIR = ROOT / "outputs" / "q3_sb3"


def normalize_slice_allocations(action_ids: np.ndarray, action_space: List[tuple[int, int, int]]) -> np.ndarray:
    allocations = np.asarray([action_space[int(idx)] for idx in action_ids], dtype=np.float32)
    return (allocations / 50.0).reshape(-1)


class SlicePlanner(Protocol):
    def select(self, core_env: MultiBSQ3Env) -> np.ndarray:
        ...


class HeuristicSlicePlanner:
    def __init__(self, action_space: List[tuple[int, int, int]]) -> None:
        self.action_space = action_space

    def _pressure_vector(self, core_env: MultiBSQ3Env, bs_idx: int) -> np.ndarray:
        current_ms = min(core_env.current_ms, 999)
        recent_start = max(0, core_env.current_ms - 100)
        pressure = []
        for slice_key in SLICE_KEYS:
            queue = core_env.queues[bs_idx][slice_key]
            cfg = SLICE_CONFIGS[slice_key]
            q_len = len(queue)
            hol_ratio = 0.0 if q_len == 0 else min((core_env.current_ms - queue[0].arrival_ms) / cfg.delay_sla_ms, 4.0)
            arrivals_recent = (
                core_env.data.cumulative_arrivals[SLICE_INDEX[slice_key], bs_idx, core_env.current_ms]
                - core_env.data.cumulative_arrivals[SLICE_INDEX[slice_key], bs_idx, recent_start]
            )
            users = core_env.data.current_users_by_bs_slice[current_ms][bs_idx][SLICE_INDEX[slice_key]]
            mean_gain = (
                float(np.mean(core_env.data.channel_gain[bs_idx, current_ms, list(users)]))
                if users
                else 0.0
            )
            score = (
                1.5 * q_len
                + 3.0 * hol_ratio
                + 0.25 * arrivals_recent
                + (0.2 if mean_gain < 1e-6 else 0.0)
            )
            pressure.append(score + 1e-3)
        return np.asarray(pressure, dtype=np.float64)

    def select(self, core_env: MultiBSQ3Env) -> np.ndarray:
        action_ids = []
        for bs_idx in range(len(BS_NAMES)):
            target = self._pressure_vector(core_env, bs_idx)
            target = target / max(float(target.sum()), 1e-9)
            best_idx = 0
            best_cost = float("inf")
            for idx, action in enumerate(self.action_space):
                alloc = np.asarray(action, dtype=np.float64) / 50.0
                cost = float(np.square(alloc - target).sum())
                if cost < best_cost:
                    best_cost = cost
                    best_idx = idx
            action_ids.append(best_idx)
        return np.asarray(action_ids, dtype=np.int64)


class TrainedSlicePlanner:
    def __init__(self, model_path: Path) -> None:
        self.model = PPO.load(model_path)

    def select(self, core_env: MultiBSQ3Env) -> np.ndarray:
        obs = core_env.global_observation().astype(np.float32)
        action, _ = self.model.predict(obs, deterministic=True)
        return np.asarray(action, dtype=np.int64)


class Q3SliceEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        fixed_power_dbm: float,
        weights: Dict[str, float],
        lambda_power: float,
        lambda_interference: float,
        lambda_fairness: float,
    ) -> None:
        super().__init__()
        self.data = load_q3_data()
        self.core_env = MultiBSQ3Env(
            data=self.data,
            weights=weights,
            lambda_power=lambda_power,
            lambda_interference=lambda_interference,
            lambda_fairness=lambda_fairness,
        )
        self.fixed_power_dbm = float(np.clip(fixed_power_dbm, POWER_MIN_DBM, POWER_MAX_DBM))
        self.slice_action_space = build_slice_action_space()
        self.action_space = spaces.MultiDiscrete([len(self.slice_action_space)] * len(BS_NAMES))
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(self.core_env.global_obs_dim,),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        obs = self.core_env.reset().astype(np.float32)
        return obs, {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.asarray(action, dtype=np.int64)
        powers = np.full((len(BS_NAMES), len(SLICE_KEYS)), self.fixed_power_dbm, dtype=np.float64)
        obs, reward, done, info = self.core_env.step(action, powers)
        info = dict(info)
        if done:
            info["episode_summary"] = self.core_env.summarize()
        return obs.astype(np.float32), float(reward), done, False, info


class Q3PowerEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(
        self,
        slice_planner: SlicePlanner,
        weights: Dict[str, float],
        lambda_power: float,
        lambda_interference: float,
        lambda_fairness: float,
    ) -> None:
        super().__init__()
        self.data = load_q3_data()
        self.core_env = MultiBSQ3Env(
            data=self.data,
            weights=weights,
            lambda_power=lambda_power,
            lambda_interference=lambda_interference,
            lambda_fairness=lambda_fairness,
        )
        self.slice_planner = slice_planner
        self.slice_action_space = build_slice_action_space()
        self.current_slice_action_ids = np.zeros(len(BS_NAMES), dtype=np.int64)
        obs_dim = self.core_env.global_obs_dim + len(BS_NAMES) * len(SLICE_KEYS)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(BS_NAMES) * len(SLICE_KEYS),),
            dtype=np.float32,
        )
        self.observation_space = spaces.Box(low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32)

    def _power_obs(self) -> np.ndarray:
        core_obs = self.core_env.global_observation().astype(np.float32)
        slice_obs = normalize_slice_allocations(self.current_slice_action_ids, self.slice_action_space)
        return np.concatenate([core_obs, slice_obs.astype(np.float32)], axis=0)

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        super().reset(seed=seed)
        self.core_env.reset()
        self.current_slice_action_ids = self.slice_planner.select(self.core_env)
        return self._power_obs(), {}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        action = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        powers = POWER_MID_DBM + POWER_HALF_RANGE_DB * action.reshape(len(BS_NAMES), len(SLICE_KEYS))
        obs, reward, done, info = self.core_env.step(self.current_slice_action_ids, powers.astype(np.float64))
        info = dict(info)
        if done:
            info["episode_summary"] = self.core_env.summarize()
            return self._power_obs(), float(reward), True, False, info
        self.current_slice_action_ids = self.slice_planner.select(self.core_env)
        return self._power_obs(), float(reward), False, False, info


def build_common_ppo_kwargs(args: argparse.Namespace) -> dict:
    return {
        "learning_rate": args.learning_rate,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "gamma": args.gamma,
        "gae_lambda": args.gae_lambda,
        "ent_coef": args.ent_coef,
        "vf_coef": args.vf_coef,
        "clip_range": args.clip_range,
        "verbose": 1,
        "device": args.device,
        "policy_kwargs": {"net_arch": list(args.net_arch)},
    }


def build_or_load_model(args: argparse.Namespace, env: gym.Env) -> PPO:
    if args.init_model is not None:
        model = PPO.load(args.init_model, env=env, device=args.device)
        model.set_random_seed(args.seed)
        return model
    return PPO("MlpPolicy", env, seed=args.seed, **build_common_ppo_kwargs(args))


def moving_average(values: List[float], window: int = 10) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return arr
    window = max(1, min(window, arr.size))
    kernel = np.ones(window, dtype=np.float64) / window
    padded = np.pad(arr, (window - 1, 0), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def run_deterministic_episode(model: PPO, env: gym.Env) -> dict:
    obs, _ = env.reset()
    done = False
    summary = None
    total_reward = 0.0
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)
        total_reward += float(reward)
        summary = info.get("episode_summary", summary)
    return {
        "episode_reward": total_reward,
        "summary": summary,
    }


class TrainingCurveCallback(BaseCallback):
    def __init__(
        self,
        eval_env_factory: Callable[[], gym.Env],
        eval_freq: int,
        eval_episodes: int,
    ) -> None:
        super().__init__()
        self.eval_env_factory = eval_env_factory
        self.eval_freq = max(0, int(eval_freq))
        self.eval_episodes = max(1, int(eval_episodes))
        self.next_eval_timestep = self.eval_freq if self.eval_freq > 0 else None
        self.episode_history: List[Dict[str, float | int | None]] = []
        self.evaluation_history: List[Dict[str, float | int | None]] = []
        self.episode_count = 0

    def _record_episode(self, info: dict) -> None:
        episode = info.get("episode", {})
        summary = info.get("episode_summary", {})
        self.episode_count += 1
        self.episode_history.append(
            {
                "episode": self.episode_count,
                "timesteps": int(self.num_timesteps),
                "reward": float(episode.get("r", 0.0)),
                "length": int(episode.get("l", 0)),
                "objective": float(summary["objective"]) if "objective" in summary else None,
                "mean_power_w": float(summary["mean_power_w"]) if "mean_power_w" in summary else None,
                "mean_interference_ratio": (
                    float(summary["mean_interference_ratio"])
                    if "mean_interference_ratio" in summary
                    else None
                ),
            }
        )

    def _run_eval(self) -> None:
        eval_env = self.eval_env_factory()
        mean_reward, std_reward = evaluate_policy(
            self.model,
            eval_env,
            n_eval_episodes=self.eval_episodes,
            deterministic=True,
        )
        rollout = run_deterministic_episode(self.model, eval_env)
        summary = rollout["summary"] or {}
        self.evaluation_history.append(
            {
                "timesteps": int(self.num_timesteps),
                "mean_reward": float(mean_reward),
                "std_reward": float(std_reward),
                "objective": float(summary["objective"]) if "objective" in summary else None,
                "mean_power_w": float(summary["mean_power_w"]) if "mean_power_w" in summary else None,
                "mean_interference_ratio": (
                    float(summary["mean_interference_ratio"])
                    if "mean_interference_ratio" in summary
                    else None
                ),
            }
        )
        eval_env.close()

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self._record_episode(info)

        if self.next_eval_timestep is not None:
            while self.num_timesteps >= self.next_eval_timestep:
                self._run_eval()
                self.next_eval_timestep += self.eval_freq
        return True

    def _on_training_end(self) -> None:
        if not self.evaluation_history or self.evaluation_history[-1]["timesteps"] != int(self.num_timesteps):
            self._run_eval()


def save_training_curve_plot(
    episode_history: List[Dict[str, float | int | None]],
    evaluation_history: List[Dict[str, float | int | None]],
    plot_out: Path,
    title: str,
) -> None:
    plot_out.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle(title)

    if episode_history:
        episodes = np.asarray([int(row["episode"]) for row in episode_history], dtype=np.int64)
        rewards = np.asarray([float(row["reward"]) for row in episode_history], dtype=np.float64)
        objectives = np.asarray(
            [float(row["objective"]) if row["objective"] is not None else np.nan for row in episode_history],
            dtype=np.float64,
        )
        powers = np.asarray(
            [float(row["mean_power_w"]) if row["mean_power_w"] is not None else np.nan for row in episode_history],
            dtype=np.float64,
        )
        interferences = np.asarray(
            [
                float(row["mean_interference_ratio"])
                if row["mean_interference_ratio"] is not None
                else np.nan
                for row in episode_history
            ],
            dtype=np.float64,
        )

        axes[0, 0].plot(episodes, rewards, alpha=0.35, label="Episode Reward")
        axes[0, 0].plot(episodes, moving_average(rewards.tolist()), linewidth=2, label="Moving Avg")
        axes[0, 0].set_title("Training Reward")
        axes[0, 0].set_xlabel("Episode")
        axes[0, 0].legend()

        axes[0, 1].plot(episodes, objectives, alpha=0.35, label="Episode Objective")
        axes[0, 1].plot(episodes, moving_average(np.nan_to_num(objectives, nan=0.0).tolist()), linewidth=2, label="Moving Avg")
        axes[0, 1].set_title("Official Objective")
        axes[0, 1].set_xlabel("Episode")
        axes[0, 1].legend()

        axes[1, 0].plot(episodes, powers, linewidth=1.5, label="Mean Power (W)")
        axes[1, 0].plot(episodes, interferences, linewidth=1.5, label="Mean Interference Ratio")
        axes[1, 0].set_title("Power And Interference")
        axes[1, 0].set_xlabel("Episode")
        axes[1, 0].legend()
    else:
        axes[0, 0].text(0.5, 0.5, "No episode history", ha="center", va="center")
        axes[0, 1].text(0.5, 0.5, "No episode history", ha="center", va="center")
        axes[1, 0].text(0.5, 0.5, "No episode history", ha="center", va="center")

    if evaluation_history:
        eval_steps = np.asarray([int(row["timesteps"]) for row in evaluation_history], dtype=np.int64)
        eval_rewards = np.asarray([float(row["mean_reward"]) for row in evaluation_history], dtype=np.float64)
        eval_objectives = np.asarray(
            [float(row["objective"]) if row["objective"] is not None else np.nan for row in evaluation_history],
            dtype=np.float64,
        )
        axes[1, 1].plot(eval_steps, eval_rewards, marker="o", label="Eval Reward")
        axes[1, 1].plot(eval_steps, eval_objectives, marker="s", label="Eval Objective")
        axes[1, 1].set_title("Evaluation Curve")
        axes[1, 1].set_xlabel("Timesteps")
        axes[1, 1].legend()
    else:
        axes[1, 1].text(0.5, 0.5, "No evaluation history", ha="center", va="center")

    for ax in axes.ravel():
        ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(plot_out, dpi=180)
    plt.close(fig)


def evaluate_slice_model(model_path: Path, env_kwargs: dict, episodes: int) -> dict:
    env = Monitor(Q3SliceEnv(**env_kwargs))
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True)
    rollout = run_deterministic_episode(model, env)
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "summary": rollout["summary"],
    }


def evaluate_power_model(model_path: Path, planner: SlicePlanner, env_kwargs: dict, episodes: int) -> dict:
    env = Monitor(Q3PowerEnv(slice_planner=planner, **env_kwargs))
    model = PPO.load(model_path)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=episodes, deterministic=True)
    rollout = run_deterministic_episode(model, env)
    return {
        "mean_reward": float(mean_reward),
        "std_reward": float(std_reward),
        "summary": rollout["summary"],
    }


def train_slice(args: argparse.Namespace) -> dict:
    weights = {"u": args.wu, "e": args.we, "m": args.wm}
    env_kwargs = {
        "fixed_power_dbm": args.fixed_power_dbm,
        "weights": weights,
        "lambda_power": args.lambda_power,
        "lambda_interference": args.lambda_interference,
        "lambda_fairness": args.lambda_fairness,
    }
    env = Monitor(Q3SliceEnv(**env_kwargs))
    callback = TrainingCurveCallback(
        eval_env_factory=lambda: Monitor(Q3SliceEnv(**env_kwargs)),
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
    )
    model = build_or_load_model(args, env)
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=False,
        callback=callback,
        reset_num_timesteps=args.reset_num_timesteps,
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    metrics = evaluate_slice_model(args.model_out, env_kwargs, episodes=args.eval_episodes)
    plot_out = args.plot_out or args.model_out.with_name(f"{args.model_out.stem}_curves.png")
    save_training_curve_plot(
        callback.episode_history,
        callback.evaluation_history,
        plot_out,
        title="Q3 Slice PPO",
    )
    return {
        "mode": "slice",
        "model_out": str(args.model_out),
        "plot_out": str(plot_out),
        "config": {
            "total_timesteps": args.total_timesteps,
            "fixed_power_dbm": args.fixed_power_dbm,
            "init_model": str(args.init_model) if args.init_model else None,
            "reset_num_timesteps": args.reset_num_timesteps,
            "weights": weights,
        },
        "training_history": callback.episode_history,
        "evaluation_history": callback.evaluation_history,
        "evaluation": metrics,
    }


def build_slice_planner(args: argparse.Namespace, action_space: List[tuple[int, int, int]]) -> SlicePlanner:
    if args.slice_mode == "model":
        if args.slice_model_in is None:
            raise SystemExit("--slice-model-in is required when --slice-mode=model.")
        return TrainedSlicePlanner(args.slice_model_in)
    return HeuristicSlicePlanner(action_space)


def train_power(args: argparse.Namespace) -> dict:
    weights = {"u": args.wu, "e": args.we, "m": args.wm}
    env_kwargs = {
        "weights": weights,
        "lambda_power": args.lambda_power,
        "lambda_interference": args.lambda_interference,
        "lambda_fairness": args.lambda_fairness,
    }
    planner = build_slice_planner(args, build_slice_action_space())
    env = Monitor(Q3PowerEnv(slice_planner=planner, **env_kwargs))
    callback = TrainingCurveCallback(
        eval_env_factory=lambda: Monitor(Q3PowerEnv(slice_planner=planner, **env_kwargs)),
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
    )
    model = build_or_load_model(args, env)
    model.learn(
        total_timesteps=args.total_timesteps,
        progress_bar=False,
        callback=callback,
        reset_num_timesteps=args.reset_num_timesteps,
    )
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    model.save(args.model_out)
    metrics = evaluate_power_model(args.model_out, planner, env_kwargs, episodes=args.eval_episodes)
    plot_out = args.plot_out or args.model_out.with_name(f"{args.model_out.stem}_curves.png")
    save_training_curve_plot(
        callback.episode_history,
        callback.evaluation_history,
        plot_out,
        title="Q3 Power PPO",
    )
    return {
        "mode": "power",
        "model_out": str(args.model_out),
        "plot_out": str(plot_out),
        "config": {
            "total_timesteps": args.total_timesteps,
            "slice_mode": args.slice_mode,
            "slice_model_in": str(args.slice_model_in) if args.slice_model_in else None,
            "init_model": str(args.init_model) if args.init_model else None,
            "reset_num_timesteps": args.reset_num_timesteps,
            "weights": weights,
        },
        "training_history": callback.episode_history,
        "evaluation_history": callback.evaluation_history,
        "evaluation": metrics,
    }


def evaluate_combined(args: argparse.Namespace) -> dict:
    weights = {"u": args.wu, "e": args.we, "m": args.wm}
    action_space = build_slice_action_space()
    planner = build_slice_planner(args, action_space)
    env_kwargs = {
        "weights": weights,
        "lambda_power": args.lambda_power,
        "lambda_interference": args.lambda_interference,
        "lambda_fairness": args.lambda_fairness,
    }
    if args.power_model_in is None:
        raise SystemExit("--power-model-in is required for evaluation.")
    return {
        "mode": "evaluate",
        "slice_mode": args.slice_mode,
        "slice_model_in": str(args.slice_model_in) if args.slice_model_in else None,
        "power_model_in": str(args.power_model_in),
        "evaluation": evaluate_power_model(args.power_model_in, planner, env_kwargs, episodes=args.eval_episodes),
    }


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--total-timesteps", type=int, default=5000)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gamma", type=float, default=0.98)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--wu", type=float, default=1.0 / 3.0)
    parser.add_argument("--we", type=float, default=1.0 / 3.0)
    parser.add_argument("--wm", type=float, default=1.0 / 3.0)
    parser.add_argument("--lambda-power", type=float, default=0.02)
    parser.add_argument("--lambda-interference", type=float, default=0.05)
    parser.add_argument("--lambda-fairness", type=float, default=0.02)
    parser.add_argument("--net-arch", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--eval-freq", type=int, default=500)
    parser.add_argument("--metrics-out", type=Path, default=None)
    parser.add_argument("--plot-out", type=Path, default=None)
    parser.add_argument("--init-model", type=Path, default=None)
    parser.add_argument(
        "--reset-num-timesteps",
        action="store_true",
        help="Reset SB3 timestep counter when continuing from --init-model.",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "SB3/PyTorch hierarchical training entry for Question 3. "
            "Slice allocation is trained as a MultiDiscrete PPO policy, and power control "
            "is trained as a continuous PPO policy conditioned on slice decisions."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    slice_parser = subparsers.add_parser("train-slice", help="Train the slice-allocation PPO.")
    add_common_args(slice_parser)
    slice_parser.add_argument("--fixed-power-dbm", type=float, default=30.0)
    slice_parser.add_argument(
        "--model-out",
        type=Path,
        default=OUTPUT_DIR / "q3_slice_ppo.zip",
    )

    power_parser = subparsers.add_parser("train-power", help="Train the power-control PPO.")
    add_common_args(power_parser)
    power_parser.add_argument("--slice-mode", choices=("heuristic", "model"), default="heuristic")
    power_parser.add_argument("--slice-model-in", type=Path, default=None)
    power_parser.add_argument(
        "--model-out",
        type=Path,
        default=OUTPUT_DIR / "q3_power_ppo.zip",
    )

    eval_parser = subparsers.add_parser("evaluate", help="Evaluate a combined slice+power controller.")
    add_common_args(eval_parser)
    eval_parser.add_argument("--slice-mode", choices=("heuristic", "model"), default="model")
    eval_parser.add_argument("--slice-model-in", type=Path, default=OUTPUT_DIR / "q3_slice_ppo.zip")
    eval_parser.add_argument("--power-model-in", type=Path, default=OUTPUT_DIR / "q3_power_ppo.zip")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight_sum = args.wu + args.we + args.wm
    if abs(weight_sum - 1.0) > 1e-9:
        raise SystemExit("Weights must sum to 1.")
    if args.total_timesteps <= 0:
        raise SystemExit("--total-timesteps must be positive.")

    if args.command == "train-slice":
        payload = train_slice(args)
    elif args.command == "train-power":
        payload = train_power(args)
    else:
        payload = evaluate_combined(args)

    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
