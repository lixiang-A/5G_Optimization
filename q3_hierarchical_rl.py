#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple

import numpy as np

from q2_mpc import read_xlsx_tables, user_sort_key


ROOT = Path(__file__).resolve().parent
Q3_DIR = ROOT / "BS2等5个文件"
TASKFLOW_XLSX = Q3_DIR / "taskflow.xlsx"
BS_FILES = ("BS1.xlsx", "BS2.xlsx", "BS3.xlsx")
BS_NAMES = ("BS1", "BS2", "BS3")
BS_POSITIONS = {
    "BS1": (0.0, 500.0),
    "BS2": (-433.0127, -250.0),
    "BS3": (433.0127, -250.0),
}
SLICE_KEYS = ("u", "e", "m")
SLICE_INDEX = {key: idx for idx, key in enumerate(SLICE_KEYS)}
TOTAL_RBS = 50
DECISION_MS = 100
TOTAL_MS = 1000
POWER_MIN_DBM = 10.0
POWER_MAX_DBM = 30.0
POWER_MID_DBM = 0.5 * (POWER_MIN_DBM + POWER_MAX_DBM)
POWER_HALF_RANGE_DB = 0.5 * (POWER_MAX_DBM - POWER_MIN_DBM)
W_RB = 360e3
NF_DB = 7.0
EPS = 1e-12


@dataclass(frozen=True)
class SliceConfig:
    name: str
    prefix: str
    rb_granularity: int
    rate_sla_mbps: float
    delay_sla_ms: int
    penalty: float
    alpha: float = 1.0


@dataclass(frozen=True)
class ArrivalSpec:
    task_id: int
    user_idx: int
    user_name: str
    slice_key: str
    arrival_ms: int
    arrival_window: int
    size_bits: float
    bs_idx: int


@dataclass
class TaskState:
    task_id: int
    user_idx: int
    user_name: str
    slice_key: str
    arrival_ms: int
    arrival_window: int
    size_bits: float
    remaining_bits: float
    bs_idx: int
    served_ms: float = 0.0

    def clone(self) -> "TaskState":
        return TaskState(
            task_id=self.task_id,
            user_idx=self.user_idx,
            user_name=self.user_name,
            slice_key=self.slice_key,
            arrival_ms=self.arrival_ms,
            arrival_window=self.arrival_window,
            size_bits=self.size_bits,
            remaining_bits=self.remaining_bits,
            bs_idx=self.bs_idx,
            served_ms=self.served_ms,
        )


@dataclass
class Q3Data:
    users: Tuple[str, ...]
    user_slices: Tuple[str, ...]
    arrivals_by_ms: Tuple[Tuple[ArrivalSpec, ...], ...]
    total_arrivals: Dict[str, int]
    channel_gain: np.ndarray
    attached_bs_idx: np.ndarray
    current_users_by_bs_slice: Tuple[Tuple[Tuple[Tuple[int, ...], ...], ...], ...]
    cumulative_arrivals: np.ndarray


@dataclass
class WindowAccumulator:
    proxy_scores: Dict[str, List[float]] = field(
        default_factory=lambda: {key: [] for key in SLICE_KEYS}
    )
    official_scores: Dict[str, List[float]] = field(
        default_factory=lambda: {key: [] for key in SLICE_KEYS}
    )
    success_counts: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in SLICE_KEYS})
    resolved_counts: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in SLICE_KEYS})
    power_w_sum: float = 0.0
    power_samples: int = 0
    interference_ratio_sum: float = 0.0
    interference_samples: int = 0

    def average_interference_ratio(self) -> float:
        if self.interference_samples <= 0:
            return 0.0
        return self.interference_ratio_sum / self.interference_samples

    def average_power_w(self) -> float:
        if self.power_samples <= 0:
            return 0.0
        return self.power_w_sum / self.power_samples


@dataclass
class EpisodeStats:
    objective_value: float = 0.0
    completed: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in SLICE_KEYS})
    dropped: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in SLICE_KEYS})
    delay_sum_ms: Dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in SLICE_KEYS})
    reward_sum: Dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in SLICE_KEYS})
    embb_rate_sum_mbps: float = 0.0
    power_w_sum: float = 0.0
    power_samples: int = 0
    interference_ratio_sum: float = 0.0
    interference_samples: int = 0
    decisions: List[Dict[str, object]] = field(default_factory=list)


SLICE_CONFIGS: Dict[str, SliceConfig] = {
    "u": SliceConfig("URLLC", "U", 10, 10.0, 5, 5.0, alpha=0.95),
    "e": SliceConfig("eMBB", "e", 5, 50.0, 100, 3.0),
    "m": SliceConfig("mMTC", "m", 2, 1.0, 500, 1.0),
}


def sigmoid(x: float) -> float:
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def classify_slice(user: str) -> str:
    return "u" if user.startswith("U") else user[0]


def build_slice_action_space() -> List[Tuple[int, int, int]]:
    actions: List[Tuple[int, int, int]] = []
    for x_u in range(0, TOTAL_RBS + 1, SLICE_CONFIGS["u"].rb_granularity):
        for x_e in range(0, TOTAL_RBS - x_u + 1, SLICE_CONFIGS["e"].rb_granularity):
            x_m = TOTAL_RBS - x_u - x_e
            if x_m >= 0 and x_m % SLICE_CONFIGS["m"].rb_granularity == 0:
                actions.append((x_u, x_e, x_m))
    return actions


def dbm_to_mw(power_dbm: np.ndarray) -> np.ndarray:
    return np.power(10.0, power_dbm / 10.0)


def jain_index(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    denom = arr.size * np.square(arr).sum()
    if denom <= EPS:
        return 0.0
    return float(np.square(arr.sum()) / denom)


def nearest_bs_idx(x: float, y: float) -> int:
    best_idx = 0
    best_dist = float("inf")
    for idx, name in enumerate(BS_NAMES):
        bx, by = BS_POSITIONS[name]
        dist = (x - bx) * (x - bx) + (y - by) * (y - by)
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx


def load_q3_data(taskflow_path: Path = TASKFLOW_XLSX, base_dir: Path = Q3_DIR) -> Q3Data:
    taskflow_tables = read_xlsx_tables(taskflow_path)
    position_rows = taskflow_tables["用户位置"]
    task_rows = taskflow_tables["用户任务流"]

    users = tuple(sorted((u for u in task_rows[0] if u != "Time"), key=user_sort_key))
    user_index = {user: idx for idx, user in enumerate(users)}
    user_slices = tuple(classify_slice(user) for user in users)
    num_users = len(users)
    num_bs = len(BS_NAMES)
    num_ms = len(task_rows)

    channel_gain = np.zeros((num_bs, num_ms, num_users), dtype=np.float64)
    for bs_idx, filename in enumerate(BS_FILES):
        tables = read_xlsx_tables(base_dir / filename)
        pathloss_rows = tables["大规模衰减"]
        fading_rows = tables["小规模瑞丽衰减"]
        for ms in range(num_ms):
            for user in users:
                uid = user_index[user]
                pathloss_db = float(pathloss_rows[ms][user])
                fading = float(fading_rows[ms][user])
                channel_gain[bs_idx, ms, uid] = (10.0 ** (-pathloss_db / 10.0)) * (fading * fading)

    attached_bs_idx = np.zeros((num_ms, num_users), dtype=np.int64)
    current_users_by_bs_slice: List[List[List[List[int]]]] = []
    for ms, row in enumerate(position_rows):
        groups = [[[] for _ in SLICE_KEYS] for _ in BS_NAMES]
        for user in users:
            uid = user_index[user]
            x = float(row[f"{user}_X"])
            y = float(row[f"{user}_Y"])
            bs_idx = nearest_bs_idx(x, y)
            attached_bs_idx[ms, uid] = bs_idx
            groups[bs_idx][SLICE_INDEX[user_slices[uid]]].append(uid)
        current_users_by_bs_slice.append(groups)

    arrivals_by_ms: List[List[ArrivalSpec]] = [[] for _ in range(num_ms)]
    total_arrivals = {key: 0 for key in SLICE_KEYS}
    cumulative_arrivals = np.zeros((len(SLICE_KEYS), num_bs, num_ms + 1), dtype=np.int64)
    next_task_id = 0
    for ms, row in enumerate(task_rows):
        window_idx = ms // DECISION_MS
        for user in users:
            size_mbit = float(row[user])
            if size_mbit <= 0.0:
                continue
            uid = user_index[user]
            slice_key = user_slices[uid]
            bs_idx = int(attached_bs_idx[ms, uid])
            arrivals_by_ms[ms].append(
                ArrivalSpec(
                    task_id=next_task_id,
                    user_idx=uid,
                    user_name=user,
                    slice_key=slice_key,
                    arrival_ms=ms,
                    arrival_window=window_idx,
                    size_bits=size_mbit * 1e6,
                    bs_idx=bs_idx,
                )
            )
            total_arrivals[slice_key] += 1
            cumulative_arrivals[SLICE_INDEX[slice_key], bs_idx, ms + 1] += 1
            next_task_id += 1

    cumulative_arrivals = cumulative_arrivals.cumsum(axis=2)
    grouped_users = tuple(
        tuple(
            tuple(tuple(group) for group in bs_groups)
            for bs_groups in ms_groups
        )
        for ms_groups in current_users_by_bs_slice
    )
    return Q3Data(
        users=users,
        user_slices=user_slices,
        arrivals_by_ms=tuple(tuple(items) for items in arrivals_by_ms),
        total_arrivals=total_arrivals,
        channel_gain=channel_gain,
        attached_bs_idx=attached_bs_idx,
        current_users_by_bs_slice=grouped_users,
        cumulative_arrivals=cumulative_arrivals,
    )


class MultiBSQ3Env:
    """
    第三问环境中，题面没有显式给出可优化的接入决策，因此这里按用户在任务到达时刻的最近微基站
    作为服务基站；真正的接入决策保留给第四问。
    """

    def __init__(
        self,
        data: Q3Data,
        weights: Dict[str, float],
        lambda_power: float = 0.02,
        lambda_interference: float = 0.05,
        lambda_fairness: float = 0.02,
        kappa_u: float = 1.0,
        kappa_r: float = 0.15,
        kappa_tau: float = 0.10,
    ) -> None:
        self.data = data
        self.weights = weights
        self.lambda_power = lambda_power
        self.lambda_interference = lambda_interference
        self.lambda_fairness = lambda_fairness
        self.kappa_u = kappa_u
        self.kappa_r = kappa_r
        self.kappa_tau = kappa_tau
        self.action_space = build_slice_action_space()
        self.noise_mw = np.array(
            [
                10.0
                ** (
                    (
                        -174.0
                        + 10.0 * math.log10(SLICE_CONFIGS[key].rb_granularity * W_RB)
                        + NF_DB
                    )
                    / 10.0
                )
                for key in SLICE_KEYS
            ],
            dtype=np.float64,
        )
        self.local_obs_dim = 1 + 1 + len(SLICE_KEYS) * 7 + len(SLICE_KEYS) + len(SLICE_KEYS)
        self.global_obs_dim = self.local_obs_dim * len(BS_NAMES)
        self.reset()

    def reset(self) -> np.ndarray:
        self.current_ms = 0
        self.queues: List[Dict[str, Deque[TaskState]]] = [
            {key: deque() for key in SLICE_KEYS} for _ in BS_NAMES
        ]
        self.stats = EpisodeStats()
        self.prev_action_ids = np.zeros(len(BS_NAMES), dtype=np.int64)
        self.prev_powers = np.full((len(BS_NAMES), len(SLICE_KEYS)), POWER_MAX_DBM, dtype=np.float64)
        return self.global_observation()

    def local_observation(self, bs_idx: int) -> np.ndarray:
        step_idx = self.current_ms // DECISION_MS
        current_ms = min(self.current_ms, TOTAL_MS - 1)
        recent_start = max(0, self.current_ms - DECISION_MS)
        total_queue = sum(len(self.queues[bs_idx][key]) for key in SLICE_KEYS)
        features: List[float] = [step_idx / max(TOTAL_MS // DECISION_MS - 1, 1), total_queue / 200.0]
        for slice_key in SLICE_KEYS:
            queue = self.queues[bs_idx][slice_key]
            cfg = SLICE_CONFIGS[slice_key]
            q_len = len(queue)
            hol_delay = 0.0 if q_len == 0 else (self.current_ms - queue[0].arrival_ms)
            avg_remaining = (
                0.0 if q_len == 0 else sum(task.remaining_bits for task in queue) / q_len / 1e6
            )
            arrivals_recent = (
                self.data.cumulative_arrivals[SLICE_INDEX[slice_key], bs_idx, self.current_ms]
                - self.data.cumulative_arrivals[SLICE_INDEX[slice_key], bs_idx, recent_start]
            )
            users = self.data.current_users_by_bs_slice[current_ms][bs_idx][SLICE_INDEX[slice_key]]
            if users:
                gains = self.data.channel_gain[bs_idx, current_ms, list(users)]
                mean_gain_db = float(np.log10(np.mean(gains) + EPS))
                min_gain_db = float(np.log10(np.min(gains) + EPS))
            else:
                mean_gain_db = -12.0
                min_gain_db = -12.0

            features.extend(
                [
                    q_len / 200.0,
                    min(hol_delay / max(cfg.delay_sla_ms, 1), 4.0),
                    avg_remaining / max(cfg.rate_sla_mbps * 0.1, 1.0),
                    arrivals_recent / 50.0,
                    (mean_gain_db + 12.0) / 8.0,
                    (min_gain_db + 12.0) / 8.0,
                    self.prev_action_ids[bs_idx] / max(len(self.action_space) - 1, 1),
                ]
            )
        features.extend([power / POWER_MAX_DBM for power in self.prev_powers[bs_idx]])
        features.extend(
            [
                self.action_space[self.prev_action_ids[bs_idx]][SLICE_INDEX[key]] / TOTAL_RBS
                for key in SLICE_KEYS
            ]
        )
        return np.asarray(features, dtype=np.float64)

    def local_observations(self) -> np.ndarray:
        return np.vstack([self.local_observation(bs_idx) for bs_idx in range(len(BS_NAMES))])

    def global_observation(self) -> np.ndarray:
        return self.local_observations().reshape(-1)

    def _record_interference(
        self,
        window: WindowAccumulator,
        signal_mw: float,
        interference_mw: float,
        noise_mw: float,
    ) -> None:
        window.interference_ratio_sum += interference_mw / max(
            signal_mw + interference_mw + noise_mw, EPS
        )
        window.interference_samples += 1

    def _resolve_task(
        self,
        task: TaskState,
        finish_time_ms: float,
        *,
        terminal_drop: bool,
        window: WindowAccumulator,
    ) -> None:
        cfg = SLICE_CONFIGS[task.slice_key]
        delay_ms = finish_time_ms - task.arrival_ms

        if terminal_drop or delay_ms > cfg.delay_sla_ms + EPS:
            official_score = -cfg.penalty
            if task.slice_key == "u":
                proxy_score = sigmoid(self.kappa_u * (cfg.delay_sla_ms - delay_ms))
            elif task.slice_key == "e":
                proxy_score = sigmoid(self.kappa_tau * (cfg.delay_sla_ms - delay_ms))
            else:
                proxy_score = -cfg.penalty
            self.stats.dropped[task.slice_key] += 1
        elif task.slice_key == "u":
            official_score = cfg.alpha ** delay_ms
            proxy_score = sigmoid(self.kappa_u * (cfg.delay_sla_ms - delay_ms))
            self.stats.completed[task.slice_key] += 1
            self.stats.delay_sum_ms[task.slice_key] += delay_ms
            window.success_counts[task.slice_key] += 1
        elif task.slice_key == "e":
            effective_rate_mbps = (
                task.size_bits / (task.served_ms / 1000.0) / 1e6 if task.served_ms > EPS else 0.0
            )
            official_score = min(effective_rate_mbps / cfg.rate_sla_mbps, 1.0)
            proxy_score = sigmoid(self.kappa_r * (effective_rate_mbps - cfg.rate_sla_mbps)) * sigmoid(
                self.kappa_tau * (cfg.delay_sla_ms - delay_ms)
            )
            self.stats.completed[task.slice_key] += 1
            self.stats.delay_sum_ms[task.slice_key] += delay_ms
            self.stats.embb_rate_sum_mbps += effective_rate_mbps
            window.success_counts[task.slice_key] += 1
        else:
            official_score = 1.0
            proxy_score = 1.0
            self.stats.completed[task.slice_key] += 1
            self.stats.delay_sum_ms[task.slice_key] += delay_ms
            window.success_counts[task.slice_key] += 1

        window.official_scores[task.slice_key].append(official_score)
        window.proxy_scores[task.slice_key].append(proxy_score)
        window.resolved_counts[task.slice_key] += 1
        self.stats.reward_sum[task.slice_key] += official_score
        total = max(self.data.total_arrivals[task.slice_key], 1)
        self.stats.objective_value += self.weights[task.slice_key] * official_score / total

    def _drop_expired_tasks(self, bs_idx: int, slice_key: str, ms: int, window: WindowAccumulator) -> None:
        queue = self.queues[bs_idx][slice_key]
        survivors: Deque[TaskState] = deque()
        deadline_ms = SLICE_CONFIGS[slice_key].delay_sla_ms
        while queue:
            task = queue.popleft()
            if ms - task.arrival_ms >= deadline_ms:
                self._resolve_task(task, float(ms), terminal_drop=False, window=window)
            else:
                survivors.append(task)
        self.queues[bs_idx][slice_key] = survivors

    def _append_arrivals(self, ms: int) -> None:
        for spec in self.data.arrivals_by_ms[ms]:
            self.queues[spec.bs_idx][spec.slice_key].append(
                TaskState(
                    task_id=spec.task_id,
                    user_idx=spec.user_idx,
                    user_name=spec.user_name,
                    slice_key=spec.slice_key,
                    arrival_ms=spec.arrival_ms,
                    arrival_window=spec.arrival_window,
                    size_bits=spec.size_bits,
                    remaining_bits=spec.size_bits,
                    bs_idx=spec.bs_idx,
                )
            )

    def _slice_intervals(self, allocations: np.ndarray) -> np.ndarray:
        intervals = np.full((len(BS_NAMES), len(SLICE_KEYS), 2), -1, dtype=np.int64)
        for bs_idx in range(len(BS_NAMES)):
            left = 0
            for slice_idx in range(len(SLICE_KEYS)):
                width = int(allocations[bs_idx, slice_idx])
                if width > 0:
                    intervals[bs_idx, slice_idx, 0] = left
                    intervals[bs_idx, slice_idx, 1] = left + width - 1
                    left += width
        return intervals

    def _overlap_ratio(self, allocations: np.ndarray, intervals: np.ndarray) -> np.ndarray:
        overlap = np.zeros((len(BS_NAMES), len(BS_NAMES), len(SLICE_KEYS), len(SLICE_KEYS)), dtype=np.float64)
        for bs_idx in range(len(BS_NAMES)):
            for other_idx in range(len(BS_NAMES)):
                if other_idx == bs_idx:
                    continue
                for slice_idx in range(len(SLICE_KEYS)):
                    own_width = int(allocations[bs_idx, slice_idx])
                    if own_width <= 0:
                        continue
                    own_left, own_right = intervals[bs_idx, slice_idx]
                    if own_left < 0:
                        continue
                    for other_slice_idx in range(len(SLICE_KEYS)):
                        other_left, other_right = intervals[other_idx, other_slice_idx]
                        if other_left < 0:
                            continue
                        overlap_rb = max(0, min(own_right, other_right) - max(own_left, other_left) + 1)
                        overlap[bs_idx, other_idx, slice_idx, other_slice_idx] = overlap_rb / own_width
        return overlap

    def _service_one_ms(
        self,
        ms: int,
        allocations: np.ndarray,
        powers_dbm: np.ndarray,
        overlap_ratio: np.ndarray,
        window: WindowAccumulator,
    ) -> None:
        tx_mw = dbm_to_mw(powers_dbm)
        window.power_w_sum += float(tx_mw.sum() / 1000.0)
        window.power_samples += 1
        self.stats.power_w_sum += float(tx_mw.sum() / 1000.0)
        self.stats.power_samples += 1

        servers = allocations // np.array(
            [SLICE_CONFIGS[key].rb_granularity for key in SLICE_KEYS], dtype=np.int64
        )
        for bs_idx in range(len(BS_NAMES)):
            for slice_idx, slice_key in enumerate(SLICE_KEYS):
                queue = self.queues[bs_idx][slice_key]
                if servers[bs_idx, slice_idx] <= 0 or not queue:
                    continue

                active: List[TaskState] = []
                for _ in range(min(int(servers[bs_idx, slice_idx]), len(queue))):
                    active.append(queue.popleft())

                unfinished: List[TaskState] = []
                for task in active:
                    user_idx = task.user_idx
                    signal_mw = tx_mw[bs_idx, slice_idx] * self.data.channel_gain[bs_idx, ms, user_idx]
                    interference_mw = 0.0
                    for other_idx in range(len(BS_NAMES)):
                        if other_idx == bs_idx:
                            continue
                        for other_slice_idx in range(len(SLICE_KEYS)):
                            rho = overlap_ratio[bs_idx, other_idx, slice_idx, other_slice_idx]
                            if rho <= 0.0:
                                continue
                            interference_mw += (
                                rho
                                * tx_mw[other_idx, other_slice_idx]
                                * self.data.channel_gain[other_idx, ms, user_idx]
                            )
                    self._record_interference(window, signal_mw, interference_mw, self.noise_mw[slice_idx])
                    self.stats.interference_ratio_sum += interference_mw / max(
                        signal_mw + interference_mw + self.noise_mw[slice_idx], EPS
                    )
                    self.stats.interference_samples += 1

                    sinr = signal_mw / max(interference_mw + self.noise_mw[slice_idx], EPS)
                    bits_this_ms = (
                        SLICE_CONFIGS[slice_key].rb_granularity * W_RB * math.log2(1.0 + sinr) / 1000.0
                    )
                    if bits_this_ms <= EPS:
                        task.served_ms += 1.0
                        unfinished.append(task)
                        continue

                    if task.remaining_bits <= bits_this_ms + EPS:
                        fraction = task.remaining_bits / bits_this_ms
                        task.served_ms += fraction
                        self._resolve_task(task, ms + fraction, terminal_drop=False, window=window)
                    else:
                        task.remaining_bits -= bits_this_ms
                        task.served_ms += 1.0
                        unfinished.append(task)

                for task in reversed(unfinished):
                    queue.appendleft(task)

    def _window_reward(self, window: WindowAccumulator) -> float:
        per_slice_proxy = {}
        success_ratios = []
        for slice_key in SLICE_KEYS:
            scores = window.proxy_scores[slice_key]
            per_slice_proxy[slice_key] = float(np.mean(scores)) if scores else 0.0
            resolved = max(window.resolved_counts[slice_key], 1)
            success_ratios.append(window.success_counts[slice_key] / resolved)
        fairness = jain_index(success_ratios)
        reward = (
            self.weights["u"] * per_slice_proxy["u"]
            + self.weights["e"] * per_slice_proxy["e"]
            + self.weights["m"] * per_slice_proxy["m"]
            - self.lambda_power * window.average_power_w()
            - self.lambda_interference * window.average_interference_ratio()
            + self.lambda_fairness * fairness
        )
        return reward

    def _terminal_drop(self, window: WindowAccumulator) -> None:
        for bs_idx in range(len(BS_NAMES)):
            for slice_key in SLICE_KEYS:
                queue = self.queues[bs_idx][slice_key]
                while queue:
                    task = queue.popleft()
                    self._resolve_task(task, float(TOTAL_MS), terminal_drop=True, window=window)

    def step(self, action_ids: np.ndarray, powers_dbm: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict[str, object]]:
        allocations = np.asarray([self.action_space[int(idx)] for idx in action_ids], dtype=np.int64)
        powers_dbm = np.clip(np.asarray(powers_dbm, dtype=np.float64), POWER_MIN_DBM, POWER_MAX_DBM)
        intervals = self._slice_intervals(allocations)
        overlap_ratio = self._overlap_ratio(allocations, intervals)
        window = WindowAccumulator()
        start_ms = self.current_ms
        end_ms = min(self.current_ms + DECISION_MS, TOTAL_MS)

        for ms in range(start_ms, end_ms):
            for bs_idx in range(len(BS_NAMES)):
                for slice_key in SLICE_KEYS:
                    self._drop_expired_tasks(bs_idx, slice_key, ms, window)
            self._append_arrivals(ms)
            self._service_one_ms(ms, allocations, powers_dbm, overlap_ratio, window)

        self.current_ms = end_ms
        done = self.current_ms >= TOTAL_MS
        if done:
            self._terminal_drop(window)

        reward = self._window_reward(window)
        self.prev_action_ids = action_ids.astype(np.int64).copy()
        self.prev_powers = powers_dbm.copy()
        decision_info = {
            "step": start_ms // DECISION_MS,
            "start_ms": start_ms,
            "slice_actions": allocations.tolist(),
            "powers_dbm": powers_dbm.round(4).tolist(),
            "window_reward": reward,
            "resolved": window.resolved_counts.copy(),
            "completed": window.success_counts.copy(),
            "mean_proxy": {
                key: (float(np.mean(window.proxy_scores[key])) if window.proxy_scores[key] else 0.0)
                for key in SLICE_KEYS
            },
            "avg_power_w": window.average_power_w(),
            "avg_interference_ratio": window.average_interference_ratio(),
        }
        self.stats.decisions.append(decision_info)
        return self.global_observation(), reward, done, {
            "official_objective": self.stats.objective_value,
            "decision": decision_info,
        }

    def summarize(self) -> Dict[str, object]:
        avg_delay = {
            key: (
                self.stats.delay_sum_ms[key] / self.stats.completed[key]
                if self.stats.completed[key]
                else None
            )
            for key in SLICE_KEYS
        }
        embb_rate = (
            self.stats.embb_rate_sum_mbps / self.stats.completed["e"]
            if self.stats.completed["e"]
            else None
        )
        return {
            "objective": self.stats.objective_value,
            "completed": self.stats.completed,
            "dropped": self.stats.dropped,
            "average_delay_ms": avg_delay,
            "embb_average_service_rate_mbps": embb_rate,
            "mean_power_w": (
                self.stats.power_w_sum / max(self.stats.power_samples, 1)
                if self.stats.power_samples
                else 0.0
            ),
            "mean_interference_ratio": (
                self.stats.interference_ratio_sum / max(self.stats.interference_samples, 1)
                if self.stats.interference_samples
                else 0.0
            ),
            "decisions": self.stats.decisions,
            "total_arrivals": self.data.total_arrivals,
            "assumption": (
                "service BS is assigned as the nearest micro base station at task arrival time; "
                "Q4 should replace this with an explicit access policy."
            ),
        }


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits)


class SharedLinearActorCritic:
    def __init__(
        self,
        local_obs_dim: int,
        global_obs_dim: int,
        num_actions: int,
        rng: np.random.Generator,
        init_scale: float = 0.05,
    ) -> None:
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.num_actions = num_actions
        self.W_slice = rng.normal(0.0, init_scale, size=(num_actions, local_obs_dim))
        self.b_slice = np.zeros(num_actions, dtype=np.float64)
        self.W_power = rng.normal(0.0, init_scale, size=(len(SLICE_KEYS), local_obs_dim))
        self.b_power = np.zeros(len(SLICE_KEYS), dtype=np.float64)
        self.log_std = np.full(len(SLICE_KEYS), -0.4, dtype=np.float64)
        self.w_value = rng.normal(0.0, init_scale, size=global_obs_dim)
        self.b_value = 0.0

    def value(self, global_obs: np.ndarray) -> float:
        return float(np.dot(self.w_value, global_obs) + self.b_value)

    def act(
        self,
        local_obs_batch: np.ndarray,
        global_obs: np.ndarray,
        rng: np.random.Generator,
        greedy: bool = False,
    ) -> Dict[str, object]:
        action_ids = []
        powers_dbm = []
        cache = {
            "local_obs": local_obs_batch.copy(),
            "global_obs": global_obs.copy(),
            "slice_probs": [],
            "slice_actions": [],
            "power_raw": [],
            "power_mean_raw": [],
            "value": self.value(global_obs),
        }
        std = np.exp(self.log_std)
        for obs in local_obs_batch:
            logits = self.W_slice @ obs + self.b_slice
            probs = softmax(logits)
            if greedy:
                action_id = int(np.argmax(probs))
            else:
                action_id = int(rng.choice(self.num_actions, p=probs))

            mean_raw = self.W_power @ obs + self.b_power
            if greedy:
                raw = mean_raw
            else:
                raw = mean_raw + std * rng.normal(size=len(SLICE_KEYS))
            power = POWER_MID_DBM + POWER_HALF_RANGE_DB * np.tanh(raw)

            action_ids.append(action_id)
            powers_dbm.append(power)
            cache["slice_probs"].append(probs)
            cache["slice_actions"].append(action_id)
            cache["power_raw"].append(raw)
            cache["power_mean_raw"].append(mean_raw)

        cache["slice_probs"] = np.asarray(cache["slice_probs"], dtype=np.float64)
        cache["slice_actions"] = np.asarray(cache["slice_actions"], dtype=np.int64)
        cache["power_raw"] = np.asarray(cache["power_raw"], dtype=np.float64)
        cache["power_mean_raw"] = np.asarray(cache["power_mean_raw"], dtype=np.float64)
        return {
            "action_ids": np.asarray(action_ids, dtype=np.int64),
            "powers_dbm": np.asarray(powers_dbm, dtype=np.float64),
            "cache": cache,
        }

    def update(
        self,
        trajectory: List[Dict[str, object]],
        gamma: float,
        actor_lr: float,
        critic_lr: float,
    ) -> Dict[str, float]:
        returns = []
        running = 0.0
        for step in reversed(trajectory):
            running = float(step["reward"]) + gamma * running
            returns.append(running)
        returns = np.asarray(list(reversed(returns)), dtype=np.float64)
        values = np.asarray([float(step["cache"]["value"]) for step in trajectory], dtype=np.float64)
        advantages = returns - values
        if advantages.size > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        grad_W_slice = np.zeros_like(self.W_slice)
        grad_b_slice = np.zeros_like(self.b_slice)
        grad_W_power = np.zeros_like(self.W_power)
        grad_b_power = np.zeros_like(self.b_power)
        grad_log_std = np.zeros_like(self.log_std)
        grad_w_value = np.zeros_like(self.w_value)
        grad_b_value = 0.0

        std = np.exp(self.log_std)
        std_sq = np.maximum(std * std, 1e-6)
        for step_idx, step in enumerate(trajectory):
            cache = step["cache"]
            global_obs = cache["global_obs"]
            value = float(cache["value"])
            ret = returns[step_idx]
            adv = advantages[step_idx]

            delta_v = ret - value
            grad_w_value += delta_v * global_obs
            grad_b_value += delta_v

            for agent_idx in range(len(BS_NAMES)):
                obs = cache["local_obs"][agent_idx]
                probs = cache["slice_probs"][agent_idx]
                chosen = int(cache["slice_actions"][agent_idx])
                dlogits = -probs
                dlogits[chosen] += 1.0
                grad_W_slice += adv * np.outer(dlogits, obs)
                grad_b_slice += adv * dlogits

                raw = cache["power_raw"][agent_idx]
                mean_raw = cache["power_mean_raw"][agent_idx]
                dmean = (raw - mean_raw) / std_sq
                grad_W_power += adv * np.outer(dmean, obs)
                grad_b_power += adv * dmean
                grad_log_std += adv * (((raw - mean_raw) ** 2) / std_sq - 1.0)

        scale = 1.0 / max(len(trajectory), 1)
        self.W_slice += actor_lr * scale * grad_W_slice
        self.b_slice += actor_lr * scale * grad_b_slice
        self.W_power += actor_lr * scale * grad_W_power
        self.b_power += actor_lr * scale * grad_b_power
        self.log_std = np.clip(self.log_std + actor_lr * scale * grad_log_std, -2.5, 1.0)
        self.w_value += critic_lr * scale * grad_w_value
        self.b_value += critic_lr * scale * grad_b_value
        return {
            "return_mean": float(np.mean(returns)) if returns.size else 0.0,
            "adv_mean": float(np.mean(advantages)) if advantages.size else 0.0,
        }

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            W_slice=self.W_slice,
            b_slice=self.b_slice,
            W_power=self.W_power,
            b_power=self.b_power,
            log_std=self.log_std,
            w_value=self.w_value,
            b_value=np.asarray([self.b_value], dtype=np.float64),
        )

    def load(self, path: Path) -> None:
        payload = np.load(path)
        self.W_slice = payload["W_slice"]
        self.b_slice = payload["b_slice"]
        self.W_power = payload["W_power"]
        self.b_power = payload["b_power"]
        self.log_std = payload["log_std"]
        self.w_value = payload["w_value"]
        self.b_value = float(payload["b_value"][0])


def run_episode(
    env: MultiBSQ3Env,
    policy: SharedLinearActorCritic,
    rng: np.random.Generator,
    greedy: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    env.reset()
    done = False
    trajectory: List[Dict[str, object]] = []
    while not done:
        local_obs = env.local_observations()
        global_obs = env.global_observation()
        act_out = policy.act(local_obs, global_obs, rng=rng, greedy=greedy)
        _, reward, done, info = env.step(act_out["action_ids"], act_out["powers_dbm"])
        trajectory.append(
            {
                "cache": act_out["cache"],
                "reward": reward,
                "info": info,
            }
        )
    return trajectory, env.summarize()


def train(
    env: MultiBSQ3Env,
    policy: SharedLinearActorCritic,
    episodes: int,
    gamma: float,
    actor_lr: float,
    critic_lr: float,
    eval_every: int,
    eval_episodes: int,
    checkpoint_out: Path | None,
    rng: np.random.Generator,
) -> Dict[str, object]:
    history: List[Dict[str, object]] = []
    best_eval = -float("inf")
    best_summary: Dict[str, object] | None = None
    start = time.perf_counter()
    for episode in range(1, episodes + 1):
        trajectory, summary = run_episode(env, policy, rng=rng, greedy=False)
        update_stats = policy.update(
            trajectory=trajectory,
            gamma=gamma,
            actor_lr=actor_lr,
            critic_lr=critic_lr,
        )
        row = {
            "episode": episode,
            "train_objective": summary["objective"],
            "mean_power_w": summary["mean_power_w"],
            "mean_interference_ratio": summary["mean_interference_ratio"],
            "completed": summary["completed"],
            "dropped": summary["dropped"],
            "update": update_stats,
        }
        history.append(row)
        if episode == 1 or episode % eval_every == 0 or episode == episodes:
            eval_scores = []
            eval_summary = None
            for _ in range(eval_episodes):
                _, eval_summary = run_episode(env, policy, rng=rng, greedy=True)
                eval_scores.append(eval_summary["objective"])
            mean_eval = float(np.mean(eval_scores))
            row["eval_objective"] = mean_eval
            if mean_eval > best_eval:
                best_eval = mean_eval
                best_summary = eval_summary
                if checkpoint_out is not None:
                    policy.save(checkpoint_out)

            print(
                f"episode={episode:04d} "
                f"train_obj={summary['objective']:.6f} "
                f"eval_obj={mean_eval:.6f} "
                f"power_w={summary['mean_power_w']:.4f} "
                f"interf={summary['mean_interference_ratio']:.4f}"
            )
        else:
            print(
                f"episode={episode:04d} "
                f"train_obj={summary['objective']:.6f} "
                f"power_w={summary['mean_power_w']:.4f} "
                f"interf={summary['mean_interference_ratio']:.4f}"
            )

    elapsed = time.perf_counter() - start
    return {
        "history": history,
        "best_eval_objective": best_eval,
        "best_summary": best_summary,
        "runtime_sec": elapsed,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Question 3 reinforcement-learning baseline. "
            "The environment is a 100 ms decision / 1 ms service simulator over three "
            "micro base stations, and the policy is a zero-dependency shared-parameter "
            "linear actor-critic that outputs discrete slice budgets plus continuous power."
        )
    )
    parser.add_argument("--episodes", type=int, default=20, help="Training episodes.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--gamma", type=float, default=0.98, help="Discount factor.")
    parser.add_argument("--actor-lr", type=float, default=0.02, help="Actor learning rate.")
    parser.add_argument("--critic-lr", type=float, default=0.01, help="Critic learning rate.")
    parser.add_argument("--eval-every", type=int, default=5, help="Evaluate greedily every N episodes.")
    parser.add_argument("--eval-episodes", type=int, default=2, help="Greedy evaluation episodes.")
    parser.add_argument("--wu", type=float, default=1.0 / 3.0, help="URLLC weight.")
    parser.add_argument("--we", type=float, default=1.0 / 3.0, help="eMBB weight.")
    parser.add_argument("--wm", type=float, default=1.0 / 3.0, help="mMTC weight.")
    parser.add_argument("--lambda-power", type=float, default=0.02, help="Training power regularizer.")
    parser.add_argument(
        "--lambda-interference",
        type=float,
        default=0.05,
        help="Training interference regularizer.",
    )
    parser.add_argument("--lambda-fairness", type=float, default=0.02, help="Training fairness bonus.")
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=ROOT / "outputs" / "q3_rl" / "q3_linear_actor_critic_best.npz",
        help="Optional checkpoint output path.",
    )
    parser.add_argument(
        "--metrics-out",
        type=Path,
        default=ROOT / "outputs" / "q3_rl" / "q3_training_metrics.json",
        help="Optional JSON metrics output path.",
    )
    parser.add_argument(
        "--checkpoint-in",
        type=Path,
        default=None,
        help="If set, skip training and only run greedy evaluation from this checkpoint.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    weight_sum = args.wu + args.we + args.wm
    if abs(weight_sum - 1.0) > 1e-9:
        raise SystemExit("Weights must sum to 1.")
    if args.episodes <= 0:
        raise SystemExit("--episodes must be positive.")
    if args.eval_every <= 0 or args.eval_episodes <= 0:
        raise SystemExit("--eval-every and --eval-episodes must be positive.")

    rng = np.random.default_rng(args.seed)
    data = load_q3_data()
    env = MultiBSQ3Env(
        data=data,
        weights={"u": args.wu, "e": args.we, "m": args.wm},
        lambda_power=args.lambda_power,
        lambda_interference=args.lambda_interference,
        lambda_fairness=args.lambda_fairness,
    )
    policy = SharedLinearActorCritic(
        local_obs_dim=env.local_obs_dim,
        global_obs_dim=env.global_obs_dim,
        num_actions=len(env.action_space),
        rng=rng,
    )

    if args.checkpoint_in is not None:
        policy.load(args.checkpoint_in)
        _, summary = run_episode(env, policy, rng=rng, greedy=True)
        print("Q3 greedy evaluation")
        print(f"checkpoint={args.checkpoint_in}")
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    result = train(
        env=env,
        policy=policy,
        episodes=args.episodes,
        gamma=args.gamma,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        eval_every=args.eval_every,
        eval_episodes=args.eval_episodes,
        checkpoint_out=args.checkpoint_out,
        rng=rng,
    )
    payload = {
        "config": {
            "episodes": args.episodes,
            "seed": args.seed,
            "gamma": args.gamma,
            "actor_lr": args.actor_lr,
            "critic_lr": args.critic_lr,
            "weights": {"u": args.wu, "e": args.we, "m": args.wm},
            "lambda_power": args.lambda_power,
            "lambda_interference": args.lambda_interference,
            "lambda_fairness": args.lambda_fairness,
        },
        **result,
    }
    if args.metrics_out is not None:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        with args.metrics_out.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)

    print()
    print("Q3 training finished")
    print(f"best_eval_objective={result['best_eval_objective']:.6f}")
    print(f"checkpoint_out={args.checkpoint_out}")
    print(f"metrics_out={args.metrics_out}")
    if result["best_summary"] is not None:
        print("best_summary")
        print(json.dumps(result["best_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
