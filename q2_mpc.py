#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import time
import zipfile
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Sequence, Tuple


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
ROOT = Path(__file__).resolve().parent
DEFAULT_XLSX = ROOT / "channel_data等2个文件(1)" / "channel_data.xlsx"
SLICE_KEYS = ("u", "e", "m")
TOTAL_RBS = 50
DECISION_MS = 100
TOTAL_MS = 1000
P_DBM = 30.0
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
    slice_key: str
    arrival_ms: int
    size_bits: float


@dataclass
class TaskState:
    task_id: int
    user_idx: int
    slice_key: str
    arrival_ms: int
    size_bits: float
    remaining_bits: float
    served_ms: float = 0.0

    def clone(self) -> "TaskState":
        return TaskState(
            task_id=self.task_id,
            user_idx=self.user_idx,
            slice_key=self.slice_key,
            arrival_ms=self.arrival_ms,
            size_bits=self.size_bits,
            remaining_bits=self.remaining_bits,
            served_ms=self.served_ms,
        )


@dataclass
class SimState:
    current_ms: int
    queues: Dict[str, Deque[TaskState]]

    def clone(self) -> "SimState":
        return SimState(
            current_ms=self.current_ms,
            queues={key: deque(task.clone() for task in queue) for key, queue in self.queues.items()},
        )


@dataclass
class TrajectoryStats:
    objective_value: float = 0.0
    reward_sum: Dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in SLICE_KEYS})
    completed: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in SLICE_KEYS})
    dropped: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in SLICE_KEYS})
    delay_sum_ms: Dict[str, float] = field(default_factory=lambda: {key: 0.0 for key in SLICE_KEYS})
    embb_rate_sum_mbps: float = 0.0
    terminal_drops: Dict[str, int] = field(default_factory=lambda: {key: 0 for key in SLICE_KEYS})


@dataclass
class DecisionRecord:
    step: int
    start_ms: int
    action: Tuple[int, int, int]
    lookahead_depth: int
    projected_value: float
    realized_delta: float
    queue_before: Dict[str, int]
    queue_after: Dict[str, int]


@dataclass(frozen=True)
class Q2Data:
    users: Tuple[str, ...]
    rates_bps: Tuple[Tuple[float, ...], ...]
    arrivals_by_ms: Tuple[Tuple[ArrivalSpec, ...], ...]
    total_arrivals: Dict[str, int]


SLICE_CONFIGS: Dict[str, SliceConfig] = {
    "u": SliceConfig("URLLC", "U", 10, 10.0, 5, 5.0, alpha=0.95),
    "e": SliceConfig("eMBB", "e", 5, 50.0, 100, 3.0),
    "m": SliceConfig("mMTC", "m", 2, 1.0, 500, 1.0),
}


def col_idx_to_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def read_xlsx_tables(path: Path) -> Dict[str, List[Dict[str, str]]]:
    tables: Dict[str, List[Dict[str, str]]] = {}
    with zipfile.ZipFile(path) as zf:
        workbook = ET.fromstring(zf.read("xl/workbook.xml"))
        rels = ET.fromstring(zf.read("xl/_rels/workbook.xml.rels"))
        rel_map = {rel.attrib["Id"]: rel.attrib["Target"] for rel in rels}
        shared_strings: List[str] = []
        if "xl/sharedStrings.xml" in zf.namelist():
            shared_root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in shared_root.findall("a:si", NS):
                text = "".join(node.text or "" for node in si.iterfind(".//a:t", NS))
                shared_strings.append(text)

        for sheet in workbook.find("a:sheets", NS):
            rel_id = sheet.attrib[
                "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
            ]
            target = rel_map[rel_id]
            sheet_path = target if target.startswith("xl/") else f"xl/{target}"
            root = ET.fromstring(zf.read(sheet_path))
            sheet_rows: List[Dict[str, str]] = []
            for row in root.find("a:sheetData", NS).findall("a:row", NS):
                values: Dict[str, str] = {}
                for cell in row.findall("a:c", NS):
                    ref = cell.attrib.get("r", "")
                    match = re.match(r"([A-Z]+)(\d+)", ref)
                    if not match:
                        continue
                    col = match.group(1)
                    cell_type = cell.attrib.get("t")
                    value_node = cell.find("a:v", NS)
                    value = "" if value_node is None else value_node.text or ""
                    if cell_type == "s" and value:
                        value = shared_strings[int(value)]
                    elif cell_type == "inlineStr":
                        value = "".join(
                            node.text or "" for node in cell.iterfind(".//a:t", NS)
                        )
                    values[col] = value
                sheet_rows.append(values)

            if not sheet_rows:
                tables[sheet.attrib["name"]] = []
                continue

            max_idx = max(
                sum((ord(ch) - 64) * (26 ** pos) for pos, ch in enumerate(col[::-1]))
                for col in sheet_rows[0]
            )
            headers = [sheet_rows[0].get(col_idx_to_name(i), "") for i in range(1, max_idx + 1)]
            rows: List[Dict[str, str]] = []
            for raw_row in sheet_rows[1:]:
                row = {
                    header: raw_row.get(col_idx_to_name(i), "")
                    for i, header in enumerate(headers, start=1)
                    if header
                }
                rows.append(row)
            tables[sheet.attrib["name"]] = rows
    return tables


def user_sort_key(user: str) -> Tuple[int, int]:
    if user.startswith("U"):
        return (0, int(user[1:]))
    if user.startswith("e"):
        return (1, int(user[1:]))
    return (2, int(user[1:]))


def classify_slice(user: str) -> str:
    return "u" if user.startswith("U") else user[0]


def build_action_space() -> List[Tuple[int, int, int]]:
    actions: List[Tuple[int, int, int]] = []
    for x_u in range(0, TOTAL_RBS + 1, SLICE_CONFIGS["u"].rb_granularity):
        for x_e in range(0, TOTAL_RBS - x_u + 1, SLICE_CONFIGS["e"].rb_granularity):
            x_m = TOTAL_RBS - x_u - x_e
            if x_m >= 0 and x_m % SLICE_CONFIGS["m"].rb_granularity == 0:
                actions.append((x_u, x_e, x_m))
    return actions


def load_q2_data(path: Path) -> Q2Data:
    tables = read_xlsx_tables(path)
    pathloss_rows = tables["大规模衰减"]
    fading_rows = tables["小规模瑞丽衰减"]
    task_rows = tables["用户任务流"]

    users = tuple(sorted((user for user in task_rows[0] if user != "Time"), key=user_sort_key))
    user_index = {user: idx for idx, user in enumerate(users)}

    rates_bps: List[Tuple[float, ...]] = []
    for ms in range(len(pathloss_rows)):
        row_rates = [0.0] * len(users)
        for user in users:
            cfg = SLICE_CONFIGS[classify_slice(user)]
            pathloss_db = float(pathloss_rows[ms][user])
            fading = float(fading_rows[ms][user])
            signal_mw = 10 ** ((P_DBM - pathloss_db) / 10.0) * (fading * fading)
            noise_dbm = -174.0 + 10.0 * math.log10(cfg.rb_granularity * W_RB) + NF_DB
            noise_mw = 10 ** (noise_dbm / 10.0)
            sinr = signal_mw / noise_mw
            rate = cfg.rb_granularity * W_RB * math.log2(1.0 + sinr)
            row_rates[user_index[user]] = rate
        rates_bps.append(tuple(row_rates))

    arrivals_by_ms: List[List[ArrivalSpec]] = [[] for _ in range(len(task_rows))]
    total_arrivals = {key: 0 for key in SLICE_KEYS}
    next_task_id = 0
    for ms, row in enumerate(task_rows):
        for user in users:
            size_mbit = float(row[user])
            if size_mbit <= 0.0:
                continue
            slice_key = classify_slice(user)
            arrivals_by_ms[ms].append(
                ArrivalSpec(
                    task_id=next_task_id,
                    user_idx=user_index[user],
                    slice_key=slice_key,
                    arrival_ms=ms,
                    size_bits=size_mbit * 1e6,
                )
            )
            total_arrivals[slice_key] += 1
            next_task_id += 1

    arrivals_tuple = tuple(tuple(items) for items in arrivals_by_ms)
    rates_tuple = tuple(rates_bps)
    return Q2Data(
        users=users,
        rates_bps=rates_tuple,
        arrivals_by_ms=arrivals_tuple,
        total_arrivals=total_arrivals,
    )


def queue_lengths(state: SimState) -> Dict[str, int]:
    return {key: len(state.queues[key]) for key in SLICE_KEYS}


class RollingQ2Planner:
    def __init__(
        self,
        data: Q2Data,
        weights: Dict[str, float],
        gamma: float = 1.0,
    ) -> None:
        self.data = data
        self.weights = weights
        self.gamma = gamma
        self.actions = build_action_space()

    def initial_state(self) -> SimState:
        return SimState(current_ms=0, queues={key: deque() for key in SLICE_KEYS})

    def _objective_delta(self, slice_key: str, task_score: float, decision_idx: int) -> float:
        total = self.data.total_arrivals[slice_key]
        if total <= 0:
            return 0.0
        return (self.gamma ** decision_idx) * self.weights[slice_key] * task_score / total

    def _resolve_task(
        self,
        task: TaskState,
        finish_time_ms: float,
        *,
        terminal_drop: bool,
        stats: TrajectoryStats | None,
        decision_idx: int,
    ) -> float:
        cfg = SLICE_CONFIGS[task.slice_key]
        delay_ms = finish_time_ms - task.arrival_ms

        if terminal_drop or delay_ms > cfg.delay_sla_ms + EPS:
            score = -cfg.penalty
            if stats is not None:
                stats.dropped[task.slice_key] += 1
                if terminal_drop:
                    stats.terminal_drops[task.slice_key] += 1
        elif task.slice_key == "u":
            score = cfg.alpha ** delay_ms
            if stats is not None:
                stats.completed[task.slice_key] += 1
                stats.delay_sum_ms[task.slice_key] += delay_ms
        elif task.slice_key == "e":
            effective_rate_mbps = (
                task.size_bits / (task.served_ms / 1000.0) / 1e6 if task.served_ms > EPS else 0.0
            )
            score = min(effective_rate_mbps / cfg.rate_sla_mbps, 1.0)
            if stats is not None:
                stats.completed[task.slice_key] += 1
                stats.delay_sum_ms[task.slice_key] += delay_ms
                stats.embb_rate_sum_mbps += effective_rate_mbps
        else:
            score = 1.0
            if stats is not None:
                stats.completed[task.slice_key] += 1
                stats.delay_sum_ms[task.slice_key] += delay_ms

        if stats is not None:
            stats.reward_sum[task.slice_key] += score
            stats.objective_value += self._objective_delta(task.slice_key, score, decision_idx)
        return self._objective_delta(task.slice_key, score, decision_idx)

    def _drop_expired_tasks(
        self,
        queue: Deque[TaskState],
        current_ms: int,
        *,
        stats: TrajectoryStats | None,
        decision_idx: int,
    ) -> Tuple[Deque[TaskState], float]:
        survivors: Deque[TaskState] = deque()
        delta = 0.0
        while queue:
            task = queue.popleft()
            deadline_ms = SLICE_CONFIGS[task.slice_key].delay_sla_ms
            if current_ms - task.arrival_ms >= deadline_ms:
                delta += self._resolve_task(
                    task,
                    float(current_ms),
                    terminal_drop=False,
                    stats=stats,
                    decision_idx=decision_idx,
                )
            else:
                survivors.append(task)
        return survivors, delta

    def _append_arrivals(self, state: SimState, ms: int) -> None:
        for spec in self.data.arrivals_by_ms[ms]:
            state.queues[spec.slice_key].append(
                TaskState(
                    task_id=spec.task_id,
                    user_idx=spec.user_idx,
                    slice_key=spec.slice_key,
                    arrival_ms=spec.arrival_ms,
                    size_bits=spec.size_bits,
                    remaining_bits=spec.size_bits,
                )
            )

    def _service_one_ms(
        self,
        state: SimState,
        ms: int,
        servers: Dict[str, int],
        *,
        stats: TrajectoryStats | None,
        decision_idx: int,
    ) -> float:
        delta = 0.0
        rates_row = self.data.rates_bps[ms]
        for slice_key in SLICE_KEYS:
            queue = state.queues[slice_key]
            if servers[slice_key] <= 0 or not queue:
                continue

            active: List[TaskState] = []
            for _ in range(min(servers[slice_key], len(queue))):
                active.append(queue.popleft())

            unfinished: List[TaskState] = []
            for task in active:
                bits_this_ms = rates_row[task.user_idx] / 1000.0
                if bits_this_ms <= EPS:
                    task.served_ms += 1.0
                    unfinished.append(task)
                    continue

                if task.remaining_bits <= bits_this_ms + EPS:
                    fraction = task.remaining_bits / bits_this_ms
                    task.served_ms += fraction
                    delta += self._resolve_task(
                        task,
                        ms + fraction,
                        terminal_drop=False,
                        stats=stats,
                        decision_idx=decision_idx,
                    )
                else:
                    task.remaining_bits -= bits_this_ms
                    task.served_ms += 1.0
                    unfinished.append(task)

            for task in reversed(unfinished):
                queue.appendleft(task)
        return delta

    def simulate_window(
        self,
        state: SimState,
        action: Tuple[int, int, int],
        *,
        stats: TrajectoryStats | None = None,
    ) -> Tuple[SimState, float]:
        next_state = state.clone()
        servers = {
            "u": action[0] // SLICE_CONFIGS["u"].rb_granularity,
            "e": action[1] // SLICE_CONFIGS["e"].rb_granularity,
            "m": action[2] // SLICE_CONFIGS["m"].rb_granularity,
        }
        decision_idx = next_state.current_ms // DECISION_MS
        window_end = min(next_state.current_ms + DECISION_MS, TOTAL_MS)
        delta = 0.0

        for ms in range(next_state.current_ms, window_end):
            for slice_key in SLICE_KEYS:
                filtered, expired_delta = self._drop_expired_tasks(
                    next_state.queues[slice_key],
                    ms,
                    stats=stats,
                    decision_idx=decision_idx,
                )
                next_state.queues[slice_key] = filtered
                delta += expired_delta

            self._append_arrivals(next_state, ms)
            delta += self._service_one_ms(
                next_state,
                ms,
                servers,
                stats=stats,
                decision_idx=decision_idx,
            )

        next_state.current_ms = window_end
        return next_state, delta

    def terminal_penalty(self, state: SimState, *, stats: TrajectoryStats | None = None) -> float:
        next_state = state.clone()
        decision_idx = max(TOTAL_MS // DECISION_MS - 1, 0)
        delta = 0.0
        for slice_key in SLICE_KEYS:
            queue = next_state.queues[slice_key]
            while queue:
                task = queue.popleft()
                delta += self._resolve_task(
                    task,
                    float(TOTAL_MS),
                    terminal_drop=True,
                    stats=stats,
                    decision_idx=decision_idx,
                )
        next_state.current_ms = TOTAL_MS
        return delta

    def _search(self, state: SimState, depth: int) -> Tuple[float, List[Tuple[int, int, int]]]:
        if depth <= 0:
            return 0.0, []

        best_value = -float("inf")
        best_sequence: List[Tuple[int, int, int]] = []
        for action in self.actions:
            next_state, delta = self.simulate_window(state, action)
            if next_state.current_ms >= TOTAL_MS:
                total_value = delta + self.terminal_penalty(next_state)
                suffix: List[Tuple[int, int, int]] = []
            elif depth == 1:
                total_value = delta
                suffix = []
            else:
                future_value, future_actions = self._search(next_state, depth - 1)
                total_value = delta + future_value
                suffix = future_actions

            if total_value > best_value + EPS:
                best_value = total_value
                best_sequence = [action] + suffix
        return best_value, best_sequence

    def solve(self, lookahead: int) -> Tuple[List[DecisionRecord], TrajectoryStats]:
        state = self.initial_state()
        stats = TrajectoryStats()
        records: List[DecisionRecord] = []

        for step in range(TOTAL_MS // DECISION_MS):
            depth = min(lookahead, TOTAL_MS // DECISION_MS - step)
            queue_before = queue_lengths(state)
            projected_value, sequence = self._search(state, depth)
            if not sequence:
                raise RuntimeError("search returned an empty action sequence")

            chosen_action = sequence[0]
            next_state, realized_delta = self.simulate_window(state, chosen_action, stats=stats)
            records.append(
                DecisionRecord(
                    step=step,
                    start_ms=step * DECISION_MS,
                    action=chosen_action,
                    lookahead_depth=depth,
                    projected_value=projected_value,
                    realized_delta=realized_delta,
                    queue_before=queue_before,
                    queue_after=queue_lengths(next_state),
                )
            )
            state = next_state

        stats.objective_value += self.terminal_penalty(state, stats=stats)
        return records, stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Solve Q2 with a rolling-horizon DP/MPC baseline. "
            "The 1 ms simulator is exact at the task level; the outer search exhaustively "
            "enumerates legal 100 ms slicing actions over a configurable lookahead depth. "
            "To close the 1000 ms finite horizon, unresolved backlog at t=1000 ms receives "
            "a terminal drop penalty."
        )
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=DEFAULT_XLSX,
        help="Path to channel_data.xlsx for Q2.",
    )
    parser.add_argument("--lookahead", type=int, default=2, help="DP/MPC lookahead depth in 100 ms decisions.")
    parser.add_argument("--wu", type=float, default=1.0 / 3.0, help="URLLC weight.")
    parser.add_argument("--we", type=float, default=1.0 / 3.0, help="eMBB weight.")
    parser.add_argument("--wm", type=float, default=1.0 / 3.0, help="mMTC weight.")
    parser.add_argument("--gamma", type=float, default=1.0, help="Optional window discount factor.")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Optional path for a JSON dump of actions and summary metrics.",
    )
    return parser.parse_args()


def summarize(records: Sequence[DecisionRecord], stats: TrajectoryStats, data: Q2Data) -> Dict[str, object]:
    slice_mean = {
        key: stats.reward_sum[key] / data.total_arrivals[key] if data.total_arrivals[key] else 0.0
        for key in SLICE_KEYS
    }
    avg_delay = {
        key: stats.delay_sum_ms[key] / stats.completed[key] if stats.completed[key] else None
        for key in SLICE_KEYS
    }
    embb_avg_rate = stats.embb_rate_sum_mbps / stats.completed["e"] if stats.completed["e"] else None
    return {
        "objective": stats.objective_value,
        "slice_mean_reward": slice_mean,
        "completed": stats.completed,
        "dropped": stats.dropped,
        "terminal_drops": stats.terminal_drops,
        "average_delay_ms": avg_delay,
        "embb_average_service_rate_mbps": embb_avg_rate,
        "total_arrivals": data.total_arrivals,
        "actions": [
            {
                "step": record.step,
                "start_ms": record.start_ms,
                "action": list(record.action),
                "lookahead_depth": record.lookahead_depth,
                "projected_value": record.projected_value,
                "realized_delta": record.realized_delta,
                "queue_before": record.queue_before,
                "queue_after": record.queue_after,
            }
            for record in records
        ],
    }


def main() -> None:
    args = parse_args()
    if not args.data_path.exists():
        raise SystemExit(f"Data file not found: {args.data_path}")
    if args.lookahead <= 0:
        raise SystemExit("--lookahead must be positive.")
    weight_sum = args.wu + args.we + args.wm
    if abs(weight_sum - 1.0) > 1e-9:
        raise SystemExit("Weights must sum to 1.")
    if args.gamma <= 0.0:
        raise SystemExit("--gamma must be positive.")

    started = time.perf_counter()
    data = load_q2_data(args.data_path)
    planner = RollingQ2Planner(
        data=data,
        weights={"u": args.wu, "e": args.we, "m": args.wm},
        gamma=args.gamma,
    )
    records, stats = planner.solve(lookahead=args.lookahead)
    summary = summarize(records, stats, data)
    elapsed = time.perf_counter() - started

    print("Q2 rolling DP/MPC results")
    print(f"data={args.data_path}")
    print(f"lookahead={args.lookahead}  gamma={args.gamma:.4f}")
    print(f"weights=(wu={args.wu:.4f}, we={args.we:.4f}, wm={args.wm:.4f})")
    print("terminal_policy=unfinished tasks at 1000 ms are penalized as terminal drops")
    print()
    print("Decision trajectory")
    for record in records:
        print(
            f"t={record.step:02d} [{record.start_ms:03d},{record.start_ms + DECISION_MS:04d})"
            f"  action={record.action}"
            f"  projected={record.projected_value: .6f}"
            f"  realized={record.realized_delta: .6f}"
            f"  q_before={record.queue_before}"
            f"  q_after={record.queue_after}"
        )

    print()
    print("Summary")
    print(f"objective={summary['objective']:.6f}")
    print(f"slice_mean_reward={summary['slice_mean_reward']}")
    print(f"completed={summary['completed']}")
    print(f"dropped={summary['dropped']}")
    print(f"terminal_drops={summary['terminal_drops']}")
    print(f"average_delay_ms={summary['average_delay_ms']}")
    print(f"embb_average_service_rate_mbps={summary['embb_average_service_rate_mbps']}")
    print(f"runtime_sec={elapsed:.3f}")

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        with args.json_out.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, ensure_ascii=False, indent=2)
        print(f"json_out={args.json_out}")


if __name__ == "__main__":
    main()
