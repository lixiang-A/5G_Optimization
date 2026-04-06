#!/usr/bin/env python3
import argparse
import heapq
import math
import re
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
XLSX_PATH = "/Users/liz/Desktop/5G_Optimization/channel_data等2个文件/channel_data.xlsx"
P_DBM = 30.0
W_RB = 360e3
NF_DB = 7.0


@dataclass
class SliceConfig:
    name: str
    prefix: str
    rb_granularity: int
    rate_sla_mbps: float
    delay_sla_ms: float
    penalty: float
    alpha: Optional[float] = None


@dataclass
class Job:
    user: str
    slice_name: str
    proc_ms: float
    rate_mbps: float


SLICE_CONFIGS = {
    "u": SliceConfig("URLLC", "U", 10, 10.0, 5.0, 5.0, alpha=0.95),
    "e": SliceConfig("eMBB", "e", 5, 50.0, 100.0, 3.0),
    "m": SliceConfig("mMTC", "m", 2, 1.0, 500.0, 1.0),
}


def col_idx_to_name(idx: int) -> str:
    name = ""
    while idx:
        idx, rem = divmod(idx - 1, 26)
        name = chr(65 + rem) + name
    return name


def read_xlsx_tables(path: str) -> Dict[str, List[Dict[str, str]]]:
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
            sheet_rows = []
            for row in root.find("a:sheetData", NS).findall("a:row", NS):
                values: dict[str, str] = {}
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
            rows = []
            for raw_row in sheet_rows[1:]:
                row = {
                    header: raw_row.get(col_idx_to_name(i), "")
                    for i, header in enumerate(headers, start=1)
                    if header
                }
                rows.append(row)
            tables[sheet.attrib["name"]] = rows
    return tables


def classify_slice(user: str) -> SliceConfig:
    if user.startswith("U"):
        return SLICE_CONFIGS["u"]
    return SLICE_CONFIGS[user[0]]


def build_jobs(path: str) -> Dict[str, List[Job]]:
    tables = read_xlsx_tables(path)
    pathloss_row = tables["大规模衰减"][0]
    fading_row = tables["小规模瑞丽衰减"][0]
    task_row = tables["任务流"][0]

    jobs = {"u": [], "e": [], "m": []}
    for user, pathloss in pathloss_row.items():
        if user == "Time":
            continue
        config = classify_slice(user)
        pathloss_db = float(pathloss)
        fading = float(fading_row[user])
        size_mbit = float(task_row[user])

        signal_mw = 10 ** ((P_DBM - pathloss_db) / 10.0) * (fading**2)
        noise_dbm = -174.0 + 10.0 * math.log10(config.rb_granularity * W_RB) + NF_DB
        noise_mw = 10 ** (noise_dbm / 10.0)
        sinr = signal_mw / noise_mw
        rate_bps = config.rb_granularity * W_RB * math.log2(1.0 + sinr)
        proc_ms = size_mbit * 1e6 / rate_bps * 1e3

        jobs[user[0].lower() if not user.startswith("U") else "u"].append(
            Job(user=user, slice_name=config.name, proc_ms=proc_ms, rate_mbps=rate_bps / 1e6)
        )

    return jobs


def list_schedule(jobs: List[Job], servers: int) -> List[Tuple[Job, float]]:
    if servers <= 0:
        return []
    machine_heap = [0.0] * servers
    schedule = []
    for job in sorted(jobs, key=lambda item: item.proc_ms):
        start = heapq.heappop(machine_heap)
        finish = start + job.proc_ms
        schedule.append((job, finish))
        heapq.heappush(machine_heap, finish)
    return schedule


def score_urllc(schedule: List[Tuple[Job, float]], config: SliceConfig) -> float:
    if not schedule:
        return -config.penalty
    qos = []
    for job, finish_ms in schedule:
        if finish_ms <= config.delay_sla_ms:
            qos.append(config.alpha ** finish_ms)
        else:
            qos.append(-config.penalty)
    return sum(qos) / len(qos)


def score_embb(schedule: List[Tuple[Job, float]], config: SliceConfig) -> float:
    if not schedule:
        return -config.penalty
    qos = []
    for job, finish_ms in schedule:
        if finish_ms > config.delay_sla_ms:
            qos.append(-config.penalty)
        else:
            qos.append(min(job.rate_mbps / config.rate_sla_mbps, 1.0))
    return sum(qos) / len(qos)


def score_mmtc(
    schedule: List[Tuple[Job, float]], total_jobs: int, config: SliceConfig
) -> float:
    if not schedule:
        return -config.penalty
    completed = sum(1 for _, finish_ms in schedule if finish_ms <= config.delay_sla_ms)
    if completed != total_jobs:
        return -config.penalty
    return completed / total_jobs


def enumerate_allocations() -> List[Tuple[int, int, int]]:
    allocations = []
    for x_u in range(0, 51, 10):
        for x_e in range(0, 51 - x_u, 5):
            x_m = 50 - x_u - x_e
            if x_m >= 0 and x_m % 2 == 0:
                allocations.append((x_u, x_e, x_m))
    return allocations


def main() -> None:
    parser = argparse.ArgumentParser(description="Solve Q1 by exhaustive enumeration.")
    parser.add_argument("--wu", type=float, default=1 / 3, help="URLLC weight")
    parser.add_argument("--we", type=float, default=1 / 3, help="eMBB weight")
    parser.add_argument("--wm", type=float, default=1 / 3, help="mMTC weight")
    args = parser.parse_args()

    weight_sum = args.wu + args.we + args.wm
    if abs(weight_sum - 1.0) > 1e-9:
        raise SystemExit("Weights must sum to 1.")

    jobs = build_jobs(XLSX_PATH)
    print("Per-user transmission profile")
    for key in ("u", "e", "m"):
        for job in sorted(jobs[key], key=lambda item: item.user):
            print(
                f"{job.user:>4}  {job.slice_name:<6}  rate={job.rate_mbps:8.3f} Mbps"
                f"  tx_time={job.proc_ms:7.3f} ms"
            )

    print("\nEnumerated allocations")
    rows = []
    for x_u, x_e, x_m in enumerate_allocations():
        u_schedule = list_schedule(jobs["u"], x_u // SLICE_CONFIGS["u"].rb_granularity)
        e_schedule = list_schedule(jobs["e"], x_e // SLICE_CONFIGS["e"].rb_granularity)
        m_schedule = list_schedule(jobs["m"], x_m // SLICE_CONFIGS["m"].rb_granularity)

        q_u = score_urllc(u_schedule, SLICE_CONFIGS["u"])
        q_e = score_embb(e_schedule, SLICE_CONFIGS["e"])
        q_m = score_mmtc(m_schedule, len(jobs["m"]), SLICE_CONFIGS["m"])
        utility = args.wu * q_u + args.we * q_e + args.wm * q_m

        rows.append((utility, x_u, x_e, x_m, q_u, q_e, q_m))

    rows.sort(reverse=True)
    for utility, x_u, x_e, x_m, q_u, q_e, q_m in rows:
        print(
            f"x=({x_u:>2}, {x_e:>2}, {x_m:>2})"
            f"  Q_u={q_u:8.6f}  Q_e={q_e:8.6f}  Q_m={q_m:8.6f}"
            f"  U={utility:8.6f}"
        )

    best_utility = rows[0][0]
    best_rows = [row for row in rows if abs(row[0] - best_utility) <= 1e-12]
    print("\nOptimal allocations")
    for utility, x_u, x_e, x_m, q_u, q_e, q_m in best_rows:
        print(
            f"x=({x_u}, {x_e}, {x_m})"
            f"  Q_u={q_u:.6f}  Q_e={q_e:.6f}  Q_m={q_m:.6f}  U={utility:.6f}"
        )


if __name__ == "__main__":
    main()
