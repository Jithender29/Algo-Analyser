"""
Run benchmarks based on the latest configuration stored in Backend/configs.jsonl
and generate interactive n vs time and n vs memory plots.

- Supports Algorithm Type = "sorting" with:
    - "bubble"    -> Bubble Sort
    - "insertion" -> Insertion Sort
    - "counting"  -> Counting Sort
    - "radix"     -> Radix Sort
    - "bucket"    -> Bucket Sort
- Data types: "random", "sorted", "reversed", "nearly_sorted".
- Data storage type is only used as a label (algorithms use Python lists).
"""

import sys
import subprocess
import time
import tracemalloc
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any


def ensure_package(pkg: str) -> None:
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])


for p in ("pandas", "plotly", "psutil"):
    ensure_package(p)

import psutil  # type: ignore
import pandas as pd  # type: ignore
import plotly.express as px  # type: ignore

from Backend.algorithms.Sorting import SORTING_ALGOS


BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "configs.jsonl"


@dataclass
class BenchmarkConfig:
    algorithm_type: str
    algorithms: List[str]
    storage_types: List[str]
    data_types: List[str]
    value_type: str
    select_all: Dict[str, bool]


def load_latest_config(path: Path) -> BenchmarkConfig:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    last_line = ""
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line

    if not last_line:
        raise ValueError("Config file is empty or has only empty lines.")

    import json

    obj = json.loads(last_line)
    cfg = obj.get("config", {})

    return BenchmarkConfig(
        algorithm_type=cfg.get("algorithmType") or "sorting",
        algorithms=list(cfg.get("algorithms") or []),
        storage_types=list(cfg.get("storageTypes") or []),
        data_types=list(cfg.get("dataTypes") or []),
        value_type=cfg.get("valueType") or "int",
        select_all=dict(cfg.get("selectAll") or {}),
    )


def generate_values(n: int, value_type: str) -> List[Any]:
    """Generate base values of the requested type."""
    if value_type == "float":
        return [random.random() * n * 10 for _ in range(n)]
    if value_type == "char":
        return [chr(random.randint(97, 122)) for _ in range(n)]  # a-z
    if value_type == "string":
        def rand_str() -> str:
            length = random.randint(3, 8)
            return "".join(chr(random.randint(97, 122)) for _ in range(length))
        return [rand_str() for _ in range(n)]
    # default: int
    return [random.randint(0, n * 10) for _ in range(n)]


def generate_data(n: int, data_type: str, value_type: str) -> List[Any]:
    base = generate_values(n, value_type)
    if data_type == "sorted":
        return sorted(base)
    if data_type == "reversed":
        return sorted(base, reverse=True)
    if data_type == "nearly_sorted":
        base.sort()
        swaps = max(1, n // 20)
        for _ in range(swaps):
            i = random.randrange(n)
            j = random.randrange(n)
            base[i], base[j] = base[j], base[i]
        return base
    return base  # random


def measure(data: List[Any], func) -> Dict[str, float]:
    """
    Measure execution time and memory usage for a given sort function.
    Returns dict with time_seconds, memory_rss_kb, tracemalloc_peak_kb.
    """
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    t0 = time.perf_counter()
    func(list(data))  # work on a copy
    t1 = time.perf_counter()
    current, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss

    elapsed = t1 - t0
    rss_peak_kb = max(rss_before, rss_after) / 1024.0
    tracemalloc_peak_kb = tracemalloc_peak / 1024.0
    return {
        "time_seconds": elapsed,
        "memory_rss_kb": rss_peak_kb,
        "tracemalloc_peak_kb": tracemalloc_peak_kb,
    }


def build_sizes() -> List[int]:
    """
    Build a sequence of n values:
    10, 50, 100, 200, 300, 400, then +400 until a safe upper bound.
    """
    sizes = [10, 50, 100, 200, 300, 400]
    n = 600
    MAX_N = 20000  # safe upper bound for O(n^2) on a normal PC
    while n <= MAX_N:
        sizes.append(n)
        n += 400
    return sizes


def run_benchmarks(cfg: BenchmarkConfig) -> pd.DataFrame:
    """Dispatch to the appropriate benchmark suite based on algorithm_type."""
    if cfg.algorithm_type == "sorting":
        algos = cfg.algorithms or list(SORTING_ALGOS.keys())
        algos = [a for a in algos if a in SORTING_ALGOS]
        if not algos:
            raise ValueError("No supported sorting algorithms selected.")

        data_types = cfg.data_types or ["random"]
        storage_types = cfg.storage_types or ["array"]

        sizes = build_sizes()
        rows: List[Dict[str, Any]] = []

        print(f"[sorting] Using algorithms: {algos}")
        print(f"[sorting] Using data types: {data_types}")
        print(f"[sorting] Using storage types (labels only): {storage_types}")
        print(f"[sorting] Value type: {cfg.value_type}")

        MAX_TIME_PER_RUN = 2.0  # seconds per run

        for n in sizes:
            print(f"\n=== n = {n} ===")
            for data_type in data_types:
                data = generate_data(n, data_type, cfg.value_type)
                for storage in storage_types:
                    for algo_name in algos:
                        sort_func = SORTING_ALGOS[algo_name]
                        label = (
                            f"sorting:{algo_name} | {storage} | {data_type} | {cfg.value_type}"
                        )
                        print(f"  Measuring {label} ...", end="", flush=True)
                        try:
                            metrics = measure(data, sort_func)
                        except MemoryError:
                            print(" MemoryError, stopping.")
                            return pd.DataFrame(rows) if rows else pd.DataFrame()

                        print(
                            f" time={metrics['time_seconds']:.6f}s "
                            f"mem={metrics['memory_rss_kb']:.2f}KB"
                        )

                        rows.append(
                            {
                                "n": n,
                                "algorithm_type": "sorting",
                                "algorithm": algo_name,
                                "storage_type": storage,
                                "data_type": data_type,
                                "value_type": cfg.value_type,
                                "label": label,
                                **metrics,
                            }
                        )

                        if metrics["time_seconds"] > MAX_TIME_PER_RUN:
                            print(
                                f"    Run exceeded {MAX_TIME_PER_RUN}s. "
                                "Stopping further sizes to keep runtime reasonable."
                            )
                            return pd.DataFrame(rows)

        return pd.DataFrame(rows)

    # For now, we only support sorting in the centralized runner.
    # Other algorithmType values (searching, graph) can be added later.
    raise ValueError(
        f"Unsupported algorithmType {cfg.algorithm_type!r} in run_benchmarks. "
        "Currently only 'sorting' is fully wired here."
    )


def make_plots(df: pd.DataFrame) -> None:
    if df.empty:
        print("No data collected, skipping plots.")
        return

    out_dir = BASE_DIR

    time_fig = px.line(
        df,
        x="n",
        y="time_seconds",
        color="label",
        markers=True,
        title="Time vs Input Size (n) – Interactive",
    )
    time_html = out_dir / "time_vs_input_from_config.html"
    time_fig.write_html(time_html)

    mem_fig = px.line(
        df,
        x="n",
        y="memory_rss_kb",
        color="label",
        markers=True,
        title="Memory (RSS KB) vs Input Size (n) – Interactive",
    )
    mem_html = out_dir / "memory_vs_input_from_config.html"
    mem_fig.write_html(mem_html)

    print(f"\nInteractive time plot written to: {time_html.resolve()}")
    print(f"Interactive memory plot written to: {mem_html.resolve()}")
    print("Tip: In Plotly, click legend entries to hide/show individual lines.")


def main() -> None:
    print(f"Loading latest configuration from: {CONFIG_PATH}")
    cfg = load_latest_config(CONFIG_PATH)
    print("Configuration loaded:")
    print(cfg)

    df = run_benchmarks(cfg)
    if df.empty:
        print("No results to plot.")
        return

    csv_path = BASE_DIR / "results_from_config.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nBenchmark results saved to: {csv_path.resolve()}")

    make_plots(df)


if __name__ == "__main__":
    main()