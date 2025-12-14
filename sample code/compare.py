"""
Compare Bubble Sort vs Insertion Sort benchmarking (with pandas, psutil, plotly).

Changes:
- Uses psutil to measure process memory (RSS).
- Stores results in a pandas.DataFrame and saves CSV.
- Uses Plotly for interactive Time/Memory vs Input Size plots (HTML).
- Also saves static matplotlib PNG for convenience.
- Installs missing packages automatically (Windows).
"""
import sys
import subprocess
import time
import tracemalloc
import random
import argparse
from pathlib import Path

# Ensure required packages are installed
def ensure_package(pkg):
    try:
        __import__(pkg)
    except Exception:
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg])

for p in ("pandas", "plotly", "psutil", "matplotlib"):
    ensure_package(p)

import psutil
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt

def bubble_sort(arr):
    n = len(arr)
    for i in range(n):
        swapped = False
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
                swapped = True
        if not swapped:
            break
    return arr

def insertion_sort(arr):
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def measure_sort_on_data(data, sort_func):
    """
    Measures execution time and memory usage (psutil RSS and tracemalloc peak).
    Returns: (elapsed_seconds, rss_peak_kb, tracemalloc_peak_kb)
    """
    proc = psutil.Process()
    rss_before = proc.memory_info().rss
    tracemalloc.start()
    t0 = time.perf_counter()
    sort_func(data)
    t1 = time.perf_counter()
    current, tracemalloc_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_after = proc.memory_info().rss

    elapsed = t1 - t0
    rss_peak = max(rss_before, rss_after)  # conservative estimate (bytes)
    return elapsed, rss_peak / 1024.0, tracemalloc_peak / 1024.0

def parse_sizes(arg_list):
    if len(arg_list) == 1 and "," in arg_list[0]:
        return [int(x) for x in arg_list[0].split(",") if x.strip()]
    return [int(x) for x in arg_list]

def main():
    parser = argparse.ArgumentParser(description="Compare Bubble vs Insertion sort: time & memory (pandas/plotly/psutil).")
    parser.add_argument(
        "--sizes",
        nargs="+",
        default=["100", "500", "1000", "2000"],
        help="Input sizes as space-separated or single comma-separated list, e.g. --sizes 100 500 or --sizes 100,500,1000",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (optional) for reproducible inputs.",
    )
    parser.add_argument(
        "--out",
        default="results_compare.csv",
        help="Output CSV file to store results (default: results_compare.csv).",
    )
    args = parser.parse_args()
    try:
        sizes = parse_sizes(args.sizes)
    except ValueError:
        print("Error: sizes must be integers. Example: --sizes 100 500 1000 or --sizes 100,500,1000")
        sys.exit(1)

    out_path = Path(args.out)
    algorithms = [("bubble", bubble_sort), ("insertion", insertion_sort)]

    rows = []  # list of dicts for pandas DataFrame

    print(f"Running comparison for sizes: {sizes}")
    for size in sizes:
        print(f"  Preparing input of size {size} ...", end="", flush=True)
        if args.seed is not None:
            random.seed(args.seed + size)
        base_data = [random.randint(0, size * 10) for _ in range(size)]
        print(" done")

        for name, func in algorithms:
            print(f"    Measuring {name} sort ...", end="", flush=True)
            data_copy = list(base_data)
            try:
                elapsed, rss_kb, tracemalloc_kb = measure_sort_on_data(data_copy, func)
            except MemoryError:
                print(" failed (MemoryError)")
                elapsed, rss_kb, tracemalloc_kb = float("nan"), float("nan"), float("nan")

            rows.append({
                "size": size,
                "algorithm": name,
                "time_seconds": elapsed,
                "memory_rss_kb": rss_kb,
                "tracemalloc_peak_kb": tracemalloc_kb
            })
            print(f" done: time={elapsed:.6f}s rss={rss_kb:.2f}KB tracemalloc_peak={tracemalloc_kb:.2f}KB")

    # Create DataFrame and save
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Results written to {out_path.resolve()}")

    # Prepare aggregated/processed data if needed (example: mean by algorithm+size)
    summary = df.groupby(["algorithm", "size"], as_index=False).agg(
        time_mean=("time_seconds", "mean"),
        time_std=("time_seconds", "std"),
        memory_rss_mean=("memory_rss_kb", "mean"),
        tracemalloc_peak_mean=("tracemalloc_peak_kb", "mean"),
    )

    # Plot interactive Time vs Input Size with Plotly
    time_fig = px.line(df, x="size", y="time_seconds", color="algorithm", markers=True,
                       title="Time vs Input Size (interactive)")
    time_html = Path("time_vs_input_interactive.html")
    time_fig.write_html(time_html)
    print(f"Interactive time plot saved to {time_html.resolve()}")

    # Plot interactive Memory (RSS) vs Input Size with Plotly
    mem_fig = px.line(df, x="size", y="memory_rss_kb", color="algorithm", markers=True,
                      title="Memory (RSS KB) vs Input Size (interactive)")
    mem_html = Path("memory_vs_input_interactive.html")
    mem_fig.write_html(mem_html)
    print(f"Interactive memory plot saved to {mem_html.resolve()}")

    # Also save static matplotlib plots (combined)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    for name in df["algorithm"].unique():
        sub = df[df["algorithm"] == name]
        plt.plot(sub["size"], sub["time_seconds"], marker="o", label=name.capitalize())
    plt.title("Time vs Input Size")
    plt.xlabel("Input size (n)")
    plt.ylabel("Time (seconds)")
    plt.grid(True)
    plt.legend()

    plt.subplot(1, 2, 2)
    for name in df["algorithm"].unique():
        sub = df[df["algorithm"] == name]
        plt.plot(sub["size"], sub["memory_rss_kb"], marker="o", label=name.capitalize())
    plt.title("Memory (RSS KB) vs Input Size")
    plt.xlabel("Input size (n)")
    plt.ylabel("Memory (KB)")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    png_path = Path("compare_plots.png")
    plt.savefig(png_path)
    print(f"Static plots saved to {png_path.resolve()}")

if __name__ == "__main__":
    main()