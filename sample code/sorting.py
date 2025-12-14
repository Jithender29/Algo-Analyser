"""
Bubble sort benchmarking script.

- Measures time and peak Python memory (using tracemalloc) for different input sizes.
- Writes results to 'results.txt' in the same folder.
- Plots Time vs Input Size and Memory vs Input Size (saves PNG files and shows them).

Usage (Windows):
    py sorting.py
    py sorting.py --sizes 100 500 1000
    py sorting.py --sizes 100,500,1000
"""
import sys
import subprocess
import time
import tracemalloc
import random
import argparse
from pathlib import Path

# ensure matplotlib is available; install if missing
try:
    import matplotlib.pyplot as plt
except Exception:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
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


def measure_for_size(size, seed=None):
    if seed is not None:
        random.seed(seed)
    data = [random.randint(0, size * 10) for _ in range(size)]

    tracemalloc.start()
    t0 = time.perf_counter()
    bubble_sort(data)
    t1 = time.perf_counter()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    elapsed = t1 - t0
    peak_kb = peak / 1024
    return elapsed, peak_kb


def parse_sizes(arg_list):
    if len(arg_list) == 1 and "," in arg_list[0]:
        return [int(x) for x in arg_list[0].split(",") if x.strip()]
    return [int(x) for x in arg_list]


def main():
    parser = argparse.ArgumentParser(description="Bubble sort benchmarking: time & memory.")
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
        default="results.txt",
        help="Output text file to store results (default: results.txt).",
    )
    args = parser.parse_args()
    try:
        sizes = parse_sizes(args.sizes)
    except ValueError:
        print("Error: sizes must be integers. Example: --sizes 100 500 1000 or --sizes 100,500,1000")
        sys.exit(1)

    out_path = Path(args.out)
    results = []

    with out_path.open("w") as f:
        f.write("size,time_seconds,peak_memory_kb\n")

    print(f"Running benchmarks for sizes: {sizes}")
    for size in sizes:
        print(f"  Measuring size {size} ...", end="", flush=True)
        try:
            elapsed, peak_kb = measure_for_size(size, seed=args.seed)
        except MemoryError:
            print(" failed (MemoryError)")
            elapsed, peak_kb = float("nan"), float("nan")
        results.append((size, elapsed, peak_kb))
        with out_path.open("a") as f:
            f.write(f"{size},{elapsed:.6f},{peak_kb:.2f}\n")
        print(f" done: time={elapsed:.6f}s peak={peak_kb:.2f}KB")

    sizes_list = [r[0] for r in results]
    times = [r[1] for r in results]
    mems = [r[2] for r in results]

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(sizes_list, times, marker="o")
    plt.title("Time vs Input Size")
    plt.xlabel("Input size (n)")
    plt.ylabel("Time (seconds)")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(sizes_list, mems, marker="o", color="orange")
    plt.title("Peak Memory vs Input Size")
    plt.xlabel("Input size (n)")
    plt.ylabel("Peak memory (KB)")
    plt.grid(True)

    plt.tight_layout()
    png_path = Path("benchmark_plots.png")
    plt.savefig(png_path)
    print(f"Results written to {out_path.resolve()}")
    print(f"Plots saved to {png_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()

