"""
AVX2 vs Naive search benchmark for VeloxDB.

Measures search latency and build-index time across varying dataset sizes,
vector dimensions, and nprobe values. Plots speedup and latency comparisons.

Usage:
    # Small-scale (default, ~1-2 min)
    python tests/benchmark_avx2.py

    # Large-scale: 1M × 768-dim embeddings (~5-15 min, needs ≥12 GB free RAM)
    python tests/benchmark_avx2.py --large

    # Run both
    python tests/benchmark_avx2.py --all
"""

import sys
import os
import time
import tempfile
import struct
import statistics
import argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ── locate the compiled extension ────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUILD_DIR = os.path.join(REPO_ROOT, "build")
sys.path.insert(0, BUILD_DIR)

try:
    import veloxdb
except ImportError as exc:
    print(f"[ERROR] Could not import veloxdb from {BUILD_DIR}: {exc}")
    print("  Build the project first:  cmake -B build && cmake --build build")
    sys.exit(1)

# ── small-scale parameters ────────────────────────────────────────────────────
DATASET_SIZES   = [1_000, 5_000, 10_000, 25_000, 50_000]
DIMENSIONS      = [32, 64, 128, 256, 512]
NPROBE_VALUES   = [1, 2, 4, 8, 16]

FIXED_N         = 10_000
FIXED_DIM       = 128
FIXED_NPROBE    = 4
NUM_CLUSTERS    = 64
KMEANS_EPOCHS   = 5
K               = 10
QUERY_REPS      = 30
METRIC          = "eucl"
SEED            = 42

# ── large-scale parameters ────────────────────────────────────────────────────
LARGE_N              = 1_000_000
LARGE_DIM            = 768
LARGE_NUM_CLUSTERS   = 256    # ~3900 vectors/cluster; sqrt(1M) ≈ 1000 for ideal recall
LARGE_KMEANS_EPOCHS  = 3      # lower to keep build under ~10 min
LARGE_NPROBE_VALUES  = [1, 4, 8, 16, 32]
LARGE_QUERY_REPS     = 20
# RAM requirement: ~3 GB mmap + ~3 GB K-means cache during build → need ≥ 8 GB free
LARGE_MIN_FREE_GB    = 8.0

RESULTS_DIR = os.path.join(REPO_ROOT, "tests", "benchmark_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

rng = np.random.default_rng(SEED)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers — shared
# ─────────────────────────────────────────────────────────────────────────────

def make_vectors(n: int, dim: int) -> np.ndarray:
    """Unit-normalised random float32 vectors (realistic for embeddings)."""
    v = rng.standard_normal((n, dim)).astype(np.float32)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.where(norms == 0, 1, norms)


def time_search(db: veloxdb.VectorIndex, queries: np.ndarray,
                nprobe: int = FIXED_NPROBE) -> list[float]:
    """Return per-query wall-clock latency in seconds."""
    latencies = []
    for q in queries:
        t0 = time.perf_counter()
        db.search(q.tolist(), K, nprobe, METRIC)
        latencies.append(time.perf_counter() - t0)
    return latencies


def median_ms(latencies: list[float]) -> float:
    return statistics.median(latencies) * 1_000


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — small-scale (Python-loop ingestion, fine up to ~100 K vectors)
# ─────────────────────────────────────────────────────────────────────────────

def build_db(vectors: np.ndarray, use_simd: bool,
             num_clusters: int = NUM_CLUSTERS,
             epochs: int = KMEANS_EPOCHS) -> veloxdb.VectorIndex:
    db = veloxdb.VectorIndex()
    db.set_simd(use_simd)
    for row in vectors:
        db.add_vector(row.tolist())
    db.build_index(num_clusters, epochs, METRIC)
    return db


def build_and_time(vectors: np.ndarray, use_simd: bool,
                   num_clusters: int, epochs: int) -> float:
    db = veloxdb.VectorIndex()
    db.set_simd(use_simd)
    for row in vectors:
        db.add_vector(row.tolist())
    t0 = time.perf_counter()
    db.build_index(num_clusters, epochs, METRIC)
    return time.perf_counter() - t0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers — large-scale (fvecs file + mmap loading)
#
# The Python-loop approach calls add_vector 1M times across the Python/C++
# boundary (~several minutes of overhead). Using load_fvecs + mmap bypasses
# this: write the file once with numpy, then each VectorIndex mmaps it in O(1).
#
# During build_index the C++ pre-caches the full mmap into a flat buffer
# (~3 GB for 1M×768). Both indices share the same file pages (MAP_PRIVATE,
# never written), so actual physical RAM for the mmap is ~3 GB regardless of
# how many instances load the file.  Peak pressure is during build: 3 GB mmap
# + 3 GB build cache → need ≥ 8 GB free.
#
# To avoid two simultaneous 3 GB build caches, we run each build sequentially:
#   build AVX2 → time AVX2 search → delete db → build naive → time naive search
# ─────────────────────────────────────────────────────────────────────────────

def write_fvecs(path: str, vectors: np.ndarray) -> None:
    """Write float32 vectors to .fvecs format (int32 dim + float32 values per row)."""
    n, dim = vectors.shape
    # Build a structured array matching the on-disk layout so we can write it
    # in one shot with numpy, which is 100-1000× faster than a Python loop.
    row_dtype = np.dtype([("dim", np.int32), ("vals", np.float32, dim)])
    buf = np.empty(n, dtype=row_dtype)
    buf["dim"]  = dim
    buf["vals"] = vectors.astype(np.float32)
    buf.tofile(path)


def load_fvecs_mmap(path: str, use_simd: bool) -> veloxdb.VectorIndex:
    """Create a VectorIndex backed by an mmap of `path`."""
    db = veloxdb.VectorIndex()
    db.set_simd(use_simd)
    db.load_fvecs(path)
    return db


def available_ram_gb() -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1_048_576
    except OSError:
        pass
    return float("inf")


# ─────────────────────────────────────────────────────────────────────────────
# Small-scale experiments (unchanged from original)
# ─────────────────────────────────────────────────────────────────────────────

def exp_latency_vs_N():
    print("\n[1/4] Latency vs dataset size …")
    avx_lat, naive_lat = [], []
    queries = make_vectors(QUERY_REPS, FIXED_DIM)

    for N in DATASET_SIZES:
        print(f"  N={N:,}")
        vecs = make_vectors(N, FIXED_DIM)
        db_avx   = build_db(vecs, use_simd=True)
        db_naive = build_db(vecs, use_simd=False)
        avx_lat.append(median_ms(time_search(db_avx,   queries)))
        naive_lat.append(median_ms(time_search(db_naive, queries)))

    return avx_lat, naive_lat


def exp_latency_vs_dim():
    print("\n[2/4] Latency vs vector dimension …")
    avx_lat, naive_lat = [], []

    for dim in DIMENSIONS:
        print(f"  dim={dim}")
        vecs    = make_vectors(FIXED_N, dim)
        queries = make_vectors(QUERY_REPS, dim)
        db_avx   = build_db(vecs, use_simd=True)
        db_naive = build_db(vecs, use_simd=False)
        avx_lat.append(median_ms(time_search(db_avx,   queries)))
        naive_lat.append(median_ms(time_search(db_naive, queries)))

    return avx_lat, naive_lat


def exp_latency_vs_nprobe():
    print("\n[3/4] Latency vs nprobe …")
    avx_lat, naive_lat = [], []
    vecs    = make_vectors(FIXED_N, FIXED_DIM)
    queries = make_vectors(QUERY_REPS, FIXED_DIM)
    db_avx   = build_db(vecs, use_simd=True)
    db_naive = build_db(vecs, use_simd=False)

    for np_val in NPROBE_VALUES:
        avx_lat.append(median_ms(time_search(db_avx,   queries, nprobe=np_val)))
        naive_lat.append(median_ms(time_search(db_naive, queries, nprobe=np_val)))

    return avx_lat, naive_lat


def exp_build_time_vs_N():
    print("\n[4/4] Index build time vs dataset size …")
    avx_build, naive_build = [], []

    for N in DATASET_SIZES:
        print(f"  N={N:,}")
        vecs = make_vectors(N, FIXED_DIM)
        avx_build.append(build_and_time(vecs, True,  NUM_CLUSTERS, KMEANS_EPOCHS))
        naive_build.append(build_and_time(vecs, False, NUM_CLUSTERS, KMEANS_EPOCHS))

    return avx_build, naive_build


# ─────────────────────────────────────────────────────────────────────────────
# Large-scale experiment — 1 M × 768
# ─────────────────────────────────────────────────────────────────────────────

def exp_large_scale() -> dict:
    """
    Run three sub-experiments at 1M × 768:
      A. Index build time: AVX2 vs scalar
      B. Search latency vs nprobe: AVX2 vs scalar
      C. Throughput (queries/sec) at fixed nprobe=8
    Builds are sequential to avoid two simultaneous 3 GB build caches.
    Returns a dict of result arrays.
    """
    free_gb = available_ram_gb()
    print(f"\n  Available RAM: {free_gb:.1f} GB  (need ≥ {LARGE_MIN_FREE_GB} GB)")
    if free_gb < LARGE_MIN_FREE_GB:
        print(
            f"  [WARNING] Only {free_gb:.1f} GB free. "
            f"1M×768 needs ≥ {LARGE_MIN_FREE_GB} GB to avoid OOM/swapping.\n"
            "  Proceeding anyway — expect heavy swap pressure or an OOM kill.\n"
            "  Consider running on a machine with more RAM, or reduce LARGE_N."
        )

    queries = make_vectors(LARGE_QUERY_REPS, LARGE_DIM)

    # ── A. Generate dataset and write to a temp fvecs file ───────────────────
    print(f"\n  Generating {LARGE_N:,} × {LARGE_DIM}-dim vectors …")
    t0 = time.perf_counter()
    # Generate in chunks to avoid allocating 3 GB at once in Python
    chunk = 100_000
    with tempfile.NamedTemporaryFile(suffix=".fvecs", delete=False) as f:
        fvecs_path = f.name
    with open(fvecs_path, "wb") as f:
        row_dtype = np.dtype([("dim", np.int32), ("vals", np.float32, LARGE_DIM)])
        for start in range(0, LARGE_N, chunk):
            end  = min(start + chunk, LARGE_N)
            part = make_vectors(end - start, LARGE_DIM)
            buf  = np.empty(end - start, dtype=row_dtype)
            buf["dim"]  = LARGE_DIM
            buf["vals"] = part
            f.write(buf.tobytes())
    gen_time = time.perf_counter() - t0
    file_gb = os.path.getsize(fvecs_path) / 1e9
    print(f"  Done — {file_gb:.2f} GB written in {gen_time:.1f}s  →  {fvecs_path}")

    # ── B. Build: AVX2 ───────────────────────────────────────────────────────
    print(f"\n  Building IVF index (AVX2, clusters={LARGE_NUM_CLUSTERS}, "
          f"epochs={LARGE_KMEANS_EPOCHS}) …")
    db_avx = load_fvecs_mmap(fvecs_path, use_simd=True)
    t0 = time.perf_counter()
    db_avx.build_index(LARGE_NUM_CLUSTERS, LARGE_KMEANS_EPOCHS, METRIC)
    avx_build_s = time.perf_counter() - t0
    print(f"  AVX2 build: {avx_build_s:.1f}s")

    # Search: AVX2
    avx_lat_by_nprobe = []
    for np_val in LARGE_NPROBE_VALUES:
        avx_lat_by_nprobe.append(median_ms(time_search(db_avx, queries, nprobe=np_val)))
    del db_avx  # release mmap + index before allocating another build cache

    # ── C. Build: scalar ─────────────────────────────────────────────────────
    print(f"\n  Building IVF index (scalar, clusters={LARGE_NUM_CLUSTERS}, "
          f"epochs={LARGE_KMEANS_EPOCHS}) …")
    db_naive = load_fvecs_mmap(fvecs_path, use_simd=False)
    t0 = time.perf_counter()
    db_naive.build_index(LARGE_NUM_CLUSTERS, LARGE_KMEANS_EPOCHS, METRIC)
    naive_build_s = time.perf_counter() - t0
    print(f"  Scalar build: {naive_build_s:.1f}s")

    # Search: scalar
    naive_lat_by_nprobe = []
    for np_val in LARGE_NPROBE_VALUES:
        naive_lat_by_nprobe.append(median_ms(time_search(db_naive, queries, nprobe=np_val)))
    del db_naive

    try:
        os.unlink(fvecs_path)
    except OSError:
        pass

    return {
        "avx_build_s":          avx_build_s,
        "naive_build_s":        naive_build_s,
        "avx_lat_by_nprobe":    avx_lat_by_nprobe,
        "naive_lat_by_nprobe":  naive_lat_by_nprobe,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — small-scale
# ─────────────────────────────────────────────────────────────────────────────

sns.set_theme(style="whitegrid", palette="deep", font_scale=1.1)
COLORS = {"avx": "#2196F3", "naive": "#F44336", "speedup": "#4CAF50"}


def _save(fig, name):
    path = os.path.join(RESULTS_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"  Saved → {path}")
    plt.close(fig)


def _speedup(avx, naive):
    return [n / a for a, n in zip(avx, naive)]


def _bar(ax, xs, x_labels, su, xlabel, title):
    colors = [COLORS["speedup"] if s >= 1 else COLORS["naive"] for s in su]
    bars = ax.bar(xs, su, color=colors, width=0.6, edgecolor="white")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1× (no gain)")
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, fontsize=9)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Speedup  (scalar / AVX2)")
    ax.set_title(title)
    ax.legend(fontsize=8)
    for bar, s in zip(bars, su):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{s:.2f}×", ha="center", va="bottom", fontsize=8)


def plot_small_scale(avx_N, naive_N,
                     avx_dim, naive_dim,
                     avx_np, naive_np,
                     avx_build, naive_build):

    # Figure 1 — 2×2 latency comparisons
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("VeloxDB — AVX2 SIMD vs Scalar Search Performance", fontsize=14, y=1.01)

    ax = axes[0, 0]
    ax.plot(DATASET_SIZES, avx_N,   marker="o", color=COLORS["avx"],  label="AVX2 (SIMD)")
    ax.plot(DATASET_SIZES, naive_N, marker="s", color=COLORS["naive"], label="Scalar (naive)")
    ax.set_xlabel("Dataset size  (N)")
    ax.set_ylabel("Median query latency (ms)")
    ax.set_title(f"Latency vs Dataset Size  (dim={FIXED_DIM})")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()

    ax = axes[0, 1]
    ax.plot(DIMENSIONS, avx_dim,   marker="o", color=COLORS["avx"],  label="AVX2 (SIMD)")
    ax.plot(DIMENSIONS, naive_dim, marker="s", color=COLORS["naive"], label="Scalar (naive)")
    ax.set_xlabel("Vector dimension (d)")
    ax.set_ylabel("Median query latency (ms)")
    ax.set_title(f"Latency vs Dimensionality  (N={FIXED_N:,})")
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: str(int(x))))
    ax.legend()

    ax = axes[1, 0]
    ax.plot(NPROBE_VALUES, avx_np,   marker="o", color=COLORS["avx"],  label="AVX2 (SIMD)")
    ax.plot(NPROBE_VALUES, naive_np, marker="s", color=COLORS["naive"], label="Scalar (naive)")
    ax.set_xlabel("nprobe")
    ax.set_ylabel("Median query latency (ms)")
    ax.set_title(f"Latency vs nprobe  (N={FIXED_N:,}, dim={FIXED_DIM})")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(DATASET_SIZES, avx_build,   marker="o", color=COLORS["avx"],  label="AVX2 (SIMD)")
    ax.plot(DATASET_SIZES, naive_build, marker="s", color=COLORS["naive"], label="Scalar (naive)")
    ax.set_xlabel("Dataset size  (N)")
    ax.set_ylabel("Build time (s)")
    ax.set_title(f"IVF Build Time vs Dataset Size  (dim={FIXED_DIM})")
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.legend()

    fig.tight_layout()
    _save(fig, "latency_comparison.png")

    # Figure 2 — speedup bars
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    fig.suptitle("VeloxDB — AVX2 Speedup over Scalar", fontsize=14, y=1.02)

    _bar(axes[0],
         range(len(DATASET_SIZES)),
         [f"{n//1000}K" for n in DATASET_SIZES],
         _speedup(avx_N, naive_N),
         "Dataset size", f"Search speedup vs N  (dim={FIXED_DIM})")

    _bar(axes[1],
         range(len(DIMENSIONS)),
         [str(d) for d in DIMENSIONS],
         _speedup(avx_dim, naive_dim),
         "Vector dimension", f"Search speedup vs dim  (N={FIXED_N:,})")

    _bar(axes[2],
         range(len(NPROBE_VALUES)),
         [str(p) for p in NPROBE_VALUES],
         _speedup(avx_np, naive_np),
         "nprobe", f"Search speedup vs nprobe  (N={FIXED_N:,}, dim={FIXED_DIM})")

    fig.tight_layout()
    _save(fig, "speedup_bars.png")

    # Figure 3 — N × dim speedup heatmap
    print("\n[+] Building speedup heatmap (N × dim) …")
    N_vals   = [1_000, 5_000, 10_000, 25_000, 50_000]
    dim_vals = [32, 64, 128, 256, 512]
    matrix   = np.zeros((len(N_vals), len(dim_vals)), dtype=float)

    for i, N in enumerate(N_vals):
        for j, dim in enumerate(dim_vals):
            print(f"    N={N:,}  dim={dim}")
            vecs    = make_vectors(N, dim)
            queries = make_vectors(QUERY_REPS, dim)
            db_a    = build_db(vecs, True)
            db_n    = build_db(vecs, False)
            a_ms    = median_ms(time_search(db_a, queries))
            n_ms    = median_ms(time_search(db_n, queries))
            matrix[i, j] = n_ms / a_ms if a_ms > 0 else 1.0

    fig, ax = plt.subplots(figsize=(9, 6))
    sns.heatmap(
        matrix, ax=ax, annot=True, fmt=".2f",
        xticklabels=dim_vals,
        yticklabels=[f"{n//1000}K" for n in N_vals],
        cmap="YlGn", linewidths=0.5,
        cbar_kws={"label": "Speedup (scalar / AVX2)"},
    )
    ax.set_xlabel("Vector dimension (d)")
    ax.set_ylabel("Dataset size (N)")
    ax.set_title("AVX2 Search Speedup Heatmap  (nprobe=4, k=10)")
    fig.tight_layout()
    _save(fig, "speedup_heatmap.png")

    # Figure 4 — build-time speedup
    fig, ax = plt.subplots(figsize=(7, 4))
    _bar(ax,
         range(len(DATASET_SIZES)),
         [f"{n//1000}K" for n in DATASET_SIZES],
         _speedup(avx_build, naive_build),
         "Dataset size", f"K-Means Build Speedup  (dim={FIXED_DIM}, clusters={NUM_CLUSTERS})")
    fig.tight_layout()
    _save(fig, "build_speedup.png")

    # Console summary
    print("\n" + "="*60)
    print("SUMMARY  (median query latency, ms)")
    print("="*60)
    print(f"{'N':>8}  {'AVX2':>10}  {'Scalar':>10}  {'Speedup':>10}")
    print("-"*60)
    for N, a, n in zip(DATASET_SIZES, avx_N, naive_N):
        print(f"{N:>8,}  {a:>10.3f}  {n:>10.3f}  {n/a:>9.2f}×")
    print()
    print(f"{'dim':>8}  {'AVX2':>10}  {'Scalar':>10}  {'Speedup':>10}")
    print("-"*60)
    for d, a, n in zip(DIMENSIONS, avx_dim, naive_dim):
        print(f"{d:>8}  {a:>10.3f}  {n:>10.3f}  {n/a:>9.2f}×")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting — large-scale
# ─────────────────────────────────────────────────────────────────────────────

def plot_large_scale(results: dict):
    avx_build_s         = results["avx_build_s"]
    naive_build_s       = results["naive_build_s"]
    avx_lat_by_nprobe   = results["avx_lat_by_nprobe"]
    naive_lat_by_nprobe = results["naive_lat_by_nprobe"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        f"VeloxDB Large-Scale: 1M × {LARGE_DIM}-dim  "
        f"(clusters={LARGE_NUM_CLUSTERS}, epochs={LARGE_KMEANS_EPOCHS})",
        fontsize=13, y=1.02,
    )

    # Panel A — build time comparison
    ax = axes[0]
    bars = ax.bar(
        ["AVX2 (SIMD)", "Scalar (naive)"],
        [avx_build_s, naive_build_s],
        color=[COLORS["avx"], COLORS["naive"]],
        width=0.5, edgecolor="white",
    )
    for bar, val in zip(bars, [avx_build_s, naive_build_s]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}s", ha="center", va="bottom", fontsize=10)
    speedup_build = naive_build_s / avx_build_s
    ax.set_title(f"IVF Build Time\n(speedup: {speedup_build:.2f}×)")
    ax.set_ylabel("Build time (s)")
    ax.set_ylim(0, max(avx_build_s, naive_build_s) * 1.25)

    # Panel B — search latency vs nprobe
    ax = axes[1]
    ax.plot(LARGE_NPROBE_VALUES, avx_lat_by_nprobe,   marker="o",
            color=COLORS["avx"],  label="AVX2 (SIMD)")
    ax.plot(LARGE_NPROBE_VALUES, naive_lat_by_nprobe, marker="s",
            color=COLORS["naive"], label="Scalar (naive)")
    ax.set_xlabel("nprobe")
    ax.set_ylabel("Median query latency (ms)")
    ax.set_title("Search Latency vs nprobe")
    ax.legend()

    # Panel C — speedup across nprobe values
    ax = axes[2]
    su = _speedup(avx_lat_by_nprobe, naive_lat_by_nprobe)
    colors = [COLORS["speedup"] if s >= 1 else COLORS["naive"] for s in su]
    bars = ax.bar(range(len(LARGE_NPROBE_VALUES)), su, color=colors,
                  width=0.6, edgecolor="white")
    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="1× (no gain)")
    ax.set_xticks(range(len(LARGE_NPROBE_VALUES)))
    ax.set_xticklabels([str(p) for p in LARGE_NPROBE_VALUES])
    ax.set_xlabel("nprobe")
    ax.set_ylabel("Speedup  (scalar / AVX2)")
    ax.set_title("Search Speedup vs nprobe")
    ax.legend(fontsize=8)
    for bar, s in zip(bars, su):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{s:.2f}×", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    _save(fig, "large_scale_benchmark.png")

    # Console summary
    print("\n" + "="*60)
    print(f"LARGE-SCALE SUMMARY  (N={LARGE_N:,}, dim={LARGE_DIM})")
    print("="*60)
    print(f"  Build (AVX2):   {avx_build_s:.1f}s")
    print(f"  Build (scalar): {naive_build_s:.1f}s")
    print(f"  Build speedup:  {speedup_build:.2f}×")
    print()
    print(f"{'nprobe':>8}  {'AVX2 (ms)':>12}  {'Scalar (ms)':>12}  {'Speedup':>10}")
    print("-"*55)
    for np_val, a, n in zip(LARGE_NPROBE_VALUES, avx_lat_by_nprobe, naive_lat_by_nprobe):
        print(f"{np_val:>8}  {a:>12.3f}  {n:>12.3f}  {n/a:>9.2f}×")
    print("="*60)


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VeloxDB AVX2 benchmark")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--large", action="store_true",
                      help="Run only the large-scale 1M×768 experiment")
    mode.add_argument("--all",   action="store_true",
                      help="Run small-scale experiments then large-scale")
    args = parser.parse_args()

    run_small = not args.large        # default and --all both include small
    run_large = args.large or args.all

    if run_small:
        print("VeloxDB AVX2 Benchmark — small scale")
        print(f"  Fixed N={FIXED_N:,}  dim={FIXED_DIM}  clusters={NUM_CLUSTERS}")
        print(f"  k={K}  nprobe={FIXED_NPROBE}  query_reps={QUERY_REPS}")
        print(f"  Results → {RESULTS_DIR}\n")

        avx_N,     naive_N     = exp_latency_vs_N()
        avx_dim,   naive_dim   = exp_latency_vs_dim()
        avx_np,    naive_np    = exp_latency_vs_nprobe()
        avx_build, naive_build = exp_build_time_vs_N()

        print("\nPlotting small-scale results …")
        plot_small_scale(avx_N, naive_N,
                         avx_dim, naive_dim,
                         avx_np, naive_np,
                         avx_build, naive_build)

    if run_large:
        print("\nVeloxDB AVX2 Benchmark — large scale")
        print(f"  N={LARGE_N:,}  dim={LARGE_DIM}  "
              f"clusters={LARGE_NUM_CLUSTERS}  epochs={LARGE_KMEANS_EPOCHS}")
        print(f"  RAM needed: ≥ {LARGE_MIN_FREE_GB} GB free  "
              f"(available: {available_ram_gb():.1f} GB)")
        print(f"  Results → {RESULTS_DIR}\n")

        large_results = exp_large_scale()

        print("\nPlotting large-scale results …")
        plot_large_scale(large_results)

    print("\nDone. Plots saved to", RESULTS_DIR)
