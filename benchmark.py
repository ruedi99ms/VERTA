"""
VERTA Benchmarking Script
=========================
Measures wall-clock duration and peak memory usage for the core VERTA
algorithms (discover, predict, intent) across different dataset sizes.

Each command is run twice per size:
  1. compute-only  (skip_plots=True)
  2. with plots    (skip_plots=False)

Usage:
    python benchmark.py --input <folder_with_csvs> \\
        --junctions 685 170 30  550 -90 30  730 440 20

    Optional:
        --sizes 25 50 100          Dataset sizes to benchmark (default: 25 50 100)
        --glob "*.csv"             File pattern (default: *.csv)
        --scale 0.2                Coordinate scale factor (default: 0.2)
        --motion_threshold 0.1     Motion threshold (default: 0.1)
        --seed 42                  Random seed (default: 42)
        --out ./benchmark_results  Output directory (default: ./benchmark_results)
        --csv benchmark.csv        Output CSV filename (default: benchmark.csv)
        --commands discover predict intent
"""

import matplotlib
matplotlib.use("Agg")

import argparse
import gc
import glob as glob_mod
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
import tracemalloc
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    workflow: str
    trajectories: int
    total_points: int
    junctions: int
    runtime_sec: float
    peak_ram_mb: float
    status: str          # "ok" | "error" | "skipped"
    include_viz: bool    # True = full run, False = compute-only
    error_msg: str = ""
    extra: Dict = field(default_factory=dict)


def _format_points(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.0f}k"
    return str(n)


def _count_total_points(trajectories) -> int:
    return sum(len(tr.x) for tr in trajectories)


@contextmanager
def measure():
    """Context manager; yields a dict populated with runtime_sec and peak_ram_mb on exit."""
    gc.collect()
    tracemalloc.start()
    t0 = time.perf_counter()
    result = {}
    try:
        yield result
    finally:
        result["runtime_sec"] = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        result["peak_ram_mb"] = peak / (1024 * 1024)
        tracemalloc.stop()
        gc.collect()


# ---------------------------------------------------------------------------
# Load trajectories (shared step — not included in timings)
# ---------------------------------------------------------------------------

def load_trajectories(folder: str, pattern: str, n: int,
                      columns: Optional[Dict[str, str]],
                      scale: float, motion_threshold: float):
    from verta.verta_data_loader import load_folder

    all_paths = sorted(glob_mod.glob(os.path.join(folder, pattern)))
    if len(all_paths) < n:
        raise ValueError(
            f"Requested {n} files but only {len(all_paths)} match '{pattern}' in {folder}"
        )

    tmp = tempfile.mkdtemp(prefix="verta_bench_")
    for p in all_paths[:n]:
        shutil.copy2(p, os.path.join(tmp, os.path.basename(p)))

    trajectories = load_folder(
        folder=tmp, pattern=pattern, columns=columns,
        require_time=False, scale=scale, motion_threshold=motion_threshold,
    )
    shutil.rmtree(tmp, ignore_errors=True)
    return trajectories


# ---------------------------------------------------------------------------
# Shared: discover helper (used by predict and intent as a prerequisite)
# ---------------------------------------------------------------------------

def _discover(trajectories, junctions, out_dir, seed):
    from verta.verta_decisions import discover_branches

    results = []
    for ji, junc in enumerate(junctions):
        junc_out = os.path.join(out_dir, f"_discover_j{ji}")
        os.makedirs(junc_out, exist_ok=True)
        assignments, _summary, centers = discover_branches(
            trajectories=trajectories,
            junction=junc,
            k=3,
            path_length=100.0,
            epsilon=0.05,
            seed=seed,
            decision_mode="hybrid",
            r_outer=50.0,
            linger_delta=0.0,
            out_dir=junc_out,
            cluster_method="dbscan",
            k_min=2, k_max=6,
            min_sep_deg=12.0,
            angle_eps=11.0,
            min_samples=5,
            junction_number=ji,
            all_junctions=junctions,
            skip_plots=True,
        )
        results.append((assignments, centers))
    return results


# ---------------------------------------------------------------------------
# Benchmark: discover
# ---------------------------------------------------------------------------

def bench_discover(trajectories, junctions, out_dir, args,
                   include_viz: bool) -> BenchmarkResult:
    from verta.verta_decisions import discover_branches

    total_pts = _count_total_points(trajectories)

    with measure() as m:
        for ji, junc in enumerate(junctions):
            junc_out = os.path.join(out_dir, f"j{ji}")
            os.makedirs(junc_out, exist_ok=True)
            discover_branches(
                trajectories=trajectories,
                junction=junc,
                k=3,
                path_length=100.0,
                epsilon=0.05,
                seed=args.seed,
                decision_mode="hybrid",
                r_outer=50.0,
                linger_delta=0.0,
                out_dir=junc_out,
                cluster_method="dbscan",
                k_min=2, k_max=6,
                min_sep_deg=12.0,
                angle_eps=11.0,
                min_samples=5,
                junction_number=ji,
                all_junctions=junctions,
                skip_plots=not include_viz,
            )

    return BenchmarkResult(
        workflow="discover",
        trajectories=len(trajectories),
        total_points=total_pts,
        junctions=len(junctions),
        runtime_sec=m["runtime_sec"],
        peak_ram_mb=m["peak_ram_mb"],
        status="ok",
        include_viz=include_viz,
    )


# ---------------------------------------------------------------------------
# Benchmark: predict
# ---------------------------------------------------------------------------

def bench_predict(trajectories, junctions, out_dir, args,
                  include_viz: bool) -> BenchmarkResult:
    from verta.verta_prediction import analyze_junction_choice_patterns

    total_pts = _count_total_points(trajectories)
    n_loaded = len(trajectories)
    r_outer_list = [50.0] * len(junctions)
    chain_df = pd.DataFrame({"trajectory": list(range(n_loaded))})

    with measure() as m:
        analyze_junction_choice_patterns(
            trajectories=trajectories,
            chain_df=chain_df,
            junctions=junctions,
            output_dir=out_dir,
            r_outer_list=r_outer_list,
            gui_mode=False,
            skip_plots=not include_viz,
        )

    return BenchmarkResult(
        workflow="predict",
        trajectories=n_loaded,
        total_points=total_pts,
        junctions=len(junctions),
        runtime_sec=m["runtime_sec"],
        peak_ram_mb=m["peak_ram_mb"],
        status="ok",
        include_viz=include_viz,
    )


# ---------------------------------------------------------------------------
# Benchmark: intent
# ---------------------------------------------------------------------------

def bench_intent(trajectories, junctions, out_dir, args,
                 include_viz: bool) -> BenchmarkResult:
    from verta.verta_intent_recognition import analyze_intent_recognition

    total_pts = _count_total_points(trajectories)

    prereq = _discover(trajectories, junctions, out_dir, args.seed)

    errors = []
    with measure() as m:
        for ji, junc in enumerate(junctions):
            assignments, _centers = prereq[ji]
            valid = assignments[assignments["branch"] >= 0]
            if len(valid) < 10:
                errors.append(f"J{ji}: {len(valid)} valid assignments (<10)")
                continue
            junc_out = os.path.join(out_dir, f"intent_j{ji}")
            os.makedirs(junc_out, exist_ok=True)
            results = analyze_intent_recognition(
                trajectories=trajectories,
                junction=junc,
                actual_branches=valid,
                output_dir=junc_out,
                prediction_distances=[100.0, 75.0, 50.0, 25.0],
                previous_choices=None,
                skip_plots=not include_viz,
            )
            if "error" in results:
                errors.append(f"J{ji}: {results['error']}")

    n_skip = sum("valid assignments" in e for e in errors)
    n_real_err = len(errors) - n_skip
    if n_real_err > 0:
        status = "error"
    elif n_skip == len(junctions):
        status = "skipped"
    else:
        status = "ok"
    return BenchmarkResult(
        workflow="intent",
        trajectories=len(trajectories),
        total_points=total_pts,
        junctions=len(junctions),
        runtime_sec=m["runtime_sec"],
        peak_ram_mb=m["peak_ram_mb"],
        status=status,
        include_viz=include_viz,
        error_msg="; ".join(errors) if errors else "",
    )


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

TABLE_FMT = "{:<12} {:>6} {:>14} {:>14} {:>10} {:>14} {:>16} {:>8}"
TABLE_COLS = ("Workflow", "Viz", "Trajectories", "Total points", "Junctions",
              "Runtime (s)", "Peak RAM (MB)", "Status")


def print_table_header():
    line = TABLE_FMT.format(*TABLE_COLS)
    print(line)
    print("-" * len(line))


def print_table_row(r: BenchmarkResult):
    pts = _format_points(r.total_points) if r.total_points else "-"
    rt = f"{r.runtime_sec:.1f}" if r.status == "ok" else "-"
    ram = f"{r.peak_ram_mb:.1f}" if r.status == "ok" else "-"
    viz = "yes" if r.include_viz else "no"
    print(TABLE_FMT.format(
        r.workflow, viz, str(r.trajectories), pts,
        str(r.junctions), rt, ram, r.status,
    ))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="VERTA Benchmark — measure duration & memory for discover / predict / intent"
    )
    p.add_argument("--input", required=True,
                   help="Folder containing trajectory CSV files")
    p.add_argument("--junctions", nargs="+", type=float, required=True,
                   help="Junction coordinates as x z r triplets (e.g. 685 170 30 550 -90 30)")
    p.add_argument("--sizes", nargs="+", type=int, default=[25, 50, 100],
                   help="Dataset sizes to benchmark (default: 25 50 100)")
    p.add_argument("--glob", default="*.csv", help="File pattern (default: *.csv)")
    p.add_argument("--scale", type=float, default=0.2, help="Coordinate scale (default: 0.2)")
    p.add_argument("--motion_threshold", type=float, default=0.1,
                   help="Motion threshold (default: 0.1)")
    p.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    p.add_argument("--out", default="./benchmark_results",
                   help="Output directory (default: ./benchmark_results)")
    p.add_argument("--csv", default="benchmark.csv",
                   help="Output CSV filename (default: benchmark.csv)")
    p.add_argument("--commands", nargs="+", default=["discover", "predict", "intent"],
                   choices=["discover", "predict", "intent"],
                   help="Commands to benchmark (default: all three)")
    return p.parse_args(argv)


def _parse_junctions(flat: List[float]):
    from verta.verta_geometry import Circle
    if len(flat) % 3 != 0:
        print("ERROR: --junctions must be x z r triplets"); sys.exit(1)
    return [Circle(cx=flat[i], cz=flat[i+1], r=flat[i+2])
            for i in range(0, len(flat), 3)]


def main(argv=None):
    args = parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    junctions = _parse_junctions(args.junctions)

    all_paths = sorted(glob_mod.glob(os.path.join(args.input, args.glob)))
    n_available = len(all_paths)
    if n_available == 0:
        print(f"ERROR: No files match '{args.glob}' in {args.input}")
        sys.exit(1)

    capped_sizes = sorted(set(min(s, n_available) for s in args.sizes))
    if capped_sizes != sorted(args.sizes):
        print(f"WARNING: Only {n_available} files available — "
              f"sizes capped to {capped_sizes} (requested {sorted(args.sizes)})")
    args.sizes = capped_sizes

    print(f"\nVERTA Benchmark")
    print(f"  Input folder : {args.input}")
    print(f"  Files found  : {n_available}")
    print(f"  Junctions    : {len(junctions)}")
    for i, j in enumerate(junctions):
        print(f"      J{i}: x={j.cx}, z={j.cz}, r={j.r}")
    print(f"  Dataset sizes: {args.sizes}")
    print(f"  Commands     : {args.commands}")
    print(f"  Scale        : {args.scale}")
    print(f"  Seed         : {args.seed}")
    print()

    columns = {
        "x": "Headset.Head.Position.X",
        "z": "Headset.Head.Position.Z",
        "t": "Time",
    }

    bench_fns = {
        "discover": bench_discover,
        "predict": bench_predict,
        "intent": bench_intent,
    }

    results: List[BenchmarkResult] = []
    print_table_header()

    for n in sorted(args.sizes):
        print(f"\n  Loading {n} trajectories ...")
        try:
            trajectories = load_trajectories(
                args.input, args.glob, n,
                columns=columns, scale=args.scale,
                motion_threshold=args.motion_threshold,
            )
        except Exception as e:
            print(f"  LOAD ERROR: {e}")
            for cmd in args.commands:
                for viz in (False, True):
                    results.append(BenchmarkResult(
                        workflow=cmd, trajectories=n, total_points=0,
                        junctions=len(junctions), runtime_sec=0.0, peak_ram_mb=0.0,
                        status="error", include_viz=viz, error_msg=str(e),
                    ))
            continue

        total_pts = _count_total_points(trajectories)
        print(f"  Loaded {len(trajectories)} trajectories, {_format_points(total_pts)} points\n")

        for cmd in args.commands:
            for include_viz in (False, True):
                tag = "viz" if include_viz else "noviz"
                run_out = os.path.join(args.out, f"{cmd}_n{n}_{tag}")
                os.makedirs(run_out, exist_ok=True)
                try:
                    r = bench_fns[cmd](trajectories, junctions, run_out, args,
                                       include_viz=include_viz)
                except Exception as e:
                    traceback.print_exc()
                    r = BenchmarkResult(
                        workflow=cmd, trajectories=len(trajectories),
                        total_points=total_pts, junctions=len(junctions),
                        runtime_sec=0.0, peak_ram_mb=0.0,
                        status="error", include_viz=include_viz, error_msg=str(e),
                    )
                results.append(r)
                print_table_row(r)

    print("-" * 100)

    # ---- Save CSV ----
    rows = []
    for r in results:
        d = asdict(r)
        extra = d.pop("extra", {})
        d["total_points_fmt"] = _format_points(d["total_points"])
        d.update(extra)
        rows.append(d)

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out, args.csv)
    df.to_csv(csv_path, index=False)

    # ---- Save JSON ----
    json_path = os.path.join(args.out, "benchmark.json")
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    # ---- Final summary ----
    print(f"\nResults saved to {csv_path}")
    print(f"Results saved to {json_path}")

    ok = df[df["status"] == "ok"]
    if len(ok):
        for viz_val, label in [(False, "Compute only (no plots)"), (True, "Full (with plots)")]:
            subset = ok[ok["include_viz"] == viz_val]
            if subset.empty:
                continue
            print(f"\n\n{label} — Runtime (seconds)")
            print("=" * 60)
            pivot = subset.pivot_table(index="workflow", columns="trajectories",
                                       values="runtime_sec", aggfunc="first")
            print(pivot.to_string(float_format="{:.1f}".format))

            print(f"\n{label} — Peak RAM (MB)")
            print("=" * 60)
            pivot = subset.pivot_table(index="workflow", columns="trajectories",
                                       values="peak_ram_mb", aggfunc="first")
            print(pivot.to_string(float_format="{:.1f}".format))

    print()


if __name__ == "__main__":
    main()
