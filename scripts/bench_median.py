#!/usr/bin/env python3
"""Run the NexusKV end-to-end benchmark N times and report medians.

Single-run numbers are noisy on a laptop GPU (clock boost, thermal state,
Windows scheduler). This wrapper runs the existing measured benchmark several
times and reports the median of each metric, keeping every raw run.

It never invents numbers: the medians are computed from the JSON that
`bench_e2e.py` actually measured, and each raw run is preserved under
`benches/results/`.

    python scripts/bench_median.py --runs 3 -- --clients 8 --max-tokens 64 --spec
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time


def median(values: list[float]) -> float:
    return statistics.median(values) if values else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--output", default=None, help="median summary JSON path")
    parser.add_argument("--label", default="run", help="prefix for per-run files")
    parser.add_argument(
        "bench_args",
        nargs=argparse.REMAINDER,
        help="arguments forwarded to bench_e2e.py after `--`",
    )
    args = parser.parse_args()

    forwarded = list(args.bench_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.runs < 1:
        print("--runs must be at least 1", file=sys.stderr)
        return 2
    if args.runs < 3:
        print(
            "warning: fewer than 3 runs; the median is not a stable estimate",
            file=sys.stderr,
        )

    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bench_e2e.py")
    results_directory = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "benches",
        "results",
    )
    os.makedirs(results_directory, exist_ok=True)

    runs: list[dict] = []
    for index in range(args.runs):
        output = os.path.join(
            results_directory, f"{args.label}-{time.strftime('%Y%m%d-%H%M%S')}-{index}.json"
        )
        command = [
            sys.executable,
            script,
            "--url",
            args.url,
            "--output",
            output,
            *forwarded,
        ]
        print(f"--- run {index + 1}/{args.runs} ---", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        sys.stdout.write(completed.stdout)
        sys.stderr.write(completed.stderr)
        if completed.returncode != 0:
            print(f"run {index + 1} failed with exit code {completed.returncode}", file=sys.stderr)
            return completed.returncode
        with open(output, "r", encoding="utf-8") as handle:
            run = json.load(handle)
        run["_output"] = output
        runs.append(run)

    throughputs = [run.get("tokens_per_sec", 0.0) for run in runs]
    ttft_p50 = [run.get("ttft_ms", {}).get("p50", 0.0) for run in runs]
    ttft_p99 = [run.get("ttft_ms", {}).get("p99", 0.0) for run in runs]
    itl_p50 = [run.get("itl_ms", {}).get("p50", 0.0) for run in runs]
    itl_p99 = [run.get("itl_ms", {}).get("p99", 0.0) for run in runs]
    per_client = [
        statistics.median(run.get("per_client_tokens_per_sec", []) or [0.0]) for run in runs
    ]
    acceptance = [run.get("acceptance", 0.0) for run in runs]

    summary = {
        "runs": args.runs,
        "url": args.url,
        "clients": runs[0].get("clients"),
        "max_tokens": runs[0].get("max_tokens"),
        "spec": runs[0].get("spec"),
        "executor": runs[0].get("executor"),
        "tokens_per_sec": {
            "median": median(throughputs),
            "min": min(throughputs),
            "max": max(throughputs),
            "all": throughputs,
        },
        "per_client_tokens_per_sec_median": median(per_client),
        "ttft_ms": {
            "p50_median": median(ttft_p50),
            "p99_median": median(ttft_p99),
        },
        "itl_ms": {
            "p50_median": median(itl_p50),
            "p99_median": median(itl_p99),
        },
        "acceptance_median": median(acceptance),
        "raw_runs": [run["_output"] for run in runs],
        "note": (
            "medians of measured runs from bench_e2e.py; each raw run is on disk"
        ),
    }

    rendered = json.dumps(summary, indent=2)
    if args.output:
        os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"wrote {args.output}")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
