#!/usr/bin/env python3
"""NexusKV end-to-end serving benchmark.

Measures real SSE token streams from a running NexusKV server: aggregate
tokens/sec, TTFT p50/p99, ITL p50/p99, and honest engine counters. This script
never fabricates numbers and does not compare against llama.cpp by itself; the
--compare flag prints the protocol for a same-GPU, same-model manual run.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import sys
import threading
import time
import urllib.parse
import urllib.request


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * fraction
    lower = int(rank)
    upper = min(lower + 1, len(ordered) - 1)
    weight = rank - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def run_client(
    scheme: str,
    host: str,
    port: int,
    payload: dict,
    barrier: threading.Barrier,
    results: list[dict],
    lock: threading.Lock,
    index: int,
) -> None:
    result: dict = {
        "client": index,
        "ok": False,
        "error": None,
        "tokens": 0,
        "start": 0.0,
        "end": 0.0,
        "token_times": [],
        "last_event": {},
    }
    connection = None
    try:
        barrier.wait(timeout=30)
        body = json.dumps(payload).encode("utf-8")
        connection = http.client.HTTPConnection(host, port, timeout=120)
        started = time.perf_counter()
        result["start"] = started
        connection.request(
            "POST",
            "/v1/generate/stream",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
        token_times: list[float] = []
        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("token"):
                token_times.append(time.perf_counter())
            result["last_event"] = event
            if event.get("is_finished"):
                break
        result["end"] = time.perf_counter()
        result["token_times"] = token_times
        result["tokens"] = len(token_times)
        result["ok"] = True
    except Exception as error:  # noqa: BLE001 - reported honestly in JSON
        result["error"] = str(error)
        result["end"] = time.perf_counter()
    finally:
        if connection is not None:
            connection.close()
        with lock:
            results.append(result)


def single_request_ttft(host: str, port: int, payload: dict, timeout: float) -> dict:
    """Run one SSE request and return measured time-to-first-token."""
    connection = None
    started = time.perf_counter()
    try:
        body = json.dumps(payload).encode("utf-8")
        connection = http.client.HTTPConnection(host, port, timeout=timeout)
        connection.request(
            "POST",
            "/v1/generate/stream",
            body=body,
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            raise RuntimeError(f"HTTP {response.status}: {response.read()[:200]!r}")
        first_token = None
        tokens = 0
        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            data = text[5:].strip()
            if not data or data == "[DONE]":
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                raise RuntimeError(str(event["error"]))
            if event.get("token"):
                if first_token is None:
                    first_token = time.perf_counter()
                tokens += 1
            if event.get("is_finished"):
                break
        end = time.perf_counter()
        return {
            "ok": True,
            "ttft_ms": ((first_token - started) * 1000.0) if first_token else 0.0,
            "total_ms": (end - started) * 1000.0,
            "tokens": tokens,
            "error": None,
        }
    except Exception as error:  # noqa: BLE001 - reported honestly in JSON
        return {
            "ok": False,
            "ttft_ms": 0.0,
            "total_ms": (time.perf_counter() - started) * 1000.0,
            "tokens": 0,
            "error": str(error),
        }
    finally:
        if connection is not None:
            connection.close()


def fetch_metrics(url: str, timeout: float) -> dict:
    try:
        with urllib.request.urlopen(f"{url}/v1/metrics", timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except Exception:  # noqa: BLE001
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--clients", type=int, default=4)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--prompt", default="benchmark: paged KV cache and continuous batching")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--spec", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--prefix-probe",
        action="store_true",
        help="measure cold-vs-warm TTFT on two identical unique prompts",
    )
    parser.add_argument("--prefix-probe-nonce", default=None)
    parser.add_argument(
        "--prefix-probe-exact",
        action="store_true",
        help="use --prompt exactly (no nonce); caller must make it unique",
    )
    parser.add_argument(
        "--prefix-probe-only",
        action="store_true",
        help="skip the concurrent workload and measure only the cold/warm probe",
    )
    parser.add_argument("--compare", action="store_true")
    args = parser.parse_args()

    parsed = urllib.parse.urlparse(args.url)
    if parsed.scheme not in ("http", ""):
        print("bench_e2e.py currently supports http:// URLs only", file=sys.stderr)
        return 2
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 80
    base_url = f"http://{host}:{port}"

    payload = {
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "spec": args.spec,
    }
    results: list[dict] = []
    if not args.prefix_probe_only:
        barrier = threading.Barrier(args.clients)
        lock = threading.Lock()
        threads = [
            threading.Thread(
                target=run_client,
                args=(parsed.scheme, host, port, payload, barrier, results, lock, index),
                daemon=True,
            )
            for index in range(args.clients)
        ]
        for thread in threads:
            thread.start()
        deadline = time.perf_counter() + args.timeout
        for thread in threads:
            remaining = max(0.0, deadline - time.perf_counter())
            thread.join(timeout=remaining)
        if any(thread.is_alive() for thread in threads):
            print("benchmark timed out", file=sys.stderr)
            return 3

    successes = [result for result in results if result["ok"]]
    failures = [result for result in results if not result["ok"]]
    ttfts: list[float] = []
    itls: list[float] = []
    per_client_tps: list[float] = []
    total_tokens = 0
    for result in successes:
        times = result["token_times"]
        total_tokens += len(times)
        if times:
            ttfts.append((times[0] - result["start"]) * 1000.0)
        for previous, current in zip(times, times[1:]):
            itls.append((current - previous) * 1000.0)
        if len(times) > 1:
            per_client_tps.append(
                (len(times) - 1) / max(1e-9, times[-1] - times[0])
            )

    wall_start = min((result["start"] for result in successes), default=0.0)
    wall_end = max((result["end"] for result in successes), default=0.0)
    aggregate_tps = total_tokens / max(1e-9, wall_end - wall_start) if successes else 0.0
    last_events = [result["last_event"] for result in successes if result["last_event"]]
    final_event = last_events[-1] if last_events else {}
    metrics = fetch_metrics(base_url, args.timeout)

    summary = {
        "url": base_url,
        "clients": args.clients,
        "successful_clients": len(successes),
        "failed_clients": len(failures),
        "errors": [result["error"] for result in failures],
        "max_tokens": args.max_tokens,
        "spec": args.spec,
        "executor": final_event.get("executor") or metrics.get("executor"),
        "total_tokens": total_tokens,
        "tokens_per_sec": aggregate_tps,
        "per_client_tokens_per_sec": per_client_tps,
        "ttft_ms": {
            "p50": percentile(ttfts, 0.50),
            "p99": percentile(ttfts, 0.99),
        },
        "itl_ms": {
            "p50": percentile(itls, 0.50),
            "p99": percentile(itls, 0.99),
        },
        "prefix_cache_hit_ratio": metrics.get("prefix_cache_hit_ratio", 0.0),
        "prefix_skip_tokens": metrics.get("prefix_skip_tokens", 0),
        "prefill_tokens": metrics.get("prefill_tokens", 0),
        "acceptance": metrics.get("acceptance", 0.0),
        "adaptive_k": metrics.get("adaptive_k", 0),
        "evictions": metrics.get("evictions", 0),
        "page_faults": metrics.get("page_faults", 0),
        "swapped_out": metrics.get("swapped_out", 0),
        "swapped_in": metrics.get("swapped_in", 0),
        "swap_bytes_out": metrics.get("swap_bytes_out", 0),
        "swap_bytes_in": metrics.get("swap_bytes_in", 0),
        "disk_spills": metrics.get("disk_spills", 0),
        "disk_loads": metrics.get("disk_loads", 0),
        "cow_copies": metrics.get("cow_copies", 0),
        "preemptions": metrics.get("preemptions", 0),
        "total_tokens_generated": metrics.get("total_tokens_generated", 0),
    }

    if args.prefix_probe:
        nonce = args.prefix_probe_nonce or f"{time.time_ns()}"
        probe_prompt = (
            args.prompt
            if args.prefix_probe_exact
            else f"{args.prompt} [prefix-probe {nonce}]"
        )
        probe_payload = dict(payload)
        probe_payload["prompt"] = probe_prompt
        probe_payload["max_tokens"] = min(args.max_tokens, 8)
        prefill_before = metrics.get("prefill_tokens", 0)
        skip_before = metrics.get("prefix_skip_tokens", 0)
        cold = single_request_ttft(host, port, probe_payload, args.timeout)
        warm = single_request_ttft(host, port, probe_payload, args.timeout)
        after = fetch_metrics(base_url, args.timeout)
        speedup = (
            cold["ttft_ms"] / warm["ttft_ms"]
            if cold["ok"] and warm["ok"] and warm["ttft_ms"] > 0
            else 0.0
        )
        summary["prefix_probe"] = {
            "prompt": probe_prompt,
            "cold": cold,
            "warm": warm,
            "ttft_speedup": speedup,
            "prefill_tokens_before": prefill_before,
            "prefill_tokens_after": after.get("prefill_tokens", 0),
            "prefill_delta": after.get("prefill_tokens", 0) - prefill_before,
            "skip_tokens_before": skip_before,
            "skip_tokens_after": after.get("prefix_skip_tokens", 0),
            "skip_delta": after.get("prefix_skip_tokens", 0) - skip_before,
            "note": "cold is the first request for this unique prompt; warm is an identical re-hit",
        }

    rendered = json.dumps(summary, indent=2)
    if args.output:
        output_directory = os.path.dirname(os.path.abspath(args.output))
        os.makedirs(output_directory, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(rendered + "\n")
        print(f"wrote {args.output}")
    else:
        print(rendered)

    if args.compare:
        print(
            "\nllama.cpp comparison protocol (manual, same hardware/model):\n"
            "  1. Run llama.cpp's server on the same GPU with the same Qwen3-0.6B\n"
            "     quantized weights and the same prompt/max-tokens/concurrency.\n"
            "  2. Record tokens/sec, TTFT p50/p99, and ITL p50/p99 from its logs.\n"
            "  3. Report the exact commit, build flags, GPU, and model file.\n"
            "This script does not assert a comparison result by itself."
        )

    if failures:
        print(f"{len(failures)} client(s) failed", file=sys.stderr)
        return 1
    if total_tokens == 0 and not args.prefix_probe_only:
        print("no tokens were produced", file=sys.stderr)
        return 1
    if summary["ttft_ms"]["p50"] > summary["ttft_ms"]["p99"]:
        print("TTFT percentiles are not ordered", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
