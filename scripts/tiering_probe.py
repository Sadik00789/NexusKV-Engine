#!/usr/bin/env python3
"""Force tiering to fire: shared prefix, eviction, then a demand page-in.

A large pool never evicts, so tiering stays silent. This script runs in rounds:

  1. concurrent clients on one shared prefix (its blocks become Cached)
  2. unique filler prompts that fill the pool and push those cached blocks to
     the tier store (`swapped_out` / `swap_bytes_out`)
  3. the same shared-prefix workload again, which hits the swapped trie nodes
     and demand-pages them back (`page_faults` / `swapped_in`)

Everything reported is a delta the engine actually measured.

    python scripts/tiering_probe.py --url http://127.0.0.1:8080 --clients 6
"""

from __future__ import annotations

import argparse
import http.client
import json
import threading
import time
import urllib.parse
import urllib.request

SHARED = (
    "system: you are a deterministic assistant. repeat the passage above. "
    "the quick brown fox jumps over the lazy dog. "
)


def post_stream(host: str, port: int, payload: dict, timeout: float) -> dict:
    connection = http.client.HTTPConnection(host, port, timeout=timeout)
    tokens = 0
    try:
        connection.request(
            "POST",
            "/v1/generate/stream",
            body=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        response = connection.getresponse()
        if response.status != 200:
            return {"ok": False, "tokens": 0, "error": f"HTTP {response.status}"}
        while True:
            line = response.readline()
            if not line:
                break
            text = line.decode("utf-8", "replace").strip()
            if not text.startswith("data:"):
                continue
            body = text[5:].strip()
            if not body or body == "[DONE]":
                continue
            try:
                event = json.loads(body)
            except json.JSONDecodeError:
                continue
            if event.get("error"):
                return {"ok": False, "tokens": tokens, "error": str(event["error"])}
            if event.get("token"):
                tokens += 1
            if event.get("is_finished"):
                break
        return {"ok": True, "tokens": tokens, "error": None}
    finally:
        connection.close()


def metrics(url: str) -> dict:
    with urllib.request.urlopen(f"{url}/v1/metrics", timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--clients", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--fillers", type=int, default=24)
    parser.add_argument(
        "--filler-blocks",
        type=int,
        default=6,
        help="approximate blocks each filler prompt occupies (16 tokens/block)",
    )
    parser.add_argument("--remeasure", type=int, default=2, help="shared-prefix rounds after eviction")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    parsed = urllib.parse.urlparse(args.url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8080

    before = metrics(args.url)

    errors: list[str] = []
    lock = threading.Lock()
    totals = {"tokens": 0}

    def worker(index: int) -> None:
        result = post_stream(
            host,
            port,
            {"prompt": SHARED, "max_tokens": args.max_tokens, "temperature": 0.0, "spec": True},
            args.timeout,
        )
        with lock:
            totals["tokens"] += result["tokens"]
            if not result["ok"] and result["error"]:
                errors.append(result["error"])

    threads = [threading.Thread(target=worker, args=(index,)) for index in range(args.clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=args.timeout)

    # Round 2: fill the pool so the now-cached shared prefix is evicted. Each
    # prompt is unique and long enough to occupy several blocks, so the pool
    # actually runs out of free blocks and the LRU has to spill.
    fillers_done = 0
    for index in range(args.fillers):
        unique = f"filler {time.time_ns()}-{index} " + ("alpha beta gamma delta " * args.filler_blocks)
        result = post_stream(
            host,
            port,
            {"prompt": unique, "max_tokens": 2, "temperature": 0.0, "spec": False},
            args.timeout,
        )
        fillers_done += 1
        if not result["ok"] and result["error"]:
            errors.append(result["error"])

    mid = metrics(args.url)

    # Round 3: re-request the shared prefix. Its trie nodes are swapped, so this
    # must demand-page them back in.
    for round_index in range(args.remeasure):
        threads = [
            threading.Thread(target=worker, args=(index,))
            for index in range(args.clients)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=args.timeout)

    after = metrics(args.url)
    summary = {
        "clients": args.clients,
        "max_tokens": args.max_tokens,
        "fillers": fillers_done,
        "tokens_generated": totals["tokens"],
        "errors": errors[:5],
        "total_blocks": after.get("total_blocks"),
        "evictions": after.get("evictions", 0) - before.get("evictions", 0),
        "page_faults": after.get("page_faults", 0) - before.get("page_faults", 0),
        "swapped_out": after.get("swapped_out", 0) - before.get("swapped_out", 0),
        "swapped_in": after.get("swapped_in", 0) - before.get("swapped_in", 0),
        "swap_bytes_out": after.get("swap_bytes_out", 0) - before.get("swap_bytes_out", 0),
        "swap_bytes_in": after.get("swap_bytes_in", 0) - before.get("swap_bytes_in", 0),
        "disk_spills": after.get("disk_spills", 0) - before.get("disk_spills", 0),
        "disk_loads": after.get("disk_loads", 0) - before.get("disk_loads", 0),
        "preemptions": after.get("preemptions", 0) - before.get("preemptions", 0),
        "prefix_skip_tokens": after.get("prefix_skip_tokens", 0),
        "evictions_before_remeasure": mid.get("evictions", 0) - before.get("evictions", 0),
        "executor": after.get("executor"),
        "attention_backend": after.get("attention_backend"),
        "note": "tier counters are deltas measured across this run",
    }
    print(json.dumps(summary, indent=2))
    return 0 if not errors and summary["tokens_generated"] > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
