#!/usr/bin/env python3
"""Measure speculative acceptance on a quote/echo prompt.

The prompt-lookup drafter earns its keep when the continuation repeats the
prompt (quoting, echoing, copying). This script sends a prompt whose answer is
literally the passage, then reports the measured acceptance and adaptive K so
the claim is a measurement rather than an assertion.

    python scripts/spec_acceptance_probe.py --url http://127.0.0.1:8080
"""

from __future__ import annotations

import argparse
import http.client
import json
import urllib.parse

PASSAGE = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox jumps over the lazy dog."
)


def run(url: str, prompt: str, max_tokens: int, timeout: float) -> dict:
    parsed = urllib.parse.urlparse(url)
    connection = http.client.HTTPConnection(parsed.hostname or "127.0.0.1", parsed.port or 8080)
    body = json.dumps(
        {
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "spec": True,
        }
    ).encode("utf-8")
    connection.request(
        "POST",
        "/v1/generate/stream",
        body=body,
        headers={"Content-Type": "application/json"},
    )
    response = connection.getresponse()
    if response.status != 200:
        raise RuntimeError(f"HTTP {response.status}: {response.read()[:300]!r}")
    events = []
    text = []
    while True:
        line = response.readline()
        if not line:
            break
        decoded = line.decode("utf-8", "replace").strip()
        if not decoded.startswith("data:"):
            continue
        payload = decoded[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        event = json.loads(payload)
        events.append(event)
        if event.get("is_finished"):
            break
        if event.get("token"):
            text.append(event["token"])
    connection.close()
    final = events[-1] if events else {}
    return {
        "events": len(events),
        "completion": "".join(text),
        "executor": final.get("executor"),
        "attention_backend": final.get("attention_backend"),
        "acceptance": final.get("speculative_acceptance_rate", 0.0),
        "adaptive_k": final.get("adaptive_k", 0),
        "prefill_tokens": final.get("prefill_tokens", 0),
        "prefix_skip_tokens": final.get("prefix_skip_tokens", 0),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument(
        "--prompt",
        default=(
            "Repeat the passage above exactly, character for character:\n\n"
            f"{PASSAGE}\n\nPassage to repeat:\n"
        ),
    )
    args = parser.parse_args()

    result = run(args.url, args.prompt, args.max_tokens, args.timeout)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
