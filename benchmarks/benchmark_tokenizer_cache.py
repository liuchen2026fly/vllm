# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the incremental (segment-level) tokenizer cache.

Simulates the workload it targets: a multi-turn agent conversation where each
turn re-sends the whole transcript and appends a short suffix. Reports the
per-turn tokenization cost with and without the cache, and verifies that the
two produce identical token ids.

Usage:
    python benchmarks/benchmark_tokenizer_cache.py --model Qwen/Qwen3-8B
"""

import argparse
import os
import time

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm.renderers.tokenizer_cache import IncrementalTokenizerCache  # noqa: E402
from vllm.tokenizers import get_tokenizer  # noqa: E402

CODE = """def process_batch(self, requests: list[Request]) -> dict[str, Any]:
    results = {}
    for req in requests:
        if req.status != Status.PENDING:
            continue
        out = self._executor.submit(self._handle, req).result(timeout=30.0)
        results[req.id] = {"ok": True, "value": out}
    return results
"""


def build_turns(tokenizer, num_turns: int, blob: int) -> list[str]:
    """Render the transcript after each turn, as a server would see it."""
    messages = [{"role": "system", "content": "You are a coding agent."}]
    rendered = []
    for i in range(num_turns):
        messages.append(
            {"role": "user", "content": f"Turn {i}: refactor.\n{CODE * blob}"}
        )
        rendered.append(
            tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        )
        messages.append(
            {"role": "assistant", "content": f"Turn {i} done.\n{CODE * blob}"}
        )
    return rendered


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--turns", type=int, default=16)
    parser.add_argument("--blob", type=int, default=6)
    parser.add_argument("--cache-gb", type=float, default=0.5)
    args = parser.parse_args()

    tokenizer = get_tokenizer(args.model)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=args.cache_gb)
    if not cache.enabled:
        raise SystemExit(f"cache self-check refused {args.model}; nothing to measure")

    turns = build_turns(tokenizer, args.turns, args.blob)

    baseline_total = 0.0
    cached_total = 0.0
    print(f"{'turn':>5} {'tokens':>9} {'no cache':>11} {'cache':>10} {'speedup':>9}")
    for i, text in enumerate(turns):
        start = time.perf_counter()
        expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        baseline = time.perf_counter() - start

        start = time.perf_counter()
        got = cache.encode(text)
        cached = time.perf_counter() - start

        if got != expected:
            raise SystemExit(f"MISMATCH at turn {i} - cache is incorrect")

        baseline_total += baseline
        cached_total += cached
        print(
            f"{i:5d} {len(expected):9d} {baseline * 1000:9.2f}ms "
            f"{cached * 1000:8.2f}ms {baseline / max(cached, 1e-9):8.1f}x"
        )

    info = cache.stat()
    print(
        f"\nsession total: no cache {baseline_total * 1000:.1f}ms -> "
        f"cache {cached_total * 1000:.1f}ms "
        f"({baseline_total / max(cached_total, 1e-9):.1f}x)"
    )
    print(f"segment hit ratio: {info.hit_ratio:.1%} ({info.hits}/{info.total})")
    print("token ids identical on every turn: yes")


if __name__ == "__main__":
    main()
