# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Differential tests for the incremental (segment-level) tokenizer cache.

The contract is strict: whenever the cache serves a request, the token ids must
be bit-identical to what the plain tokenizer call would have produced. These
tests assert that directly rather than checking a proxy.
"""

import asyncio
import random
import string
import threading

import pytest

from vllm.renderers.base import BaseRenderer
from vllm.renderers.tokenizer_cache import IncrementalTokenizerCache
from vllm.tokenizers import get_tokenizer

# A spread of tokenizer families: ByteLevel BPE with a chat template, a small
# instruct model, a SentencePiece/Llama tokenizer, and one with essentially no
# added tokens (which must make the cache disable itself).
CHAT_MODELS = [
    "Qwen/Qwen2.5-0.5B-Instruct",
    "HuggingFaceTB/SmolLM2-135M-Instruct",
]
ALL_MODELS = CHAT_MODELS + [
    "hmellor/tiny-random-LlamaForCausalLM",
    "openai-community/gpt2",
]

CODE = 'def f(x: int) -> dict:\n    return {"a": x, "b": [1, 2, 3]}\n'


def _conversation(turns: int) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": "You are a coding agent with tools."}
    ]
    for i in range(turns):
        messages.append(
            {
                "role": "user",
                "content": f"Turn {i}: refactor this.\n```python\n{CODE * 3}```",
            }
        )
        messages.append(
            {"role": "assistant", "content": f"Turn {i} analysis.\n{CODE * 4}\nDone."}
        )
    return messages


def _corpus(tokenizer) -> list[str]:
    """Texts that exercise the boundary cases, built from this tokenizer."""
    specials = [t for t in tokenizer.all_special_tokens if t]
    sep = specials[0] if specials else ""

    texts = [
        "",
        "plain text with no special tokens at all",
        CODE * 10,
        "你好世界\U0001f30f mixed 多字节 content",
        "   leading and trailing whitespace   ",
    ]
    if sep:
        texts += [
            f"a{sep}b",
            f"{sep}{sep}{sep}",
            f"{sep}starts with one",
            f"ends with one{sep}",
            f"trailing   {sep}   leading",
            f"{CODE}{sep}{CODE}{sep}{CODE}",
            # A special token appearing inside content rather than as a boundary.
            f"the user typed {sep} by hand{sep}and then continued",
        ]

    chat_template = getattr(tokenizer, "chat_template", None)
    if chat_template:
        for turns in (1, 3, 8):
            texts.append(
                tokenizer.apply_chat_template(
                    _conversation(turns),
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
    return texts


def _get(model: str):
    return get_tokenizer(model)


# --------------------------------------------------------------- correctness


@pytest.mark.parametrize("model", ALL_MODELS)
def test_encode_is_bit_identical(model: str):
    """The whole point: cached output must equal uncached output exactly."""
    tokenizer = _get(model)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=0.05)
    if not cache.enabled:
        pytest.skip(f"cache self-check disabled for {model}")

    for text in _corpus(tokenizer):
        expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        # Twice: cold (all misses) and warm (mostly hits). Both must match.
        assert cache.encode(text) == expected, f"cold miss mismatch on {text[:60]!r}"
        assert cache.encode(text) == expected, f"warm hit mismatch on {text[:60]!r}"


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_growing_conversation_stays_exact_and_hits(model: str):
    """The agent workload: each turn re-sends the prefix and appends a suffix."""
    tokenizer = _get(model)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=0.05)
    if not cache.enabled:
        pytest.skip(f"cache self-check disabled for {model}")

    for turns in range(1, 12):
        text = tokenizer.apply_chat_template(
            _conversation(turns), tokenize=False, add_generation_prompt=True
        )
        expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        assert cache.encode(text) == expected, f"mismatch at turn {turns}"

    # By the last turns nearly every segment is a repeat, so the hit rate must
    # be high — otherwise the cache is correct but pointless.
    info = cache.stat()
    assert info.total > 0
    assert info.hit_ratio > 0.7, f"hit ratio too low: {info}"


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_capacity_pressure_does_not_break_correctness(model: str):
    """Under eviction the cache must still return exact results."""
    tokenizer = _get(model)
    # Tiny capacity: almost everything gets evicted immediately.
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=1e-6)
    if not cache.enabled:
        pytest.skip(f"cache self-check disabled for {model}")

    for text in _corpus(tokenizer):
        expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        assert cache.encode(text) == expected


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_concurrent_encode_is_exact(model: str):
    """The renderer calls this from a thread pool."""
    tokenizer = _get(model)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=0.05)
    if not cache.enabled:
        pytest.skip(f"cache self-check disabled for {model}")

    texts = [
        tokenizer.apply_chat_template(
            _conversation(t), tokenize=False, add_generation_prompt=True
        )
        for t in range(1, 9)
    ]
    expected = [
        list(tokenizer(t, add_special_tokens=False)["input_ids"]) for t in texts
    ]

    errors: list[str] = []
    barrier = threading.Barrier(8)

    def worker(seed: int) -> None:
        barrier.wait()
        for i in range(len(texts)):
            idx = (i + seed) % len(texts)
            if cache.encode(texts[idx]) != expected[idx]:
                errors.append(f"thread {seed} mismatch on text {idx}")

    threads = [threading.Thread(target=worker, args=(s,)) for s in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, errors


# ------------------------------------------------------------------ gating


class _StubTokenizer:
    """Minimal TokenizerLike stand-in driven by a fake vocabulary."""

    def __init__(self, added: dict[str, int], *, break_identity: bool = False):
        self._added = added
        self._break_identity = break_identity

    def get_added_vocab(self) -> dict[str, int]:
        return dict(self._added)

    @property
    def all_special_tokens(self) -> list[str]:
        return list(self._added)

    def __call__(self, text, add_special_tokens: bool = True, **kwargs):
        # One id per character, plus a sentinel that depends on the *whole*
        # string when asked to break the identity — which is exactly the
        # failure mode a prefix-space pre-tokenizer produces.
        ids = [ord(c) % 997 for c in text]
        if self._break_identity and text:
            ids = [len(text) % 97] + ids
        if add_special_tokens:
            ids = [1] + ids
        return {"input_ids": ids}


def test_self_check_disables_when_identity_breaks():
    tokenizer = _StubTokenizer({"<s>": 0, "</s>": 1}, break_identity=True)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=0.01)
    assert not cache.enabled
    assert not cache.is_eligible(add_special_tokens=False)


def test_self_check_passes_on_a_composable_tokenizer():
    tokenizer = _StubTokenizer({"<s>": 0, "</s>": 1})
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=0.01)
    assert cache.enabled
    text = "hello<s>world</s>tail"
    assert cache.encode(text) == tokenizer(text, add_special_tokens=False)["input_ids"]


def test_disabled_without_added_tokens():
    cache = IncrementalTokenizerCache(_StubTokenizer({}), capacity_gb=0.01)
    assert not cache.enabled


def test_add_special_tokens_gating():
    # This stub prepends id 1 when add_special_tokens=True, so it is NOT a
    # no-op and those requests must be refused.
    cache = IncrementalTokenizerCache(_StubTokenizer({"<s>": 0}), capacity_gb=0.01)
    assert cache.enabled
    assert cache.is_eligible(add_special_tokens=False)
    assert not cache.is_eligible(add_special_tokens=True)


@pytest.mark.parametrize("model", CHAT_MODELS)
def test_add_special_tokens_gating_real_tokenizer(model: str):
    tokenizer = _get(model)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=0.01)
    if not cache.enabled:
        pytest.skip(f"cache self-check disabled for {model}")

    # Whatever the self-check concluded, it must be consistent with reality.
    probe = "hello world, this is a probe string"
    is_noop = list(tokenizer(probe, add_special_tokens=True)["input_ids"]) == list(
        tokenizer(probe, add_special_tokens=False)["input_ids"]
    )
    if is_noop:
        assert cache.is_eligible(add_special_tokens=True)
    else:
        assert not cache.is_eligible(add_special_tokens=True)


# -------------------------------------------------------------------- fuzz

# Alphabets chosen to hit what would break the concatenation identity:
# whitespace runs (the Metaspace/prefix-space trap), multi-byte text, and
# code punctuation.
_FUZZ_ALPHABETS = [
    string.ascii_letters + " \n\t",
    string.printable,
    "你好世界再来一次测试内容 \n",
    "\U0001f30f\U0001f600éüß ",
    "def return if else class import from as with await ()[]{}:,.\n\t ",
    "     \n\n\t\t   ",
]


def _random_text(rng: random.Random, specials: list[str]) -> str:
    parts: list[str] = []
    for _ in range(rng.randint(1, 8)):
        alphabet = rng.choice(_FUZZ_ALPHABETS)
        # Mix of empty, tiny, and large chunks.
        length = rng.choice([0, 1, 2, 5, 17, 63, 64, 65, 200, 1000])
        parts.append("".join(rng.choice(alphabet) for _ in range(length)))
        if specials and rng.random() < 0.7:
            parts.append(rng.choice(specials))
    return "".join(parts)


@pytest.mark.parametrize("model", ALL_MODELS)
def test_fuzz_encode_is_bit_identical(model: str):
    """Randomised differential test - the real guarantee behind the cache."""
    tokenizer = _get(model)
    cache = IncrementalTokenizerCache(tokenizer, capacity_gb=0.05)
    if not cache.enabled:
        pytest.skip(f"cache self-check disabled for {model}")

    get_added_vocab = getattr(tokenizer, "get_added_vocab", None)
    specials = (
        list(get_added_vocab().keys())
        if callable(get_added_vocab)
        else [t for t in tokenizer.all_special_tokens if t]
    )

    rng = random.Random(1234)
    for i in range(200):
        text = _random_text(rng, specials)
        expected = list(tokenizer(text, add_special_tokens=False)["input_ids"])
        assert cache.encode(text) == expected, f"case {i}: {text[:120]!r}"


# ------------------------------------------------------- chat fast path glue


class _FakeCache:
    """Stands in for IncrementalTokenizerCache in the dispatch tests."""

    def __init__(self, armed: bool):
        self.chat_path_enabled = armed
        self.seen: list[str] = []

    def encode(self, text: str) -> list[int]:
        self.seen.append(text)
        return [1, 2, 3]


class _StubRenderer:
    """Borrows the BaseRenderer helpers without building a full renderer."""

    _chat_cache_usable = BaseRenderer._chat_cache_usable
    _chat_render_cached = BaseRenderer._chat_render_cached
    _chat_render_cached_async = BaseRenderer._chat_render_cached_async
    use_unified_vision_chunk = False

    def __init__(self, cache):
        self._tokenizer_cache = cache


def _recorder(calls):
    def render(**kw):
        calls.append(kw)
        return "RENDERED" if kw.get("tokenize") is False else [9, 9]

    return render


def test_chat_fast_path_used_when_armed():
    cache = _FakeCache(armed=True)
    calls: list[dict] = []
    out = _StubRenderer(cache)._chat_render_cached(
        {"tokenize": True, "add_generation_prompt": True}, _recorder(calls)
    )
    assert out == [1, 2, 3]
    assert cache.seen == ["RENDERED"]
    assert calls == [{"tokenize": False, "add_generation_prompt": True}]


def test_chat_fast_path_skipped_when_not_armed():
    cache = _FakeCache(armed=False)
    calls: list[dict] = []
    out = _StubRenderer(cache)._chat_render_cached({"tokenize": True}, _recorder(calls))
    assert out == [9, 9]
    assert cache.seen == []
    assert calls == [{"tokenize": True}]


def test_chat_fast_path_skipped_when_caller_wants_text():
    # tokenize=False means the caller wants the string, not token ids.
    cache = _FakeCache(armed=True)
    calls: list[dict] = []
    out = _StubRenderer(cache)._chat_render_cached(
        {"tokenize": False}, _recorder(calls)
    )
    assert out == "RENDERED"
    assert cache.seen == []


def test_chat_fast_path_async():
    cache = _FakeCache(armed=True)
    calls: list[dict] = []
    sync = _recorder(calls)

    async def render(**kw):
        return sync(**kw)

    out = asyncio.run(
        _StubRenderer(cache)._chat_render_cached_async({"tokenize": True}, render)
    )
    assert out == [1, 2, 3]
    assert cache.seen == ["RENDERED"]
