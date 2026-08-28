"""Per-row top-k selection for the Qwen4Exp QSA indexer, for Ascend NPU.

Upstream selects the highest-scoring compressed blocks with the CUDA kernels
``torch.ops._C.{persistent,cooperative}_topk``.  Those live in vLLM's compiled
``_C`` extension, which is not built on Ascend (VLLM_TARGET_DEVICE=empty), so we
provide the same selection with device-portable tensor ops.

Contract, matching the upstream operator schema
``persistent_topk(Tensor logits, Tensor lengths, Tensor! output, Tensor workspace,
int k, int max_seq_len) -> ()``:

    logits  [rows, cols]  float32  - score of every candidate column
    lengths [rows]        int32    - number of *valid* leading columns per row
    output  [rows, k]     int32    - written in place with the selected column
                                     indices; slots with no valid candidate are
                                     filled with the sentinel -1, which is the
                                     invalid marker the downstream expansion
                                     kernel already emits and understands.

Shapes are static and there is no host synchronisation or data-dependent
indexing, so the whole thing is safe to capture in an NPU graph.
"""

from __future__ import annotations

import torch

INVALID_INDEX = -1


def topk_per_row(
    logits: torch.Tensor,
    lengths: torch.Tensor,
    output: torch.Tensor,
    k: int,
) -> None:
    if logits.ndim != 2:
        raise ValueError("QSA top-k logits must be [rows, cols]")
    rows, cols = logits.shape
    if lengths.shape != (rows,):
        raise ValueError("QSA top-k lengths must have one entry per row")
    if output.shape != (rows, k):
        raise ValueError("QSA top-k output must be [rows, k]")
    if k <= 0:
        raise ValueError("QSA top-k requires a positive k")

    device = logits.device
    counts = lengths.to(torch.int64).unsqueeze(1)                       # [rows, 1]

    # Score only the leading `lengths[row]` columns; everything past the row's
    # valid prefix is pushed below any real score so it can never be selected.
    columns = torch.arange(cols, device=device).unsqueeze(0)            # [1, cols]
    in_range = columns < counts
    scores = torch.where(in_range, logits.float(), float("-inf"))

    width = min(k, cols)
    chosen = scores.topk(width, dim=-1, sorted=True).indices.to(torch.int32)

    if width < k:  # fewer columns than requested slots: pad the tail
        chosen = torch.cat(
            [chosen, chosen.new_full((rows, k - width), INVALID_INDEX)], dim=1
        )

    # A row with fewer than k valid columns leaves its trailing slots empty.
    slots = torch.arange(k, device=device).unsqueeze(0)                 # [1, k]
    output.copy_(torch.where(slots < counts, chosen, torch.full_like(chosen, INVALID_INDEX)))
