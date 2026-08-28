"""Compatibility alias for the prefill-context-parallel cache helpers.

These helpers moved to ``vllm.v1.attention.ops.pcp``.  Out-of-tree platform
plugins (vllm-ascend among them) still import them from the old location, so
keep a thin re-export here rather than forcing every plugin to move in lockstep.
"""

from vllm.v1.attention.ops.pcp import *  # noqa: F401,F403
from vllm.v1.attention.ops.pcp import (  # noqa: F401
    _gather_prefill_cache_inputs,
)
