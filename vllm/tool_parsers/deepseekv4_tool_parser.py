# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import regex as re

from vllm.logger import init_logger
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.deepseekv32_tool_parser import DeepSeekV32ToolParser

logger = init_logger(__name__)


class DeepSeekV4ToolParser(DeepSeekV32ToolParser):
    """
    DeepSeek V4 tool parser.

    DSV4 uses the same DSML XML format as V3.2, but with
    ``tool_calls`` instead of ``function_calls`` as the
    wrapper tag:

    <｜DSML｜tool_calls>
    <｜DSML｜invoke name="get_weather">
    <｜DSML｜parameter name="location" string="true">杭州</｜DSML｜parameter>
    </｜DSML｜invoke>
    </｜DSML｜tool_calls>
    """

    def __init__(self, tokenizer: TokenizerLike):
        super().__init__(tokenizer)

        # Override the wrapper tokens from function_calls → tool_calls
        self.tool_call_start_token = "<｜DSML｜tool_calls>"
        self.tool_call_end_token = "</｜DSML｜tool_calls>"

        # Rebuild the complete-match regex with the new wrapper tokens
        self.tool_call_complete_regex = re.compile(
            r"<｜DSML｜tool_calls>(.*?)</｜DSML｜tool_calls>",
            re.DOTALL,
        )
