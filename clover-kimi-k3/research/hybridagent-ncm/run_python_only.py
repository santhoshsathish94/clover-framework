"""Launcher for HybridAgent batch inference without the search tool (no Tavily key on the host).
Only the executor construction differs from `python -m hybrid_inference.run_batch`: the python tool is
registered, the search tool is not; pass --max_search_calls 0 so a <search> call gets the cap notice."""
import functools
import sys

import hybrid_inference.run_batch as rb
from hybrid_agent import build_tool_executor

rb.build_tool_executor = functools.partial(build_tool_executor, require_search=False)

if __name__ == "__main__":
    rb.main(sys.argv[1:])
