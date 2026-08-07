import asyncio
from unittest.mock import patch

from mcp import Client

from innerthink.client import InnerThinkClient
from innerthink.mcp_server import mcp


def test_mcp_lists_and_calls_qwen_tools() -> None:
    async def exercise() -> None:
        with patch.object(InnerThinkClient, "generate", return_value={"answer": "4"}):
            async with Client(mcp) as client:
                tools = await client.list_tools()
                names = {tool.name for tool in tools.tools}
                result = await client.call_tool(
                    "qwen_reason",
                    {"prompt": "What is 2 + 2?"},
                )

        assert names == {
            "qwen_reason",
            "qwen_compare",
            "qwen_intervene",
            "recall_reasoning_feedback",
            "remember_reasoning_feedback",
            "snowflake_economy_summary",
            "qwen_dataset_demo",
        }
        assert result.structured_content == {"answer": "4"}

    asyncio.run(exercise())
