import os
import sys

from langchain_core.documents import Document
from langchain_core.tools import tool

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


@tool("retrieve_in_vectordb", response_format="content_and_artifact")
async def retrieve_in_vectordb(query: str, k: int = 6):
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "lib.MCP.retriever_server"],  # <-- ваш сервер
        env={**os.environ, "PYTHONPATH": os.getcwd()},
    )

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            res = await session.call_tool(
                "retrieve_in_vectordb",
                arguments={"query": query, "k": k},
            )

    payload = res.structuredContent or {}
    results = payload.get("results", []) or []

    docs = [
        Document(page_content=r.get("content", ""), metadata=r.get("metadata") or {})
        for r in results
        if isinstance(r, dict)
    ]

    return {"content": f"Retrieved {len(docs)} docs via MCP.", "artifact": docs}