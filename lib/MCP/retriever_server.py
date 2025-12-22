from mcp.server.fastmcp import FastMCP
from lib.models.retrieval_tool import retriever

mcp = FastMCP("Deteiling Retriever", json_response=True)

@mcp.tool()
def retrieve_in_vectordb(query: str, k: int = 6):
    docs = retriever.invoke(query) or []
    docs = docs[:k]
    return {
        "query": query,
        "k": k,
        "results": [{"content": d.page_content, "metadata": d.metadata} for d in docs],
    }

@mcp.tool()
def health():
    return {"ok": True}


if __name__ == "__main__":
    mcp.run()


# if __name__ == "__main__":
#     mcp.run(transport="http", host="127.0.0.1", port=8001, path="/mcp")