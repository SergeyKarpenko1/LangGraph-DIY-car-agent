from langchain_core.messages import HumanMessage
from lib.models.retrieval_tool import retriever_tool


from lib.models.retrieval_tool import retriever_tool


def test_retriever_tool(query: str):
    """Проверка работы retriever_tool + печать порезанного (pruned) контента."""
    print(f"\n>>> Проверка инструмента retriever_tool\nЗапрос: {query}\n")

    result = retriever_tool.invoke({"query": query})

    if isinstance(result, dict):
        content = result.get("content") or ""
        artifact = result.get("artifact") or []

        print("=== CONTENT (кратко) ===")
        print(content[:1500] + ("…" if len(content) > 1500 else ""))

        print("\n=== PRUNED DOCS (из artifact) ===")
        if not artifact:
            print("(artifact пуст)")
        else:
            for i, doc in enumerate(artifact, 1):
                meta = getattr(doc, "metadata", {}) or {}
                text = getattr(doc, "page_content", "") or ""

                print(f"\n--- DOC #{i} ---")
                print(f"meta: {meta}")
                print(f"len(text): {len(text)} chars")

                # выводим побольше, чтобы было видно, что именно осталось после pruning
                preview_len = 2000
                print(text[:preview_len] + ("…" if len(text) > preview_len else ""))

    else:
        print(result)

    print("\n>>> Проверка завершена\n")


if __name__ == "__main__":
    test_query = "Чем лучше чернить резину?"
    test_retriever_tool(test_query)