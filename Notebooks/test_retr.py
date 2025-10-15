from langchain_core.messages import HumanMessage
from retrieval_tool import retriever_tool


def test_retriever_tool(query: str):
    """Проверка работы retriever_tool (через ContextualCompressionRetriever)."""
    print(f"\n>>> Проверка инструмента retriever_tool\nЗапрос: {query}\n")

    # Вызываем инструмент напрямую
    result = retriever_tool.invoke({"query": query})

    # Если инструмент вернул результат в виде dict — красиво форматируем
    if isinstance(result, dict):
        content = result.get("content") or ""
        artifact = result.get("artifact") or []
        print("=== CONTENT ===")
        print(content[:1500] + ("…" if len(content) > 1500 else ""))
        print("\n=== ARTIFACT (метаданные / документы) ===")
        for i, doc in enumerate(artifact, 1):
            meta = doc.metadata if hasattr(doc, "metadata") else {}
            text = getattr(doc, "page_content", str(doc))[:200]
            print(f"[{i}] {meta}\n    {text}\n")
    else:
        # Fallback: если результат не словарь, просто выводим
        print(result)

    print("\n>>> Проверка завершена\n")


# === ЗАПУСК ТЕСТА ===
if __name__ == "__main__":
    test_query = "Чем лучше чернить резину?"
    test_retriever_tool(test_query)