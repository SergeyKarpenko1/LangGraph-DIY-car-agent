import hashlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.retrievers import ContextualCompressionRetriever, EnsembleRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.tools.retriever import create_retriever_tool
from langchain_chroma import Chroma
from langchain_community.document_loaders import (
    DirectoryLoader,
    UnstructuredMarkdownLoader,
)
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()

from huggingface_hub import login

login(token=os.getenv("HUGGINGFACE_HUB_TOKEN"))

embeddings = HuggingFaceEmbeddings(model_name="deepvk/USER-bge-m3")

gpt_oss_20b = init_chat_model(
    model="openai/gpt-oss-20b:free",
    model_provider="openai",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    # # прокидывание провайдер-специфичных аргументов:
    extra_body={"temperature": 0}
)

# Параметры данных/индекса
md_folder = "/Users/sergey/Desktop/Deteiling_agent/Data/cleaned"
persist_dir = "/Users/sergey/Desktop/Deteiling_agent/Data/ChromaDB"
collection_name = "VectorDB_deepvk_USER-bge-m3"

# Открываем (или создаём пустую) коллекцию Chroma
vectordb = Chroma(
    collection_name=collection_name,
    embedding_function=embeddings,      # важно: тот же эмбеддер, что использовался при создании
    persist_directory=persist_dir,
)

def _stable_id(doc: Document, idx: int) -> str:
    """Устойчивый ID по источнику и номеру чанка (sha1 от 'source::idx')."""
    src = (doc.metadata or {}).get("source") or (doc.metadata or {}).get("file") or (doc.metadata or {}).get("path") or ""
    key = f"{str(src)}::{idx:06d}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()

def _load_and_chunk_markdown(folder: str) -> list[Document]:
    """Загрузка Markdown и чанкинг."""
    loader = DirectoryLoader(
        path=folder,
        glob="**/*.md",
        loader_cls=UnstructuredMarkdownLoader,
        loader_kwargs={"mode": "single"},
    )
    docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", " "],
    )
    return splitter.split_documents(docs)

def _reconstruct_docs_from_chroma(db: Chroma) -> list[Document]:
    """Восстановить документы из уже существующей коллекции (на случай отсутствия исходников)."""
    raw = db._collection.get(include=["documents", "metadatas", "ids"])  # приватное API, допустимо для восстановления
    docs = []
    for txt, md in zip(raw.get("documents", []), raw.get("metadatas", [])):
        docs.append(Document(page_content=txt or "", metadata=md or {}))
    return docs

# 2) Если коллекция пуста — единовременно наполняем её
if vectordb._collection.count() == 0:
    recursive_chunks = _load_and_chunk_markdown(md_folder)
    ids, to_add = [], []
    for i, d in enumerate(recursive_chunks):
        md = dict(d.metadata or {})
        md.setdefault("source", md.get("file") or md.get("path") or "md")
        md["chunk"] = i
        to_add.append(Document(page_content=d.page_content, metadata=md))
        ids.append(_stable_id(d, i))
    vectordb.add_documents(to_add, ids=ids)
    # vectordb.persist()  # опционально; современные версии сохраняют автоматически
else:
    # Коллекция уже существует — ничего не добавляем
    pass

# 3) Готовим корпус для BM25 (не персистится). Предпочтительно — из исходников; если их нет — из Chroma.
if Path(md_folder).exists():
    recursive_chunks = _load_and_chunk_markdown(md_folder)
else:
    recursive_chunks = _reconstruct_docs_from_chroma(vectordb)

# 4) Ретриверы: MMR поверх Chroma + BM25, затем Ensemble
mmr = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 6, "fetch_k": 40, "lambda_mult": 0.5},
)
bm25 = BM25Retriever.from_documents(recursive_chunks)

ensemble = EnsembleRetriever(
    retrievers=[mmr, bm25],
    weights=[0.6, 0.4],
    
)

extract_prompt_fewshot = PromptTemplate.from_template(
    "You are an EXTRACTIVE span selector for a QA system.\n"
    "Given a Question and a Document, return up to 5 VERBATIM spans from the Document that directly answer the Question.\n"
    "Rules:\n"
    "- Copy text EXACTLY as in the Document (NO paraphrasing, keep punctuation & casing).\n"
    "- Each span ≤ 400 characters.\n"
    "- Put each span on a new line, prefixed with '- '.\n"
    "- If nothing is relevant, return an empty string.\n"
    "- Do NOT add explanations or headers.\n\n"

    "### EXAMPLES\n"
    "Example 1 (Positive — tar/bitumen removal)\n"
    "Question:\n"
    "Как удалить битум с кузова автомобиля?\n"
    "Document:\n"
    "Специальная автохимия для удаления битума на авто Чем оттереть битум с автомобиля? Имеются и специальные составы для удаления битумной смолы. "
    "Они имеют свои преимущества: - Безопасность — токсичные составляющие отсутствуют, которые есть в бензине и дизельном топливе, а значит, они безопасны "
    "для использования. - Дают хороший эффект — их специально разрабатывают для подобных пятен, они не оставляют лишних следов. - Защита — они имеют "
    "специальные составляющие, защищающие металлическую поверхность, она не будет деформироваться и ржаветь. - Удобны в использовании — большая часть "
    "такой продукции предлагается во флаконах. Их распылитель позволяет легко и точно наносить состав. - Экономичность — многие водители не хотят зря "
    "тратить деньги, и именно подобные средства позволяют удалить пятно битума несколькими каплями.\n"
    "Expected spans:\n"
    "- Имеются и специальные составы для удаления битумной смолы.\n"
    "- Безопасность — токсичные составляющие отсутствуют, которые есть в бензине и дизельном топливе, а значит, они безопасны для использования.\n"
    "- Дают хороший эффект — их специально разрабатывают для подобных пятен, они не оставляют лишних следов.\n"
    "- Удобны в использовании — большая часть такой продукции предлагается во флаконах. Их распылитель позволяет легко и точно наносить состав.\n"
    "- Экономичность — многие водители не хотят зря тратить деньги, и именно подобные средства позволяют удалить пятно битума несколькими каплями.\n\n"

    "Example 2 (Negative — unrelated → empty output)\n"
    "Question:\n"
    "Как убрать ржавчину с кузова автомобиля?\n"
    "Document:\n"
    "Как очистить битум с кузова автомобиля? Бороться со битумными пятнами несложно. Можно пользоваться и народными средствами либо наносить спреи...\n"
    "Expected spans:\n"
    "\n"  # пусто умышленно

    "Example 3 (Neutral — insects removal)\n"
    "Question:\n"
    "Как удалить следы насекомых с кузова?\n"
    "Document:\n"
    "Несколько правил по очистке поверхности автомобиля Если вы собираетесь отмыть авто от мошек, но сохранить поверхность в надлежащем качестве, "
    "не забывайте о таких правилах: Не удалять на сухую ... Старайтесь не наносить состав в жаркий день, когда кузов машины горячий, жидкость быстро "
    "испарится, не успев размягчить пятна.\n"
    "Expected spans:\n"
    "- Несколько правил по очистке поверхности автомобиля ... не забывайте о таких правилах: Не удалять на сухую\n"
    "- Старайтесь не наносить состав в жаркий день, когда кузов машины горячий, жидкость быстро испарится, не успев размягчить пятна.\n\n"

    "### NOW DO THE TASK\n"
    "Question:\n{question}\n\n"
    "Document:\n{context}\n"
    "Return only the spans as specified."
)

extractor = LLMChainExtractor.from_llm(llm=gpt_oss_20b, prompt=extract_prompt_fewshot)

ccr = ContextualCompressionRetriever(
    base_retriever=ensemble,
    base_compressor=extractor,
)


retriever_tool = create_retriever_tool(
    ensemble,
    name="retrieve_in_vectordb",
    description="Search and return information about car care, detailing and everything related to self-washing, cleaning.",
    response_format="content_and_artifact",
)
