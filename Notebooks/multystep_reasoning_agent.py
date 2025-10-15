import os
from typing import List, Literal, TypedDict

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.schema import Document
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from pydantic import BaseModel, Field

from Notebooks.llm import gpt_oss_20b, gpt_oss_120b
from Notebooks.retrieval_tool import retriever_tool

load_dotenv()

# Модель для генерации ответов
llm_gpt_oss_120b = gpt_oss_120b

# Модель для классификации документов и перефразирования
llm_gpt_oss_20b = gpt_oss_20b

# Инструменты 
tavily_search = TavilySearch()
tools = [retriever_tool, tavily_search]     

llm_with_tools = llm_gpt_oss_120b.bind_tools(tools)
tools_node = ToolNode(tools=tools)

class AgentState(TypedDict):
    messages: List[BaseMessage]
    documents: list[Document]
    on_topic: str
    rephrased_question: str
    proceed_to_generate: bool
    rephrase_count: int
    question: HumanMessage
    force_web: bool         

class GradeQuestion(BaseModel):
    score: Literal['Yes', 'No'] = Field(
        description='Is the question relevanse to the topic? Yes or No'
    )

class GradeDocumenr(BaseModel):
    score: Literal['Yes', 'No'] = Field(
        description='Is the document relevanse to the question? Yes or No'
    )

# Константы для единообразия
DOCS_KEY = "documents"
GO_FLAG  = "proceed_to_generate"
MAX_REPHRASES = 1


# ================== REWRITE ==================
def question_rewriter(state: AgentState):
    print(f"Entering question_rewriter with keys: {list(state.keys())}")

    # ЕДИНЫЕ ключи
    state[DOCS_KEY] = []
    state[GO_FLAG] = False
    state["on_topic"] = ""
    state["rephrase_count"] = state.get("rephrase_count", 0)
    state["rephrased_question"] = ""

    # messages как список
    msgs = state.get("messages") or []
    if not isinstance(msgs, list):
        msgs = [msgs]
    state["messages"] = msgs

    # Нормализуем/выводим question в HumanMessage
    def _to_text(x):
        if isinstance(x, dict):
            c = x.get("content", "")
            if isinstance(c, list):
                c = "\n".join(
                    (seg.get("text","") if isinstance(seg, dict) and seg.get("type") in ("text","input_text") else str(seg))
                    for seg in c
                )
            return str(c)
        return str(getattr(x, "content", x))

    q = state.get("question")
    if q is None:
        # ищем последний HumanMessage в истории; при отсутствии — берём последний текст
        hm = next((m for m in reversed(msgs) if isinstance(m, HumanMessage)), None)
        if hm is None:
            if msgs:
                hm = HumanMessage(content=_to_text(msgs[-1]))
                state["messages"].append(hm)
            else:
                raise ValueError("question_rewriter: ни 'question', ни содержимого в 'messages' не найдено.")
        state["question"] = hm
    else:
        if isinstance(q, HumanMessage):
            state["question"] = q
        elif isinstance(q, dict) or isinstance(q, str):
            state["question"] = HumanMessage(content=_to_text(q))
        else:
            state["question"] = HumanMessage(content=_to_text(q))

    if state["question"] not in state["messages"]:
        state["messages"].append(state["question"])

    # 3) Перефразирование вопроса
    if len(state["messages"]) > 1:
        current_question = state["question"].content

        normalize = lambda v: (
            "\n".join(p.get("text","") if isinstance(p, dict) and "text" in p else str(p) for p in v)
            if isinstance(v, list) else str(v)
        )

        history_text = "\n".join(
            normalize(m.content) if isinstance(m, BaseMessage)
            else normalize(m.get("content","")) if isinstance(m, dict)
            else normalize(m)
            for m in state["messages"][:-1]
        )

        template = (
            "You are a helpful assistant that rephrases the user's question to be a "
            "standalone question optimized for retrieval.\n\n"
            "Chat history (may be empty):\n{history}\n\n"
            "Question:\n{question}\n\n"
            "Return only the rephrased question."
        )
        prompt = ChatPromptTemplate.from_template(template)
        prompt_text = prompt.format(history=history_text, question=current_question)
        response = llm_gpt_oss_120b.invoke(prompt_text)
        state["rephrased_question"] = (response.content or "").strip()
        state["rephrase_count"] += 1
    else:
        state["rephrased_question"] = state["question"].content

    # HITL: подтверждение перефразаирования
    from langgraph.types import (
        interrupt,  # если импорт уже есть сверху — оставьте его там
    )
    original = state["question"].content
    rephr = state["rephrased_question"]

    payload = {
        "type": "confirm_rewrite",
        "message": (
            "Проверьте перефразированный вопрос. Вопрос должен относиться к DIY авто-детейлингу, мойке "
            "и самостоятельному уходу за автомобилем. Выберите вариант или быстро отредактируйте."
        ),
        "original_question": original,
        "rephrased_question": rephr,
        "options": ["approve_rephrased", "use_original", "edit"]
    }
    decision = interrupt(payload)

    #  Обработка решения: approve_rephrased / use_original / edit (или быстрая правка строкой)
    if isinstance(decision, str):
        dec_raw = decision.strip()
        dec = dec_raw.lower()

        if dec in ("approve", "approve_rephrased", "ok", "да"):
            # оставляем перефраз как есть
            pass
        elif dec in ("orig", "use_original", "original"):
            state["rephrased_question"] = original
        elif dec.startswith("edit:"):
            edited = dec_raw.split("edit:", 1)[1].strip()
            if edited:
                state["rephrased_question"] = edited
        else:
            # неизвестная команда — ничего не меняем
            pass

    elif isinstance(decision, dict):
        act = str(decision.get("decision", "")).lower()
        if act in ("approve", "approve_rephrased"):
            pass
        elif act in ("use_original", "orig", "original"):
            state["rephrased_question"] = original
        elif act == "edit":
            edited = (decision.get("edited_question") or "").strip()
            if edited:
                state["rephrased_question"] = edited
        else:
            # поддержим случай, когда прилетела только edited_question без decision
            edited = (decision.get("edited_question") or "").strip()
            if edited:
                state["rephrased_question"] = edited
            # прочие варианты игнорируем, оставляем текущий rephr

    return state

# ================== CLASSIFY ==================
def question_classifier(state: AgentState) -> AgentState:
    print("Entering question_classifier")

    system_message = SystemMessage(content=(
        "You are a strict binary classifier. Decide if the question is about DIY car care / detailing or about car.\n"
        "Answer EXACTLY 'Yes' or 'No'. No explanations."
    ))
    # ВАЖНО: кавычки в f-строке
    human_message = HumanMessage(content=f"User question: {state['rephrased_question']}")

    grade_prompt = ChatPromptTemplate.from_messages([system_message, human_message]).format_messages()
    # Без structured_output — парсим текстом (надёжно для любого провайдера)  [oai_citation:5‡python.langchain.com](https://python.langchain.com/docs/concepts/structured_outputs/?utm_source=chatgpt.com)
    result = llm_gpt_oss_120b.invoke(grade_prompt)
    state["on_topic"] = (result.content or "").strip().lower().startswith("y") and "Yes" or "No"
    print(f"[question_classifier] on_topic={state['on_topic']}")
    return state

# ================== TOPIC ROUTER ==================
def on_topic_router(state: AgentState) -> str:
    print("Entering on_topic_router")
    on_topic = (state.get("on_topic","") or "").strip().lower()
    if on_topic == "yes":
        print("Routing to plan")
        return "plan"
    print("Routing to off_topic_response")
    return "off_topic_response"

# ================== PLAN / TOOL CALL ==================
def plan_or_call_tool(state: AgentState) -> AgentState:
    """Сгенерировать AIMessage с вызовом инструмента.
       По умолчанию — retrieve_in_vectordb; при force_web=True — tavily_search."""
    rephrased_question = state["rephrased_question"]

    if state.get("force_web"):
        system_message = SystemMessage(
            content=(
                "You can use tools. For external knowledge on the public web, "
                "you MUST call the tool 'tavily_search' with the user's standalone question. "
                "Do not answer before the tool call."
            )
        )
        user = HumanMessage(content=f"Search the public web for: {rephrased_question}")
        # форсируем конкретный tool
        ai: AIMessage = llm_gpt_oss_120b.bind_tools(
            tools, tool_choice=tavily_search.name
        ).invoke([system_message, user])
    else:
        system_message = SystemMessage(
            content=(
                "You can use tools. For external knowledge from the in-house vector DB, "
                "you MUST call the tool 'retrieve_in_vectordb' with the user's standalone question. "
                "Do not answer before retrieval."
            )
        )
        user = HumanMessage(content=f"Retrieve context for this question: {rephrased_question}")
        ai: AIMessage = llm_with_tools.invoke([system_message, user])

    return {"messages": (state.get("messages") or []) + [system_message, user, ai]}

# ================== COLLECT CONTEXT ==================def collect_context(state: AgentState) -> AgentState:

def collect_context(state: AgentState) -> AgentState:
    messages = state.get("messages") or []
    tool_messages = [m for m in messages if isinstance(m, ToolMessage)]

    docs: List[Document] = []
    if tool_messages:
        last_tool: ToolMessage = tool_messages[-1]

        # имя инструмента (разные версии кладут либо .name, либо .tool)
        tool_name = getattr(last_tool, "name", None) or getattr(last_tool, "tool", None)

        # полезная нагрузка может быть в .artifact ИЛИ в .content
        payload = getattr(last_tool, "artifact", None)
        if payload is None:
            payload = getattr(last_tool, "content", None)
            # content может быть JSON-строкой
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    # если не JSON — используем как простой текст
                    if payload.strip():
                        docs.append(
                            Document(page_content=payload.strip(),
                                     metadata={"provider": tool_name or "tool"})
                        )
                    payload = None  # чтобы ниже не дублировать

        # 1) Если это уже список Document — просто принять
        if isinstance(payload, list) and all(isinstance(d, Document) for d in payload):
            docs = payload

        # 2) Tavily: dict с ключом "results" (и опционально "answer")
        elif tool_name == tavily_search.name:
            items = []
            if isinstance(payload, dict):
                # основной список результатов
                if "results" in payload and isinstance(payload["results"], list):
                    items = payload["results"]
                else:
                    # на всякий случай — если инструмент вернул один результат
                    items = [payload]
                # опционально добавим краткий "answer" как отдельный документ
                ans = payload.get("answer")
                if isinstance(ans, str) and ans.strip():
                    docs.append(Document(
                        page_content=ans.strip(),
                        metadata={"provider": "tavily", "type": "answer"}
                    ))
            elif isinstance(payload, list):
                items = payload
            elif payload is not None:
                items = [payload]

            for r in items:
                if not r:
                    continue
                if isinstance(r, dict):
                    title = r.get("title") or ""
                    url = r.get("url") or ""
                    text = r.get("content") or r.get("snippet") or title or url or ""
                    if text.strip():
                        docs.append(Document(
                            page_content=text.strip(),
                            metadata={"source": url, "title": title, "provider": "tavily"}
                        ))
                else:
                    docs.append(Document(
                        page_content=str(r),
                        metadata={"provider": "tavily"}
                    ))

        # 3) Иные инструменты, которые вернули список строк/словари
        elif isinstance(payload, list):
            for r in payload:
                if isinstance(r, dict):
                    text = r.get("page_content") or r.get("content") or r.get("text") or ""
                    if text.strip():
                        docs.append(Document(page_content=text.strip(), metadata=r.get("metadata") or {}))
                elif r:
                    docs.append(Document(page_content=str(r)))

    print(f"[collect_context] collected {len(docs)} documents (tool={tool_name})")
    state[DOCS_KEY] = docs
    state[GO_FLAG] = len(docs) > 0
    state["force_web"] = False
    return state

# ================== GRADE RETRIEVAL ==================
def retrieval_grader(state: AgentState) -> AgentState:
    """Фильтруем state['documents'] простым Yes/No без structured_output"""
    print("Entering retrieval_grader")
    docs = state.get(DOCS_KEY) or []
    if not docs:
        state[GO_FLAG] = False
        print("retrieval_grader: no documents; proceed_to_generate=False")
        return state

    sys = SystemMessage(content="You are a grader. Answer exactly 'Yes' or 'No'. No explanations.")
    kept: List[Document] = []
    for d in docs:
        hm = HumanMessage(content=f"User question:\n{state['rephrased_question']}\n\nRetrieved document:\n{d.page_content}")
        prompt = ChatPromptTemplate.from_messages([sys, hm]).format_messages()
        try:
            res = llm_gpt_oss_20b.invoke(prompt)
            ans = (res.content or "").strip().lower()
        except Exception as e:
            print(f"[retrieval_grader] provider error: {e!r}; fallback 'no'")
            ans = "no"
        if ans.startswith("y"):
            kept.append(d)

    state[DOCS_KEY] = kept
    state[GO_FLAG] = len(kept) > 0
    print(f"retrieval_grader: kept={len(kept)} {GO_FLAG}={state[GO_FLAG]}")
    return state

# ================== PROCEED ROUTER ==================
def proceed_router(state: AgentState) -> str:
    """
    Если есть релевантные документы — генерируем ответ.
    Если достигнут лимит перефразов — cannot_answer.
    Иначе — refine_question.
    """
    print("Entering proceed_router")
    if state.get(GO_FLAG, False):
        print("Routing to generate_answer")
        return "generate_answer"
    if state.get("rephrase_count", 0) >= MAX_REPHRASES:
        print("Routing to cannot_answer (max rephrases)")
        return "cannot_answer"
    print("Routing to refine_question")
    return "refine_question"

# ================== REFINE QUESTION ==================
def refine_question(state: AgentState) -> AgentState:
    """Чуть перефразируем вопрос; НЕ вызываем ретривер в этом узле"""
    print("Entering refine_question")
    rc = state.get("rephrase_count", 0)
    if rc >= MAX_REPHRASES:
        print("Maximum rephrase attempts reached")
        return state

    original = state["rephrased_question"]
    sys = SystemMessage(content="You slightly rephrase the user's question to improve retrieval. Return only the refined question.")
    hm = HumanMessage(content=f"Original question: {original}\n\nProvide a slightly refined question.")
    prompt = ChatPromptTemplate.from_messages([sys, hm]).format_messages()
    response = llm_gpt_oss_120b.invoke(prompt)
    refined = (response.content or "").strip()

    print(f"refine_question: {original}  ->  {refined}")
    state["rephrased_question"] = refined
    state["rephrase_count"] = rc + 1
    return state

# ================== GENERATE ANSWER ==================
RAG_TEMPLATE = """Answer the question based on the following context and the chat history.
Especially take the latest question into consideration.

Chat history:
{history}

Context:
{context}

Question:
{question}
"""
rag_prompt = ChatPromptTemplate.from_template(RAG_TEMPLATE)
rag_chain = rag_prompt | llm_gpt_oss_120b

def _history_to_text(msgs: List[BaseMessage]) -> str:
    lines: List[str] = []
    for m in msgs or []:
        if isinstance(m, HumanMessage):
            role = "User"
        elif isinstance(m, AIMessage):
            role = "Assistant"
        elif isinstance(m, SystemMessage):
            role = "System"
        else:
            role = "Message"
        if getattr(m, "content", None):
            lines.append(f"{role}: {m.content}")
    return "\n".join(lines)

def generate_answer(state: AgentState) -> AgentState:
    print("Entering generate_answer")
    if not state.get("messages"):
        raise ValueError("State must include 'messages' before generating an answer.")

    docs = state.get(DOCS_KEY) or []
    context_text = "\n\n---\n\n".join(d.page_content for d in docs[:6])
    history_text = _history_to_text(state["messages"])
    q = state["rephrased_question"]

    response = rag_chain.invoke({"question": q, "history": history_text, "context": context_text})
    generation = (response.content or "").strip()
    print(f"generate_answer: {generation[:160]}...")
    state["messages"] = (state.get("messages") or []) + [AIMessage(content=generation)]
    return state

# ================== FALLBACKS ==================def cannot_answer(state: AgentState) -> AgentState | Command:
def cannot_answer(state: AgentState) -> AgentState | Command:
    payload = {
        "type": "web_fallback",
        "message": (
            "Мы не нашли информации по вашему запросу в нашей базе данных. Выполнить поиск в интернете (Tavily) и дополнить ответ?"
        ),
        "options": ["yes", "no"]
    }
    decision = interrupt(payload)

    dec = (decision if isinstance(decision, str)
           else str(decision.get("decision",""))).strip().lower()

    if dec in ("yes", "y", "да", "ok"):
        # выставляем флаг и отправляемся обратно в plan → tools
        return Command(update={"force_web": True}, goto="plan")
    else:
        msgs = state.get("messages") or []
        msgs.append(AIMessage(content="Извините, не удалось найти достаточный контекст для ответа."))
        state["messages"] = msgs
        return state

def off_topic_response(state: AgentState) -> AgentState:
    print("Entering off_topic_response")
    state["messages"] = state.get("messages") or []
    state["messages"].append(AIMessage(content="Вопрос выходит за рамки DIY-ухода и детейлинга."))
    return state

graph = StateGraph(AgentState)

# Узлы
graph.add_node("rewrite", question_rewriter)
graph.add_node("classify", question_classifier)
graph.add_node("plan", plan_or_call_tool)
graph.add_node("tools", ToolNode(tools=tools))  # выполнение инструмента
graph.add_node("collect", collect_context)
graph.add_node("grade", retrieval_grader)
graph.add_node("refine_question", refine_question)
graph.add_node("generate_answer", generate_answer)
graph.add_node("cannot_answer", cannot_answer)
graph.add_node("off_topic_response", off_topic_response)

# Рёбра
graph.set_entry_point("rewrite")
graph.add_edge("rewrite", "classify")
graph.add_conditional_edges(
    "classify",
    on_topic_router,
    {"plan": "plan", "off_topic_response": "off_topic_response"},
)
graph.add_edge("plan", "tools")
graph.add_edge("tools", "collect")
graph.add_edge("collect", "grade")
graph.add_conditional_edges(
    "grade",
    proceed_router,
    {"generate_answer": "generate_answer", "refine_question": "refine_question", "cannot_answer": "cannot_answer"},
)
graph.add_edge("refine_question", "plan")
graph.add_edge("generate_answer", END)
graph.add_edge("off_topic_response", END)
graph.add_edge("cannot_answer", END)


graph = graph.compile()