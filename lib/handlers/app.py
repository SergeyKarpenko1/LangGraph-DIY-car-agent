# app.py
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.documents import Document
from langgraph.types import Command

from lib.workers.multystep_reasoning_agent_stream import graph as platform_graph
from lib.workers.multystep_reasoning_agent_stream import get_fastapi_graph
from lib.utils.constant import DOCS_KEY, GO_FLAG



# -------------------- Lifespan -------------------- #

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("LIFESPAN: запуск приложения, инициализируем ресурсы")

    # Для FastAPI нужен checkpointer (interrupt()/resume(), get_state)
    app.state.graph = get_fastapi_graph()

    # Быстрая проверка “готовности” графа (особенно после ваших правок compile/checkpointer)
    g = app.state.graph
    app.state.graph_ok = bool(getattr(g, "checkpointer", None)) and hasattr(g, "astream") and hasattr(g, "get_state")
    print(f"LIFESPAN: graph_ok={app.state.graph_ok}")

    yield

    print("LIFESPAN: приложение остановлено")


app = FastAPI(title="LangGraph API", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------- Middleware -------------------- #

@app.middleware("http")
async def simple_logger(request: Request, call_next):
    print(f"[Middleware] Incoming request: {request.method} {request.url.path}")
    response = await call_next(request)
    print(f"[Middleware] Response status: {response.status_code}")
    return response

# -------------------- Models --------------------

class ChatStartRequest(BaseModel):
    user_message: str = Field(..., min_length=1)
    thread_id: Optional[str] = None


class ChatResumeRequest(BaseModel):
    thread_id: str = Field(..., min_length=1)
    resume: Any  # строка/объект/и т.д. (JSON-serializable)


# -------------------- SSE helpers --------------------

def _to_jsonable(obj: Any) -> Any:
    if isinstance(obj, BaseMessage):
        return _msg_to_dict(obj)
    if isinstance(obj, Document):
        return {"page_content": obj.page_content, "metadata": obj.metadata}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, set):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _msg_to_dict_sse(m: BaseMessage) -> Dict[str, Any]:
    """
    Компактная сериализация для SSE (updates), чтобы события приходили инкрементально,
    а не буферизовались из-за больших документов/выводов инструментов.
    """
    base: Dict[str, Any] = {"type": m.type, "content": getattr(m, "content", None)}

    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls is not None:
        base["tool_calls"] = tool_calls

    if isinstance(m, ToolMessage):
        base["tool_call_id"] = getattr(m, "tool_call_id", None)
        tool_name = getattr(m, "name", None) or getattr(m, "tool", None)
        base["name"] = tool_name

        artifact = getattr(m, "artifact", None)
        if isinstance(artifact, list) and all(isinstance(d, Document) for d in artifact):
            base["artifact_count"] = len(artifact)
            base["artifact"] = [
                {
                    "page_content": (d.page_content or "")[:260],
                    "metadata": d.metadata or {},
                }
                for d in artifact[:10]
            ]

        # Tavily: сохраняем только ссылки (и короткий сниппет), а не весь JSON
        content = getattr(m, "content", None)
        if tool_name == "tavily_search":
            parsed = None
            if isinstance(content, str):
                try:
                    parsed = json.loads(content)
                except Exception:
                    parsed = None
            elif isinstance(content, dict):
                parsed = content

            if isinstance(parsed, dict):
                results = parsed.get("results")
                if isinstance(results, list):
                    slim_results: List[Dict[str, Any]] = []
                    for r in results[:10]:
                        if not isinstance(r, dict):
                            continue
                        url = r.get("url")
                        if not url:
                            continue
                        slim_results.append(
                            {
                                "url": url,
                                "title": r.get("title") or r.get("source") or url,
                                "content": (r.get("content") or r.get("snippet") or "")[:260],
                            }
                        )
                    base["tavily_results"] = slim_results
                    base["artifact_count"] = base.get("artifact_count") or len(slim_results)

            base["content"] = None
        else:
            if isinstance(base.get("content"), str) and base["content"] and len(base["content"]) > 800:
                base["content"] = base["content"][:799] + "…"

    return base


def _to_jsonable_sse(obj: Any) -> Any:
    if isinstance(obj, BaseMessage):
        return _msg_to_dict_sse(obj)
    if isinstance(obj, Document):
        return {"page_content": (obj.page_content or "")[:260], "metadata": obj.metadata}
    if isinstance(obj, dict):
        return {str(k): _to_jsonable_sse(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable_sse(x) for x in obj]
    if isinstance(obj, set):
        return [_to_jsonable_sse(x) for x in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def _sse(event: str | None, data: Any, *, slim: bool = False) -> str:
    payload = json.dumps((_to_jsonable_sse(data) if slim else _to_jsonable(data)), ensure_ascii=False, default=str)
    out = ""
    if event:
        out += f"event: {event}\n"
    out += f"data: {payload}\n\n"
    return out


def _msg_to_dict(m: BaseMessage) -> Dict[str, Any]:
    base = {"type": m.type, "content": getattr(m, "content", None)}
    # tool_calls иногда полезны фронту
    tool_calls = getattr(m, "tool_calls", None)
    if tool_calls is not None:
        base["tool_calls"] = tool_calls
    # ToolMessage: tool_call_id и т.п.
    if isinstance(m, ToolMessage):
        base["tool_call_id"] = getattr(m, "tool_call_id", None)
        base["name"] = getattr(m, "name", None) or getattr(m, "tool", None)
        artifact = getattr(m, "artifact", None)
        if artifact is not None:
            # обычно это список Document (response_format="content_and_artifact")
            if isinstance(artifact, list) and all(isinstance(d, Document) for d in artifact):
                base["artifact"] = [
                    {
                        "page_content": (d.page_content or "")[:800],
                        "metadata": d.metadata or {},
                    }
                    for d in artifact[:12]
                ]
                base["artifact_count"] = len(artifact)
            else:
                base["artifact"] = artifact
    return base


def _serialize_interrupt(obj: Any) -> Any:
    """
    В stream_mode="updates" interrupt приходит как:
      {"__interrupt__": (Interrupt(...), ...)}
    Объекты Interrupt не всегда JSON-сериализуемы — приводим к dict.
    """
    if isinstance(obj, (list, tuple)):
        return [_serialize_interrupt(x) for x in obj]
    # Interrupt(value=..., resumable=..., ns=[...])
    value = getattr(obj, "value", None)
    resumable = getattr(obj, "resumable", None)
    ns = getattr(obj, "ns", None)
    if value is not None or resumable is not None or ns is not None:
        return {"value": value, "resumable": resumable, "ns": ns}
    return obj


def _extract_last_ai_text(state: Dict[str, Any]) -> Optional[str]:
    msgs = state.get("messages") or []
    # messages могут быть объектами BaseMessage
    for m in reversed(msgs):
        if isinstance(m, AIMessage) and getattr(m, "content", None):
            return m.content
    return None


def _load_or_init_state(thread_id: str, user_message: str) -> Dict[str, Any]:
    """
    Если thread существует — подгружаем сохранённый state и добавляем новый HumanMessage.
    Если нет — создаём минимальный state.
    """
    cfg = {"configurable": {"thread_id": thread_id}}

    g = getattr(app.state, "graph", platform_graph)
    prev: Optional[Dict[str, Any]] = None
    try:
        prev = g.get_state(cfg).values  # требует checkpointer
    except Exception:
        prev = None

    hm = HumanMessage(content=user_message)

    if isinstance(prev, dict) and prev.get("messages"):
        msgs = list(prev["messages"]) + [hm]
        # Ваш rewrite узел сам нормализует ключи, но question лучше явно дать
        merged = dict(prev)
        merged["messages"] = msgs
        merged["question"] = hm
        # сбрасываем per-turn поля, чтобы новый вопрос не наследовал старые значения
        merged[DOCS_KEY] = []
        merged[GO_FLAG] = False
        merged["on_topic"] = ""
        merged["rephrased_question"] = user_message
        merged["force_web"] = False
        merged["rephrase_count"] = 0
        merged["did_rewrite"] = False
        return merged

    return {
        "messages": [hm],
        "question": hm,
        DOCS_KEY: [],
        GO_FLAG: False,
        "on_topic": "",
        "rephrased_question": user_message,
        "force_web": False,
        "rephrase_count": 0,
        "did_rewrite": False,
    }


# -------------------- Streaming core --------------------

async def _stream_graph_updates(input_obj: Any, thread_id: str):
    cfg = {"configurable": {"thread_id": thread_id}}
    yield _sse("meta", {"thread_id": thread_id}, slim=True)

    g = getattr(app.state, "graph", platform_graph)
    seen_nodes: set[str] = set()
    try:
        # ВАЖНО: список режимов => на выходе (mode, chunk)
        async for mode, chunk in g.astream(input_obj, cfg, stream_mode=["messages", "tasks", "updates"]):
            if mode == "messages":
                # chunk == (message_chunk, metadata)
                msg_chunk, metadata = chunk

                # Рекомендую делать фильтр мягче, чем == ["final_answer"]
                tags = metadata.get("tags", []) or []
                if ("final_answer" in tags) and getattr(msg_chunk, "content", None):
                    yield _sse("token", {"delta": msg_chunk.content}, slim=True)
                continue

            if mode == "tasks":
                # В режиме tasks прилетает событие старта: {"id","name","input",...}
                # и событие окончания: {"id","name","result",...}
                if isinstance(chunk, dict) and isinstance(chunk.get("name"), str) and "input" in chunk:
                    node = chunk["name"]
                    if node not in seen_nodes:
                        seen_nodes.add(node)
                        yield _sse("step", {"node": node}, slim=True)
                continue

            if mode == "updates":
                # interrupt прилетает в __interrupt__
                if isinstance(chunk, dict) and "__interrupt__" in chunk:
                    payload = _serialize_interrupt(chunk["__interrupt__"])
                    yield _sse("interrupt", {"thread_id": thread_id, "payload": payload}, slim=True)
                    return

                # при желании оставьте update-ивенты (для дебага/статусов)
                yield _sse("update", chunk, slim=True)

        # финал (как у вас было)
        final_state = None
        try:
            final_state = g.get_state(cfg).values
        except Exception:
            final_state = None

        answer = _extract_last_ai_text(final_state or {})
        yield _sse("done", {"thread_id": thread_id, "answer": answer}, slim=True)

    except asyncio.CancelledError:
        return
    except Exception as e:
        yield _sse("error", {"thread_id": thread_id, "message": str(e)}, slim=True)


# -------------------- Healthchecks -------------------- #

@app.get("/health")
async def healthcheck():
    return {"status": "ok"}

@app.get("/ready")
async def readiness():
    # “готовность” — граф должен быть compiled (astream/get_state)
    ok = bool(getattr(app.state, "graph_ok", False))
    if not ok:
        raise HTTPException(status_code=503, detail={"status": "not_ready"})
    return {"status": "ok"}

@app.get("/chat/state/{thread_id}")
async def chat_state(thread_id: str):
    """Вернуть сохранённый state по thread_id (если есть checkpointer)."""
    cfg = {"configurable": {"thread_id": thread_id}}
    g = getattr(app.state, "graph", platform_graph)
    try:
        state = g.get_state(cfg).values
    except Exception as e:
        raise HTTPException(status_code=404, detail={"thread_id": thread_id, "message": str(e)})
    return {"thread_id": thread_id, "state": _to_jsonable(state)}


# -------------------- Endpoints --------------------

@app.post("/chat/stream")
async def chat_stream(req: ChatStartRequest):
    thread_id = req.thread_id or str(uuid.uuid4())
    state = _load_or_init_state(thread_id, req.user_message)

    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        # полезно за Nginx/прокси: отключает буферизацию, иначе “стрим” будет приходить пачками
        "X-Accel-Buffering": "no",
    }

    return StreamingResponse(
        _stream_graph_updates(state, thread_id),
        media_type="text/event-stream",
        headers=headers,
    )


@app.post("/chat/resume/stream")
async def chat_resume_stream(req: ChatResumeRequest):
    """
    Продолжение после interrupt():
    граф нужно вызвать повторно с тем же thread_id и Command(resume=...).
     [oai_citation:3‡docs.langchain.com](https://docs.langchain.com/oss/python/langgraph/interrupts)
    """
    headers = {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }

    cmd = Command(resume=req.resume)

    return StreamingResponse(
        _stream_graph_updates(cmd, req.thread_id),
        media_type="text/event-stream",
        headers=headers,
    )

# uvicorn lib.handlers.app:app --reload --port 8000
