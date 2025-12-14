from __future__ import annotations

import base64
import json
import logging
import os
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests
import streamlit as st

logging.basicConfig(level=logging.INFO)

NUMBER_OF_MESSAGES_TO_DISPLAY = 30


PAGE_ICON: str = "ImageGraph.png" if os.path.exists("ImageGraph.png") else "🚗"

st.set_page_config(
    page_title="Detailing Agent — Streamly UI",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": "https://github.com/",
        "About": """
            ## Detailing Agent (Streamly-style UI)
            Чат-UI в стиле Streamly, подключается к локальному backend (`FastAPI`) по SSE.
        """,
    },
)

st.title("Detailing Car Agent")
st.caption(
    "DIY-агент по уходу за авто: ищет контекст в локальной базе знаний и при необходимости делает веб-поиск. "
    "Иногда останавливается и просит подтверждение (human-in-the-loop)."
)


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: Any


def img_to_base64(image_path: str) -> Optional[str]:
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception as e:
        logging.info("img_to_base64 failed: %s", e)
        return None


def _get_query_params() -> Dict[str, List[str]]:
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        return {k: ([v] if isinstance(v, str) else list(v)) for k, v in dict(qp).items()}
    except Exception:
        return st.experimental_get_query_params()


def _set_query_params(**kwargs: str) -> None:
    try:
        qp = st.query_params  # type: ignore[attr-defined]
        for k, v in kwargs.items():
            qp[k] = v
    except Exception:
        st.experimental_set_query_params(**kwargs)


def _iter_sse_lines(resp: requests.Response) -> Iterator[str]:
    # Важно: маленький chunk_size снижает буферизацию и даёт "живые" апдейты по мере прихода.
    for raw in resp.iter_lines(chunk_size=1, decode_unicode=True):
        if raw is None:
            continue
        yield raw


def _iter_sse_events(resp: requests.Response) -> Iterator[Tuple[str, str]]:
    event: Optional[str] = None
    data_lines: List[str] = []

    for line in _iter_sse_lines(resp):
        if not line:
            if data_lines:
                yield (event or "message", "\n".join(data_lines))
            event = None
            data_lines = []
            continue

        if line.startswith(":"):
            continue

        if line.startswith("event:"):
            event = line[len("event:") :].strip()
            continue

        if line.startswith("data:"):
            data_lines.append(line[len("data:") :].lstrip())
            continue

    if data_lines:
        yield (event or "message", "\n".join(data_lines))


def _extract_interrupt_value(payload: Any) -> Any:
    if isinstance(payload, dict) and "payload" in payload:
        payload = payload["payload"]

    if isinstance(payload, list) and payload:
        first = payload[0]
        if isinstance(first, dict) and "value" in first:
            return first["value"]
        return first
    if isinstance(payload, dict) and "value" in payload:
        return payload["value"]
    return payload


def _api_url(base: str, path: str) -> str:
    return base.rstrip("/") + path


def _raise_for_status_verbose(r: requests.Response) -> None:
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        body = (r.text or "").strip()
        if len(body) > 1500:
            body = body[:1500] + "…"
        raise requests.HTTPError(f"{e}; response_body={body}") from e


def api_health(base: str) -> Tuple[bool, str]:
    try:
        r = requests.get(_api_url(base, "/ready"), timeout=2)
        if r.status_code == 200:
            return True, "ok"
        return False, f"{r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)


def api_get_state(base: str, thread_id: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        r = requests.get(_api_url(base, f"/chat/state/{thread_id}"), timeout=5)
        if r.status_code == 404:
            return None, None
        _raise_for_status_verbose(r)
        return r.json(), None
    except Exception as e:
        return None, str(e)


def api_stream_start(base: str, thread_id: str, user_message: str) -> Iterator[SseEvent]:
    r = requests.post(
        _api_url(base, "/chat/stream"),
        json={"thread_id": thread_id, "user_message": user_message},
        headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        stream=True,
        timeout=300,
    )
    r.encoding = "utf-8"
    _raise_for_status_verbose(r)
    for event, data_text in _iter_sse_events(r):
        try:
            data = json.loads(data_text)
        except Exception:
            data = data_text
        yield SseEvent(event=event, data=data)


def api_stream_resume(base: str, thread_id: str, resume: Any) -> Iterator[SseEvent]:
    r = requests.post(
        _api_url(base, "/chat/resume/stream"),
        json={"thread_id": thread_id, "resume": resume},
        headers={"Accept": "text/event-stream", "Cache-Control": "no-cache"},
        stream=True,
        timeout=300,
    )
    r.encoding = "utf-8"
    _raise_for_status_verbose(r)
    for event, data_text in _iter_sse_events(r):
        try:
            data = json.loads(data_text)
        except Exception:
            data = data_text
        yield SseEvent(event=event, data=data)


def _init_state() -> None:
    qp = _get_query_params()
    qp_thread_id = (qp.get("thread_id") or [None])[0]
    qp_backend = (qp.get("backend") or [None])[0]

    st.session_state.setdefault("backend_url", qp_backend or "http://127.0.0.1:8000")
    st.session_state.setdefault("thread_id", qp_thread_id or "")
    st.session_state.setdefault("chat", [])  # list[dict]
    st.session_state.setdefault("internals", [])  # list[dict]
    st.session_state.setdefault("pending_interrupt", None)
    st.session_state.setdefault("pending_interrupt_raw", None)
    st.session_state.setdefault("pending_interrupt_thread_id", "")
    st.session_state.setdefault("autoload_attempted", False)
    st.session_state.setdefault("recent_threads", [])  # list[str]
    st.session_state.setdefault("health_ok", False)
    st.session_state.setdefault("health_status", "")
    st.session_state.setdefault("last_state_error", "")
    st.session_state.setdefault("pending_interrupt_chat_index", None)
    st.session_state.setdefault("resume_request", None)
    st.session_state.setdefault("resume_thread_id", "")
    st.session_state.setdefault("is_streaming", False)
    st.session_state.setdefault("show_actions", True)
    st.session_state.setdefault("show_agentic_workflow", False)
    st.session_state.setdefault("message_input", "")
    st.session_state.setdefault("pending_user_message", None)
    st.session_state.setdefault("clear_message_input", False)
    st.session_state.setdefault("suggested_questions", [
        "Как правильно мыть кузов без автомойки?",
        "Чем очистить салон от пятен кофе?",
        "Как обезжирить кузов перед нанесением керамики?"
    ])


def _remember_thread(thread_id: str) -> None:
    tid = (thread_id or "").strip()
    if not tid:
        return
    recent: List[str] = list(st.session_state.get("recent_threads") or [])
    if tid in recent:
        recent.remove(tid)
    recent.insert(0, tid)
    st.session_state.recent_threads = recent[:20]


def _refresh_health() -> None:
    ok, status = api_health(st.session_state.backend_url)
    st.session_state.health_ok = ok
    st.session_state.health_status = status


def _append_internal(event: str, data: Any) -> None:
    st.session_state.internals.append({"event": event, "data": data})


def _append_chat(role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    st.session_state.chat.append({"role": role, "content": content, "meta": meta or {}})


def _ensure_assistant_meta(chat_index: int) -> Dict[str, Any]:
    meta = st.session_state.chat[chat_index].setdefault("meta", {})
    meta.setdefault("actions", [])
    meta.setdefault("_seen_nodes", set())
    meta.setdefault("_seen_tool_calls", set())
    meta.setdefault("_seen_tool_results", set())
    meta.setdefault("collected_docs", None)
    meta.setdefault("relevant_docs", None)
    meta.setdefault("tavily_results", None)
    meta.setdefault("vectordb_count", None)
    return meta


def _short_json(x: Any, max_len: int = 220) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = str(x)
    s = " ".join(s.split())
    if len(s) > max_len:
        return s[: max_len - 1] + "…"
    return s


def _as_list(x: Any) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _doc_source(doc: Any) -> str:
    if not isinstance(doc, dict):
        return ""
    meta = doc.get("metadata") or {}
    if isinstance(meta, dict):
        src = meta.get("source") or meta.get("url") or meta.get("link") or ""
        return str(src) if src is not None else ""
    return ""


def _doc_title(doc: Any) -> str:
    if not isinstance(doc, dict):
        return ""
    meta = doc.get("metadata") or {}
    if isinstance(meta, dict):
        title = meta.get("title") or meta.get("name") or ""
        if title:
            return str(title)
    return ""


def _doc_snippet(doc: Any, n: int = 140) -> str:
    if not isinstance(doc, dict):
        return ""
    txt = doc.get("page_content") or ""
    txt = str(txt)
    txt = " ".join(txt.split())
    if len(txt) > n:
        return txt[: n - 1] + "…"
    return txt


def _fingerprint_doc(doc: Any) -> str:
    src = _doc_source(doc)
    snip = _doc_snippet(doc, n=120)
    return f"{src}::{snip}"


def _short_local_source(src: str) -> str:
    s = (src or "").strip()
    if not s:
        return ""
    base = os.path.basename(s)
    # strip known suffixes
    for suf in ("_cleaned", "-cleaned"):
        if base.endswith(suf):
            base = base[: -len(suf)]
    # strip extension(s)
    for ext in (".md", ".markdown", ".txt", ".pdf", ".html", ".htm"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
            break
    return base or s


def _try_parse_json(x: Any) -> Optional[Any]:
    if isinstance(x, (dict, list)):
        return x
    if not isinstance(x, str):
        return None
    s = x.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        return None


def _extract_tavily_results(tool_content: Any) -> List[Dict[str, Any]]:
    parsed = _try_parse_json(tool_content)
    if not isinstance(parsed, dict):
        return []
    results = parsed.get("results")
    if not isinstance(results, list):
        return []
    out: List[Dict[str, Any]] = []
    for r in results:
        if not isinstance(r, dict):
            continue
        url = r.get("url")
        if not url:
            continue
        out.append(
            {
                "url": str(url),
                "title": str(r.get("title") or r.get("source") or url),
                "content": r.get("content") or r.get("snippet") or "",
            }
        )
    return out


def _extract_actions_from_update(update: Any, meta: Dict[str, Any]) -> List[str]:
    actions: List[str] = []
    if not isinstance(update, dict):
        return actions

    seen_nodes = meta.get("_seen_nodes")
    if not isinstance(seen_nodes, set):
        seen_nodes = set(seen_nodes or [])
        meta["_seen_nodes"] = seen_nodes

    seen_tool_calls = meta.get("_seen_tool_calls")
    if not isinstance(seen_tool_calls, set):
        seen_tool_calls = set(seen_tool_calls or [])
        meta["_seen_tool_calls"] = seen_tool_calls

    seen_tool_results = meta.get("_seen_tool_results")
    if not isinstance(seen_tool_results, set):
        seen_tool_results = set(seen_tool_results or [])
        meta["_seen_tool_results"] = seen_tool_results

    for node, payload in update.items():
        if isinstance(node, str) and node not in seen_nodes:
            seen_nodes.add(node)
            node_label = {
                "plan": "Планирование",
                "tools": "Вызов инструмента",
                "collect": "Сбор контекста",
                "grade": "Оценка релевантности документов",
                "refine_question": "Уточняю запрос",
                "cannot_answer": "Недостаточно контекста → предлагаю web-поиск",
                "off_topic_response": "Вопрос не по теме",
                "generate_answer": "Генерирую ответ…",
            }.get(node)
            if node_label:
                actions.append(node_label)

        if not isinstance(payload, dict):
            continue

        docs = payload.get("documents")
        if isinstance(docs, list) and node == "collect":
            meta["collected_docs"] = docs
            actions.append(f"Контекст: найдено {len(docs)}")
        if isinstance(docs, list) and node == "grade":
            meta["relevant_docs"] = docs
            collected = meta.get("collected_docs")
            total = len(collected) if isinstance(collected, list) else None
            suffix = f"/{total}" if isinstance(total, int) else ""
            actions.append(f"Отбор: релевантных {len(docs)}{suffix}")

        msgs = payload.get("messages")
        if not isinstance(msgs, list):
            continue

        for m in msgs:
            if not isinstance(m, dict):
                continue
            m_type = (m.get("type") or "").lower()

            if m_type == "ai":
                tool_calls = m.get("tool_calls") or []
                if isinstance(tool_calls, list):
                    for tc in tool_calls:
                        if not isinstance(tc, dict):
                            continue
                        tc_id = tc.get("id") or tc.get("tool_call_id")
                        name = tc.get("name")
                        args = tc.get("args")
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            name = name or fn.get("name")
                            args = args or fn.get("arguments")
                        key = tc_id or f"{name}:{_short_json(args)}"
                        if key in seen_tool_calls:
                            continue
                        seen_tool_calls.add(key)
                        if name == "retrieve_in_vectordb":
                            actions.append("Поиск в базе знаний")
                        elif name == "tavily_search":
                            actions.append("Поиск в интернете")

            if m_type == "tool":
                tool_name = m.get("name") or m.get("tool") or "tool"
                count = m.get("artifact_count")
                tcid = m.get("tool_call_id") or ""
                if isinstance(count, int):
                    key = f"{tool_name}:{count}:{tcid}"
                    if key in seen_tool_results:
                        continue
                    seen_tool_results.add(key)
                    if tool_name == "retrieve_in_vectordb":
                        meta["vectordb_count"] = count
                        actions.append(f"База знаний: найдено {count}")
                    elif tool_name == "tavily_search":
                        actions.append(f"Web: найдено {count}")
                    else:
                        actions.append(f"`{tool_name}`: {count}")
                else:
                    # Не печатаем большие payload'ы tool output (особенно Tavily) — вместо этого извлекаем ссылки.
                    if tool_name == "tavily_search":
                        tavily_results = m.get("tavily_results")
                        results = tavily_results if isinstance(tavily_results, list) else _extract_tavily_results(m.get("content"))
                        if results:
                            meta["tavily_results"] = results
                            key = f"{tool_name}:results:{len(results)}:{tcid}"
                            if key not in seen_tool_results:
                                seen_tool_results.add(key)
                                actions.append(f"Web: найдено {len(results)} результатов")

    return actions


def _render_actions_md(actions: List[str]) -> str:
    if not actions:
        return ""
    return "\n".join([f"- {a}" for a in actions])


def _render_status_md(meta: Dict[str, Any]) -> str:
    lines: List[str] = []

    actions = meta.get("actions") or []
    if isinstance(actions, list) and actions:
        # показываем только последние N, чтобы не раздувать
        tail = actions[-12:]
        lines.extend([f"- {a}" for a in tail])

    collected = meta.get("collected_docs")
    relevant = meta.get("relevant_docs")

    if isinstance(relevant, list) and relevant:
        relevant_by_fp = {_fingerprint_doc(d) for d in relevant}
        collected_by_fp = {_fingerprint_doc(d) for d in collected} if isinstance(collected, list) else set()
        rejected_count = len(collected_by_fp - relevant_by_fp) if collected_by_fp else 0

        lines.append("")
        lines.append("**Источники (прошли отбор)**")
        for d in relevant[:8]:
            src = _doc_source(d)
            title = _doc_title(d)
            snippet = _doc_snippet(d)
            label = title or (_short_local_source(src) if src else snippet)
            if src.startswith("http://") or src.startswith("https://"):
                lines.append(f"- [{label}]({src})")
            elif src:
                lines.append(f"- `{_short_local_source(src)}` — {snippet}")
            else:
                lines.append(f"- {snippet}")

        if rejected_count:
            lines.append(f"- Не прошло отбор: {rejected_count}")

    tavily = meta.get("tavily_results")
    if isinstance(tavily, list) and tavily:
        selected_urls: set[str] = set()
        if isinstance(relevant, list):
            selected_urls = {_doc_source(d) for d in relevant if _doc_source(d).startswith(("http://", "https://"))}

        lines.append("")
        lines.append("**Web ссылки**")
        for r in tavily[:8]:
            if not isinstance(r, dict):
                continue
            url = str(r.get("url") or "")
            if not url:
                continue
            title = str(r.get("title") or url)
            mark = " (выбрано)" if url in selected_urls else ""
            lines.append(f"- [{title}]({url}){mark}")

    return "\n".join(lines).strip()


def _load_chat_from_state(state_payload: Dict[str, Any]) -> None:
    state = (state_payload or {}).get("state") or {}
    msgs = state.get("messages") or []
    chat: List[Dict[str, Any]] = []
    internals: List[Dict[str, Any]] = []

    for m in msgs:
        if not isinstance(m, dict):
            internals.append({"event": "state_message", "data": m})
            continue
        m_type = (m.get("type") or "").lower()
        content = m.get("content")
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)

        if m_type in ("human", "user"):
            chat.append({"role": "user", "content": text, "meta": {"type": m_type}})
        elif m_type in ("ai", "assistant"):
            chat.append({"role": "assistant", "content": text, "meta": {"type": m_type}})
        else:
            internals.append({"event": f"message:{m_type or 'unknown'}", "data": m})

    st.session_state.chat = chat
    st.session_state.internals = internals


def _run_stream(events: Iterable[SseEvent], chat_index: int) -> None:
    meta = _ensure_assistant_meta(chat_index)
    actions_placeholder = st.empty()
    assistant_placeholder = st.empty()
    acc = ""
    st.session_state.is_streaming = True

    try:
        for ev in events:
            _append_internal(ev.event, ev.data)

            if ev.event == "meta" and isinstance(ev.data, dict):
                tid = ev.data.get("thread_id")
                if isinstance(tid, str) and tid:
                    st.session_state.thread_id = tid
                    _remember_thread(tid)
                    _set_query_params(thread_id=tid)
                continue

            if ev.event == "step" and isinstance(ev.data, dict):
                node = ev.data.get("node")
                if isinstance(node, str) and node:
                    actions = meta.get("actions") or []
                    new_actions = _extract_actions_from_update({node: {}}, meta)
                    if new_actions:
                        actions.extend(new_actions)
                        meta["actions"] = actions
                        if st.session_state.get("show_actions"):
                            actions_placeholder.markdown(_render_status_md(meta))
                continue

            if ev.event in ("update", "step") and st.session_state.get("show_actions"):
                actions = meta.get("actions") or []
                new_actions = _extract_actions_from_update(ev.data, meta)
                if new_actions:
                    actions.extend(new_actions)
                    meta["actions"] = actions
                    actions_placeholder.markdown(_render_status_md(meta))
                continue

            if ev.event == "token" and isinstance(ev.data, dict):
                delta = ev.data.get("delta") or ""
                if delta:
                    acc += str(delta)
                    st.session_state.chat[chat_index]["content"] = acc
                    assistant_placeholder.markdown(acc)
                continue

            if ev.event == "interrupt":
                st.session_state.pending_interrupt_raw = ev.data
                st.session_state.pending_interrupt = _extract_interrupt_value(ev.data)
                if isinstance(ev.data, dict):
                    tid = ev.data.get("thread_id")
                    if isinstance(tid, str) and tid.strip():
                        st.session_state.pending_interrupt_thread_id = tid.strip()
                st.session_state.pending_interrupt_chat_index = chat_index

                intr_val = st.session_state.pending_interrupt
                msg = (
                    intr_val.get("message")
                    if isinstance(intr_val, dict) and intr_val.get("message")
                    else "Необходимо уточнение от человека."
                )
                # Показываем прогресс/причину остановки сразу, не в конце.
                if isinstance(intr_val, dict):
                    itype = intr_val.get("type")
                    if itype == "web_fallback":
                        meta_actions = meta.get("actions") or []
                        if not meta_actions or meta_actions[-1] != "Недостаточно контекста → предлагаю web-поиск":
                            meta_actions.append("Недостаточно контекста → предлагаю web-поиск")
                        meta["actions"] = meta_actions
                        actions_placeholder.markdown(_render_status_md(meta))
                    elif itype == "confirm_rewrite":
                        meta_actions = meta.get("actions") or []
                        if not meta_actions or meta_actions[-1] != "Ожидаю подтверждения перефразирования":
                            meta_actions.append("Ожидаю подтверждения перефразирования")
                        meta["actions"] = meta_actions
                        actions_placeholder.markdown(_render_status_md(meta))
                assistant_placeholder.warning(msg)
                return

            if ev.event == "done":
                answer = ""
                if isinstance(ev.data, dict):
                    answer = ev.data.get("answer") or acc
                else:
                    answer = acc
                if answer:
                    st.session_state.chat[chat_index]["content"] = str(answer)
                    assistant_placeholder.markdown(answer)
                st.session_state.pending_interrupt = None
                st.session_state.pending_interrupt_raw = None
                st.session_state.pending_interrupt_chat_index = None
                return

            if ev.event == "error":
                msg = ev.data.get("message") if isinstance(ev.data, dict) else str(ev.data)
                assistant_placeholder.error(msg)
                st.session_state.chat[chat_index]["content"] = f"Ошибка: {msg}"
                return
    finally:
        st.session_state.is_streaming = False


def _render_interrupt_ui() -> None:
    intr = st.session_state.pending_interrupt
    if not intr:
        return

    if not isinstance(intr, dict):
        st.warning("Получен interrupt (неожиданный формат). Можно отправить raw resume ниже.")
        return

    itype = intr.get("type")
    message = intr.get("message") or "Выберите как поступить."
    st.warning(message)

    thread_id = (
        (st.session_state.get("pending_interrupt_thread_id") or "").strip()
        or (st.session_state.thread_id or "").strip()
    )

    if itype == "confirm_rewrite":
        original = intr.get("original_question", "")
        rephrased = intr.get("rephrased_question", "")

        st.markdown("**Оригинал**")
        st.code(original or "", language=None)
        st.markdown("**Перефразировано**")
        st.code(rephrased or "", language=None)

        label_to_decision = {
            "Подтвердить перефразирование": "approve_rephrased",
            "Оставить как есть": "use_original",
            "Редактировать": "edit",
        }
        choice_label = st.radio(
            "Решение",
            options=list(label_to_decision.keys()),
            horizontal=True,
            key="interrupt_confirm_rewrite_choice",
        )
        decision_value = label_to_decision.get(choice_label, "approve_rephrased")

        edited = ""
        if decision_value == "edit":
            edited = st.text_input(
                "Отредактированный вопрос",
                value=rephrased or original,
                key="interrupt_edit_text",
            )

        c1, c2 = st.columns(2)
        with c1:
            if st.button(
                "Подтвердить",
                type="primary",
                key="resume_confirm_rewrite",
                disabled=bool(st.session_state.get("is_streaming")),
            ):
                resume: Any = {"decision": decision_value}
                if decision_value == "edit":
                    resume["edited_question"] = edited
                st.session_state.resume_request = resume
                st.session_state.resume_thread_id = thread_id
                st.session_state.pending_interrupt = None
                st.session_state.pending_interrupt_raw = None
                st.rerun()
        with c2:
            if st.button("Сбросить", key="clear_interrupt_confirm"):
                st.session_state.pending_interrupt = None
                st.session_state.pending_interrupt_raw = None
                st.rerun()
        return

    if itype == "web_fallback":
        dec = st.radio(
            "Искать в интернете (Tavily)?",
            options=["Да", "Нет"],
            horizontal=True,
            key="interrupt_web_fallback_choice",
        )
        if st.button(
            "Подтвердить",
            type="primary",
            key="resume_web_fallback",
            disabled=bool(st.session_state.get("is_streaming")),
        ):
            st.session_state.resume_request = dec
            st.session_state.resume_thread_id = thread_id
            st.session_state.pending_interrupt = None
            st.session_state.pending_interrupt_raw = None
            st.rerun()
        return

    st.info(f"interrupt.type={itype!r} — используйте raw resume ниже.")


def _render_raw_resume() -> None:
    if not st.session_state.pending_interrupt:
        return

    with st.expander("Advanced: raw resume", expanded=False):
        raw = st.text_area("resume (JSON или строка)", value="", height=120, key="raw_resume_text")
        if st.button("Send raw resume", key="send_raw_resume", disabled=bool(st.session_state.get("is_streaming"))):
            resume: Any = raw
            try:
                resume = json.loads(raw)
            except Exception:
                resume = raw
            thread_id = (
                (st.session_state.get("pending_interrupt_thread_id") or "").strip()
                or (st.session_state.thread_id or "").strip()
            )
            st.session_state.resume_request = resume
            st.session_state.resume_thread_id = thread_id
            st.session_state.pending_interrupt = None
            st.session_state.pending_interrupt_raw = None
            st.rerun()


def _queue_user_message() -> None:
    text = (st.session_state.get("message_input") or "").strip()
    if not text:
        return
    if st.session_state.get("pending_interrupt") or st.session_state.get("is_streaming"):
        return
    st.session_state.pending_user_message = text
    st.session_state.clear_message_input = True

def _send_suggested(question: str) -> None:
    if st.session_state.get("pending_interrupt") or st.session_state.get("is_streaming"):
        return
    st.session_state.pending_user_message = question
    st.session_state.clear_message_input = True


def main() -> None:
    _init_state()

    st.markdown(
        """
        <style>
        .cover-glow {
            width: 100%;
            height: auto;
            padding: 3px;
            box-shadow:
                0 0 5px #330000,
                0 0 10px #660000,
                0 0 15px #990000,
                0 0 20px #CC0000,
                0 0 25px #FF0000,
                0 0 30px #FF3333,
                0 0 35px #FF6666;
            position: relative;
            z-index: -1;
            border-radius: 45px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Chat UI styling (centered + bubble look like the reference screenshot)
    st.markdown(
        """
        <style>
        /* Center main content area */
        section.main > div.block-container {
            max-width: 980px;
            padding-top: 2.2rem;
            padding-bottom: 7rem; /* room for sticky chat input */
        }

        /* Chat bubbles */
        div[data-testid="stChatMessage"] {
            padding-top: 0.35rem;
            padding-bottom: 0.35rem;
        }
        div[data-testid="stChatMessageContent"] {
            background: #F3F5F7;
            border-radius: 14px;
            padding: 0.95rem 1.1rem;
            box-shadow: 0 1px 0 rgba(0,0,0,0.06);
            width: fit-content;
            max-width: 640px;
        }

        /* Lighter actions/status text inside assistant bubbles */
        .agent-actions {
            color: rgba(0,0,0,0.58);
            font-size: 0.92rem;
            line-height: 1.25rem;
            margin-bottom: 0.55rem;
        }
        .agent-actions ul {
            margin-top: 0.2rem;
            margin-bottom: 0.2rem;
        }

        /* Chat input (red border like screenshot) */
        .message-label {
            max-width: 820px;
            margin: 0.35rem auto 0.2rem auto;
            color: rgba(0,0,0,0.65);
            font-size: 0.9rem;
        }

        .suggestions-row {
            max-width: 820px;
            margin: 0.2rem auto 1rem auto;
            display: flex;
            flex-wrap: wrap;
            gap: 0.4rem;
        }
        .suggestions-row div[data-testid="stButton"] {
            margin: 0 !important;
        }
        .suggestions-row div[data-testid="stButton"] > button {
            border-radius: 9999px !important;
            border: 1px solid #d0d7de !important;
            padding: 0.35rem 0.85rem !important;
            background: #f6f8fa !important;
            color: #2f353d !important;
            font-size: 0.9rem !important;
            box-shadow: none !important;
        }
        .suggestions-row div[data-testid="stButton"] > button:hover {
            border-color: #ff4b4b !important;
            color: #ff4b4b !important;
            background: #fff !important;
        }

        div[data-testid="stTextInput"] > div {
            max-width: 820px;
            margin-left: auto;
            margin-right: auto;
        }

        div[data-testid="stTextInput"] [data-baseweb] {
            border: 0 !important;
            box-shadow: none !important;
            background: transparent !important;
            outline: none !important;
        }
        div[data-testid="stTextInput"] div[data-baseweb="input"] {
            border: 2px solid #ff4b4b !important;
            border-radius: 10px !important;
            background: #ffffff !important;
            padding-left: 0.35rem;
        }
        div[data-testid="stTextInput"] input {
            padding: 0.85rem 1rem !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # --- Sidebar (Streamly-style) ---
    img_base64 = img_to_base64("car2.png")
    if img_base64:
        st.sidebar.markdown(
            f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
            unsafe_allow_html=True,
        )
    st.sidebar.markdown("---")

    st.sidebar.markdown("---")
    st.session_state.show_actions = st.sidebar.checkbox("Show Agent Actions", value=bool(st.session_state.show_actions))

    st.sidebar.markdown("---")
    show_basic_info = st.sidebar.checkbox("Show Basic Interactions", value=True)
    if show_basic_info:
        st.sidebar.markdown(
            """
            ### Как работает агент

        1) **Классификация запроса (on-topic / off-topic)**  
           Агент проверяет, относится ли вопрос к **DIY-уходу за авто / мойке / детейлингу**.  
           Если вопрос не по теме — агент предложит уточнение/перефразирование.  
           Если после перефразирования тема всё равно не совпадает — агент честно сообщит, что это вне рамок.

        2) **Перефразирование для поиска (HITL — подтверждение)**  
           Агент формирует «поисковую» версию вашего вопроса (учитывая историю диалога) и
           **останавливается** с блоком *interrupt*, чтобы вы выбрали:  
           - **Подтвердить перефразирование** (рекомендуется)  
           - **Оставить как есть**  
           - **Редактировать** (внести правки вручную)

        3) **Поиск контекста через инструменты**  
           По умолчанию агент выполняет поиск в **внутренней базе (Chroma: MMR + BM25)** через `retrieve_in_vectordb`.  
           Если в базе не нашлось релевантных данных, агент может предложить **поиск в интернете** через `tavily_search`
           (только после вашего согласия).

        4) **Оценка найденных документов (фильтрация)**  
           Каждый найденный фрагмент проверяется отдельной моделью-грейдером:
           **оставляем только то, что действительно отвечает на вопрос**.

        5) **Генерация ответа (финальная модель, streaming)**  
           Финальный ответ строится на основе:
           - отобранного контекста,
           - истории диалога,
           - текущего вопроса.  
           Ответ показывается **потоком** (streaming).

        6) **Если контекста нет — попытка уточнить вопрос**  
           При отсутствии релевантных документов агент слегка уточняет формулировку и повторяет поиск.
           При достижении лимита попыток — предлагает веб-поиск (Tavily) или завершает без ответа.

        **Подсказки**
        - **Задайте вопрос** про мойку/детейлинг/самостоятельный уход за авто.  
        - **Если агент остановился**, выберите вариант в блоке *interrupt* и нажмите «Подтвердить».  
        - **Новый диалог** — кнопка “New thread”.
        """
        )

    # st.sidebar.markdown("---")
    # if img_base64:
    #     st.sidebar.markdown(
    #         f'<img src="data:image/png;base64,{img_base64}" class="cover-glow">',
    #         unsafe_allow_html=True,
    #     )

    # --- Connection / session controls ---
    with st.sidebar.expander("Backend & Session", expanded=True):
        st.session_state.backend_url = st.text_input(
            "backend_url",
            value=st.session_state.backend_url,
            help="Напр.: http://127.0.0.1:8000",
            disabled=bool(st.session_state.get("is_streaming")),
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("Health", disabled=bool(st.session_state.get("is_streaming"))):
                _refresh_health()
        with c2:
            st.write(
                "✅ ok" if st.session_state.health_ok else ("⚠️ " + (st.session_state.health_status or "unknown"))
            )

        ok = bool(st.session_state.health_ok)
        if not st.session_state.health_status:
            _refresh_health()
            ok = bool(st.session_state.health_ok)

        thread_controls_disabled = bool(st.session_state.get("is_streaming")) or bool(st.session_state.pending_interrupt)

        recent = list(st.session_state.recent_threads or [])
        if st.session_state.thread_id:
            _remember_thread(st.session_state.thread_id)
            recent = list(st.session_state.recent_threads or [])

        selected = st.selectbox(
            "thread_id (recent)",
            options=[""] + recent,
            index=(1 if st.session_state.thread_id in recent else 0),
            label_visibility="collapsed",
            placeholder="Выберите ранее использованный thread_id",
            disabled=thread_controls_disabled,
        )
        if selected and selected != st.session_state.thread_id:
            st.session_state.thread_id = selected
            st.session_state.autoload_attempted = False

        st.session_state.thread_id = st.text_input(
            "thread_id",
            value=st.session_state.thread_id,
            key="thread_id_input",
            help="Скопируйте сюда thread_id, чтобы продолжить сессию.",
            disabled=thread_controls_disabled,
        )

        c3, c4 = st.columns(2)
        with c3:
            if st.button("New thread", disabled=bool(st.session_state.get("is_streaming"))):
                st.session_state.thread_id = str(uuid.uuid4())
                _remember_thread(st.session_state.thread_id)
                st.session_state.chat = []
                st.session_state.internals = []
                st.session_state.pending_interrupt = None
                st.session_state.pending_interrupt_raw = None
                st.session_state.pending_interrupt_thread_id = ""
                st.session_state.autoload_attempted = False
                _set_query_params(thread_id=st.session_state.thread_id)
                st.rerun()
        with c4:
            if st.button("Load state", disabled=bool(st.session_state.get("is_streaming")) or (not ok)):
                if st.session_state.thread_id:
                    payload, err = api_get_state(st.session_state.backend_url, st.session_state.thread_id)
                    st.session_state.last_state_error = err or ""
                    if payload:
                        _load_chat_from_state(payload)
                    elif err:
                        st.error(f"Не удалось загрузить state: {err}")
                    else:
                        st.info("State не найден (возможно backend перезапускался).")
                st.rerun()

    # --- Main content ---
    ok = bool(st.session_state.health_ok)
    if not ok:
        st.warning("Backend недоступен. Запусти `uvicorn lib.handlers.app:app --reload --port 8000`.")

    # Resume (if interrupt accepted)
    # Render chat history (last N)
    for m in st.session_state.chat[-NUMBER_OF_MESSAGES_TO_DISPLAY:]:
        role = m["role"]
        if role == "assistant":
            avatar = "car2.png" if os.path.exists("car2.png") else ("🤖")
        else:
            avatar = "🙂"
        with st.chat_message(role, avatar=avatar):
            if role == "assistant" and st.session_state.get("show_actions"):
                meta = m.get("meta") or {}
                if isinstance(meta, dict):
                    md = _render_status_md(meta)
                    if md:
                        st.markdown(f'<div class="agent-actions">{md}</div>', unsafe_allow_html=True)
            if m.get("content"):
                st.markdown(m["content"])

    streaming_container = st.container()

    if st.session_state.get("resume_request") is not None and ok:
        resume_tid = (
            (st.session_state.get("resume_thread_id") or "").strip()
            or (st.session_state.get("pending_interrupt_thread_id") or "").strip()
            or (st.session_state.thread_id or "").strip()
        )
        if not resume_tid:
            st.error("Не могу отправить resume: пустой thread_id (backend вернёт 422).")
            st.session_state.resume_request = None
            st.session_state.resume_thread_id = ""
            st.rerun()

        idx = st.session_state.get("pending_interrupt_chat_index")
        if not isinstance(idx, int) or idx < 0 or idx >= len(st.session_state.chat):
            _append_chat("assistant", "", meta={"actions": []})
            idx = len(st.session_state.chat) - 1

        with streaming_container:
            with st.chat_message("assistant", avatar=("car2.png" if os.path.exists("car2.png") else "🤖")):
                _run_stream(
                    api_stream_resume(
                        st.session_state.backend_url,
                        thread_id=resume_tid,
                        resume=st.session_state.resume_request,
                    ),
                    chat_index=idx,
                )
        st.session_state.resume_request = None
        st.session_state.resume_thread_id = ""
        st.session_state.pending_interrupt_thread_id = ""
        st.rerun()

    # Interrupt UI
    if st.session_state.pending_interrupt:
        st.subheader("Агент ожидает подтверждения от Вас")
        _render_interrupt_ui()
        _render_raw_resume()

    disabled = bool(st.session_state.pending_interrupt) or bool(st.session_state.get("is_streaming")) or (not ok)

    if st.session_state.get("clear_message_input"):
        st.session_state.message_input = ""
        st.session_state.clear_message_input = False

    st.markdown('<div class="message-label">Message:</div>', unsafe_allow_html=True)
    st.text_input(
        "Message",
        key="message_input",
        placeholder="Введите вопрос про DIY детейлинг…",
        disabled=disabled,
        label_visibility="collapsed",
        on_change=_queue_user_message,
    )

    suggestions = [s.strip() for s in st.session_state.get("suggested_questions") or [] if s and s.strip()]
    if suggestions:
        st.markdown("<div class='suggestions-row'>", unsafe_allow_html=True)
        for idx, question in enumerate(suggestions):
            if st.button(question, key=f"suggested_{idx}"):
                _send_suggested(question)
        st.markdown("</div>", unsafe_allow_html=True)

    pending = st.session_state.get("pending_user_message")
    if isinstance(pending, str) and pending.strip():
        chat_input = pending.strip()
        st.session_state.pending_user_message = None
        with streaming_container:
            if not st.session_state.thread_id:
                st.session_state.thread_id = str(uuid.uuid4())
                _remember_thread(st.session_state.thread_id)
                _set_query_params(thread_id=st.session_state.thread_id)

            _append_chat("user", chat_input)
            with st.chat_message("assistant", avatar=("car2.png" if os.path.exists("car2.png") else "🤖")):
                _append_chat("assistant", "", meta={"actions": []})
                chat_index = len(st.session_state.chat) - 1
                _run_stream(
                    api_stream_start(st.session_state.backend_url, st.session_state.thread_id, chat_input),
                    chat_index=chat_index,
                )
        st.rerun()


if __name__ == "__main__":
    main()
