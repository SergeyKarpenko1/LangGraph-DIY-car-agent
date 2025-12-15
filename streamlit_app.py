from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests
import streamlit as st


@dataclass(frozen=True)
class SseEvent:
    event: str
    data: Any


def _get_query_params() -> Dict[str, List[str]]:
    try:
        # Streamlit >= 1.30
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
    for raw in resp.iter_lines(decode_unicode=True):
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
    # server sends: {"payload": [{"value": {...}, "resumable": True, "ns": [...]}, ...]}
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
        stream=True,
        timeout=300,
    )
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
        stream=True,
        timeout=300,
    )
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

def _ensure_assistant_meta(chat_index: int) -> Dict[str, Any]:
    meta = st.session_state.chat[chat_index].setdefault("meta", {})
    meta.setdefault("actions", [])
    meta.setdefault("_seen_nodes", set())
    meta.setdefault("_seen_tool_calls", set())
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
            label = {
                "rewrite": "Перефразирование вопроса",
                "classify": "Проверка, что вопрос по теме",
                "plan": "Планирование и выбор инструмента",
                "tools": "Вызов инструмента",
                "collect": "Сбор контекста",
                "grade": "Оценка релевантности документов",
                "refine_question": "Небольшое изменение запроса",
                "generate_answer": "Генерация ответа",
                "cannot_answer": "Недостаточно контекста",
                "off_topic_response": "Вопрос не по теме",
            }.get(node, node)
            if node == "grade":
                actions.append(label)
            else:
                actions.append(f"Выполняется: {label}")

        if not isinstance(payload, dict):
            continue

        docs = payload.get("documents")
        if isinstance(docs, list) and node == "collect":
            actions.append(f"Найдено: {len(docs)} документов")
        if isinstance(docs, list) and node == "grade":
            actions.append(f"Релевантных: {len(docs)} документов")

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
                        # openai-compatible shapes: {name,args} or {function:{name,arguments}}
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
                            actions.append("Поиск в базе данных")
                        elif name == "tavily_search":
                            actions.append("Поиск в интернете")
                        elif name:
                            actions.append(f"Вызов инструмента `{name}`: {_short_json(args)}")
                        else:
                            actions.append(f"Вызов инструмента: {_short_json(tc)}")

            if m_type == "tool":
                tool_name = m.get("name") or m.get("tool") or "tool"
                count = m.get("artifact_count")
                if isinstance(count, int):
                    tcid = m.get("tool_call_id") or ""
                    res_key = f"{tool_name}:{count}:{tcid}"
                    if res_key in seen_tool_results:
                        continue
                    seen_tool_results.add(res_key)

                    if tool_name == "retrieve_in_vectordb":
                        actions.append(f"В базе данных найдено {count} документов")
                    elif tool_name == "tavily_search":
                        actions.append(f"В интернете найдено {count} результатов")
                    else:
                        actions.append(f"Результат `{tool_name}`: {count} документов")
                else:
                    content = m.get("content")
                    if content:
                        tcid = m.get("tool_call_id") or ""
                        res_key = f"{tool_name}:{_short_json(content)}:{tcid}"
                        if res_key in seen_tool_results:
                            continue
                        seen_tool_results.add(res_key)

                        if tool_name == "retrieve_in_vectordb":
                            actions.append("В базе данных найден контекст")
                        elif tool_name == "tavily_search":
                            actions.append("В интернете найден контекст")
                        else:
                            actions.append(f"Результат `{tool_name}`: {_short_json(content)}")

    return actions


def _render_actions_md(actions: List[str]) -> str:
    if not actions:
        return ""
    return "\n".join([f"- {a}" for a in actions])


def _append_chat(role: str, content: str, meta: Optional[Dict[str, Any]] = None) -> None:
    st.session_state.chat.append({"role": role, "content": content, "meta": meta or {}})


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

            if ev.event in ("update", "step"):
                actions = meta.get("actions") or []
                new_actions = _extract_actions_from_update(ev.data, meta)
                if new_actions:
                    actions.extend(new_actions)
                    meta["actions"] = actions
                    actions_placeholder.markdown(_render_actions_md(actions))
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
                assistant_placeholder.warning(msg)
                meta_actions = meta.get("actions") or []
                meta_actions.append(f"Ожидаю уточения от человека (interrupt: {intr_val.get('type') if isinstance(intr_val, dict) else 'unknown'})")
                meta["actions"] = meta_actions
                actions_placeholder.markdown(_render_actions_md(meta_actions))
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
    except Exception as e:
        assistant_placeholder.error(str(e))
        st.session_state.chat[chat_index]["content"] = f"Ошибка запроса к backend: {e}"
        _append_internal("exception", {"message": str(e)})
        return
    finally:
        st.session_state.is_streaming = False


def _render_interrupt_ui(base: str) -> None:
    intr = st.session_state.pending_interrupt
    if not intr:
        return

    if not isinstance(intr, dict):
        st.warning("Получен interrupt (неожиданный формат). Можно отправить raw resume ниже.")
        return

    itype = intr.get("type")
    message = intr.get("message") or "Выберете как поступить."
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
            edited = st.text_input("Отредактированный вопрос", value=rephrased or original, key="interrupt_edit_text")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Подтвердить", type="primary", key="resume_confirm_rewrite", disabled=bool(st.session_state.get("is_streaming"))):
                resume: Any = {"decision": decision_value}
                if decision_value == "edit":
                    resume["edited_question"] = edited
                st.session_state.resume_request = resume
                st.session_state.resume_thread_id = thread_id
                st.session_state.pending_interrupt = None
                st.session_state.pending_interrupt_raw = None
                st.rerun()
        with col2:
            if st.button("Сбросить", key="clear_interrupt_confirm"):
                st.session_state.pending_interrupt = None
                st.session_state.pending_interrupt_raw = None
                st.rerun()
        return

    if itype == "web_fallback":
        dec = st.radio("Искать в интернете (Tavily)?", options=["Да", "Нет"], horizontal=True, key="interrupt_web_fallback_choice")
        if st.button("Подтвердить", type="primary", key="resume_web_fallback", disabled=bool(st.session_state.get("is_streaming"))):
            st.session_state.resume_request = dec
            st.session_state.resume_thread_id = thread_id
            st.session_state.pending_interrupt = None
            st.session_state.pending_interrupt_raw = None
            st.rerun()
        return

    st.info(f"interrupt.type={itype!r} — используйте raw resume ниже.")


def _render_raw_resume(base: str) -> None:
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
            st.session_state.resume_request = resume
            st.session_state.resume_thread_id = st.session_state.thread_id
            st.session_state.pending_interrupt = None
            st.session_state.pending_interrupt_raw = None
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Detailing Agent UI", layout="wide")
    _init_state()

    st.title("Detailing Car Agent")
    st.caption(
        "DIY-агент по уходу за авто: ищет контекст в локальной базе знаний и при необходимости делает веб-поиск. "
        "Иногда останавливается и просит подтверждение (human-in-the-loop)."
    )

    if not st.session_state.health_status:
        _refresh_health()
    ok = bool(st.session_state.health_ok)
    if (
        ok
        and st.session_state.thread_id
        and not st.session_state.chat
        and not st.session_state.autoload_attempted
    ):
        st.session_state.autoload_attempted = True
        payload, err = api_get_state(st.session_state.backend_url, st.session_state.thread_id)
        st.session_state.last_state_error = err or ""
        if payload:
            _load_chat_from_state(payload)

    st.subheader("Чат")
    if not ok:
        st.warning(
            "Backend недоступен. Запусти `uvicorn lib.handlers.app:app --reload --port 8000`."
        )
    if st.session_state.pending_interrupt:
        st.info("Агент остановился и ждёт решения оператора.")
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
        with st.chat_message("assistant"):
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
    for m in st.session_state.chat:
        with st.chat_message(m["role"]):
            if m["role"] == "assistant":
                meta = m.get("meta") or {}
                actions = meta.get("actions") or []
                if isinstance(actions, list) and actions:
                    st.markdown(_render_actions_md(actions))
            if m.get("content"):
                st.markdown(m["content"])

    if st.session_state.pending_interrupt:
        st.subheader("Агент ожидает подтверждения от Вас")
        _render_interrupt_ui(st.session_state.backend_url)
        _render_raw_resume(st.session_state.backend_url)

    disabled = bool(st.session_state.pending_interrupt) or bool(st.session_state.get("is_streaming")) or (not ok)
    with st.form("prompt_form", clear_on_submit=True):
        user_text = st.text_input("Введите вопрос про DIY детейлинг…", disabled=disabled, label_visibility="collapsed")
        submitted = st.form_submit_button("Отправить", disabled=disabled)

    st.caption("Сессия (thread_id)")
    recent = list(st.session_state.recent_threads or [])
    if st.session_state.thread_id:
        _remember_thread(st.session_state.thread_id)
        recent = list(st.session_state.recent_threads or [])

    thread_controls_disabled = bool(st.session_state.get("is_streaming")) or bool(st.session_state.pending_interrupt)
    selected = st.selectbox(
        "Выбор thread_id",
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

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Новый thread", key="new_thread", disabled=bool(st.session_state.get("is_streaming"))):
            st.session_state.thread_id = str(uuid.uuid4())
            _remember_thread(st.session_state.thread_id)
            st.session_state.chat = []
            st.session_state.internals = []
            st.session_state.pending_interrupt = None
            st.session_state.pending_interrupt_raw = None
            st.session_state.autoload_attempted = False
            _set_query_params(thread_id=st.session_state.thread_id)
            st.rerun()
    with c2:
        if st.button("Загрузить state", key="load_state", disabled=bool(st.session_state.get("is_streaming")) or (not ok)):
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

    if submitted and user_text:
        if not st.session_state.thread_id:
            st.session_state.thread_id = str(uuid.uuid4())
            _remember_thread(st.session_state.thread_id)
            _set_query_params(thread_id=st.session_state.thread_id)
        _append_chat("user", user_text)

        with st.chat_message("assistant"):
            _append_chat("assistant", "", meta={"actions": []})
            chat_index = len(st.session_state.chat) - 1
            _run_stream(
                api_stream_start(st.session_state.backend_url, st.session_state.thread_id, user_text),
                chat_index=chat_index,
            )
        st.rerun()


if __name__ == "__main__":
    main()

# streamlit run streamlit_app.py
