# Detailing Agent

## О проекте
Deteiling Agent — это AI-ассистент для DIY детейлинга и ухода за автомобилем. Проект построен на связке LangChain + LangGraph, использует гибридное извлечение знаний из собственной векторной базы и веб-поиска, а также human-in-the-loop для подтверждения перефраза и переходов к web-search.

## Основные возможности
- Гибридный ретривер: ChromaDB (MMR) + BM25, опциональный reranker на трансформере и компрессия ответов.
- Многошаговый граф LangGraph: переписывание вопроса, классификация, планирование вызова инструментов, оценка контекста, генерация ответа.
- Human-in-the-loop через `interrupt`: подтверждение перефраза, решение о fallback на Tavily Search.
- SSE backend + Streamlit UI с отображением внутренних шагов и подсветкой собранных документов.
- Набор утилит для создания/обновления векторной базы и проверки ретривера.

## Используемые технологии
- Python 3.12, uv
- LangChain, LangGraph, LangChain Chroma, LangChain Community
- HuggingFace Embeddings (`deepvk/USER-bge-m3`)
- ChromaDB, rank-bm25, optional reranker (`naver/xprovence-reranker-bgem3-v1`)
- OpenRouter (GPT-OSS 20B/120B), Tavily Search API
- FastAPI, Streamlit
- Unstructured для загрузки Markdown-документов

## Ключевые модули
- `lib/clients/llm.py` — фабрика для моделей OpenRouter (включая streaming).
- `lib/models/embedder.py`, `lib/models/builderDB.py`, `lib/clients/create_vector_db.py` — эмбеддинги и создание/обновление коллекции Chroma.
- `lib/models/retrieval_tool.py` — гибридный retriever (MMR + BM25 + optional reranker) как LangChain-инструмент.
- `lib/workers/multystep_reasoning_agent.py` — базовая версия графа.
- `lib/workers/multystep_reasoning_agent_stream.py` — потоковая версия графа c interrupt и checkpointer (используется FastAPI/CLI).
- `lib/handlers/app.py` — FastAPI backend с SSE (`/chat/stream`, `/chat/resume/stream`).
- `streamlit_app.py`, `streamlit_app_streamly.py` — Streamlit UI (Streamly-style отображает шаги графа, документы, Tavily результаты).
- `tests/unit/test_retr.py` — smoke-тест для retriever_tool.
- `langgraph.json` — конфигурация для `langgraph dev` (импортирует потоковый граф и `.env`).

## Подготовка окружения
1. Установите Python 3.12+ и при желании [uv](https://github.com/astral-sh/uv).
2. Склонируйте репозиторий и создайте `.env` (см. ниже).
3. Установите зависимости:
   ```bash
   # рекомендуется через uv
   uv sync
   # или классический путь
   pip install -e .
   ```

## Переменные окружения
Пример `.env` уже в репозитории. Минимальный набор:
- `OPENROUTER_API_KEY` — ключ OpenRouter.
- `HUGGINGFACE_HUB_TOKEN` — токен HuggingFace (для загрузки эмбеддингов/моделей).
- `TAVILY_API_KEY` — ключ Tavily Search (опционален, если веб-поиск не нужен).
- `CHROMA_PERSIST_DIR` — путь к ChromaDB (по умолчанию авто-детект `Data/processed/chromadb` или `Data/ChromaDB`).
- `CHROMA_COLLECTION_NAME` — имя коллекции (`VectorDB_deepvk_USER-bge-m3`).
- `BACKEND_URL` — URL backend для Streamlit (дефолт `http://127.0.0.1:8000`).

## Подготовка данных
1. Сложите Markdown-материалы по детейлингу в `Data/raw/articles`.
2. Создайте/обновите ChromaDB:
   ```bash
   uv run python lib/clients/create_vector_db.py
   ```
   По умолчанию БД будет в `Data/processed/chromadb` (можно переопределить через `.env`).

## Локальный запуск
### Smoke-тест ретривера
```bash
uv run python tests/unit/test_retr.py
```

### FastAPI + Streamlit (dev)
Backend на FastAPI (SSE) и Streamlit UI можно поднимать несколькими способами:

- **Makefile**:
  ```bash
  make run           # backend + Streamlit (frontend стартует через 10 секунд)
  make run-backend   # только FastAPI
  make run-frontend  # только Streamlit (можно переопределить BACKEND_URL)
  make format        # ruff format
  make lint          # ruff check
  ```
- **Вручную**:
  ```bash
  uv run uvicorn lib.handlers.app:app --reload --host 127.0.0.1 --port 8000
  BACKEND_URL=http://127.0.0.1:8000 uv run streamlit run streamlit_app_streamly.py
  ```
- **LangGraph CLI**:
  ```bash
  langgraph dev
  ```
  `langgraph.json` импортирует `lib/workers/multystep_reasoning_agent_stream.py:graph` и подключает `.env`.

Streamlit UI отображает внутренние шаги графа (rewrite → classify → plan → tools → grade → generate) и модальные окна HITL.

### Docker / docker-compose
В репозитории есть два минимальных Dockerfile:
- `Dockerfile.backend` — FastAPI/uvicorn (порт 8000).
- `Dockerfile.frontend` — Streamlit (`streamlit_app_streamly.py`, порт 8501, переменная `BACKEND_URL` по умолчанию указывает на сервис `backend`).

`docker-compose.yml` собирает и запускает оба сервиса:
```bash
docker compose up --build
```
Доступ по умолчанию:
- backend — http://localhost:8000
- frontend — http://localhost:8501

Перед запуском убедитесь, что `.env` доступен compose (подключён к сервису backend). Чтобы база Chroma сохранялась между пересборками, смонтируйте volume `./Data/processed/chromadb:/app/Data/processed/chromadb`.

## Тестирование
Пока доступен ручной smoke-тест ретривера (`tests/unit/test_retr.py`). Рекомендуется дополнять проект pytest-тестами (проверка LangGraph узлов, FastAPI эндпоинтов, Streamlit-интеграции).

## Структура репозитория
```
Data/                # Markdown-данные и ChromaDB
lib/                 # Клиенты, модели, графы, FastAPI и вспомогательные утилиты
Notebooks/           # Jupyter-ноутбуки для экспериментов и оценки качества
streamlit_app*.py    # Streamlit UI (обычный и Streamly-style)
langgraph.json       # Конфигурация LangGraph CLI
Dockerfile.*         # Образы backend/frontend
docker-compose.yml   # Одновременный запуск двух сервисов
Makefile             # Утилиты для dev-циклов (запуск, форматирование, линт)
pyproject.toml       # Зависимости и метаданные проекта
```
