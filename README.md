# Detailing Agent

## О проекте
Deteiling Agent — это AI-ассистент для DIY детейлинга и ухода за автомобилем. Проект построен на связке LangChain + LangGraph и использует гибридное извлечение знаний из собственной векторной базы и веб-поиска. Агент умеет переформулировать запросы, оценивать релевантность найденных документов и при необходимости привлекать оператора для принятия решений.

## Основные возможности
- Гибридный ретривер: ChromaDB с MMR + BM25 и компрессия ответов через `LLMChainExtractor`.
- Многошаговый граф рассуждений (LangGraph) с этапами переписывания вопроса, классификации, оценки контекста и генерации ответа.
- Human-in-the-loop: подтверждение перефраза и решение о переходе к веб-поиску через `interrupt`.
- Автоматический fallback на Tavily Search, если локальной информации недостаточно.
- Набор утилит для создания векторной базы и тестирования инструмента извлечения.

![Скриншот графа](https://github.com/SergeyKarpenko1/LangGraph-DIY-car-agent/blob/main/ImageGraph.png) 

## Используемые технологии
- Python 3.12
- LangChain, LangGraph, LangChain Chroma, LangChain Community
- HuggingFace Embeddings (`deepvk/USER-bge-m3`)
- ChromaDB, BM25 (rank-bm25)
- OpenRouter (GPT-OSS 20B/120B), Tavily Search API
- Unstructured для загрузки Markdown-документов

## Ключевые модули
- `Notebooks/llm.py` — инициализация LLM через OpenRouter.
- `Notebooks/retrieval_tool.py` — создание Chroma-векторной базы, гибридного ретривера и инструмента для LangChain.
- `Notebooks/multystep_reasoning_agent.py` — граф рассуждений с состоянием агента и логикой HITL.
- `Notebooks/create_vector_db.py` — одноразовая инициализация/обновление коллекции Chroma.
- `Notebooks/test_retr.py` — утилита для ручной проверки `retriever_tool`.
- `langgraph.json` — конфигурация для запуска графа через LangGraph CLI.

## Подготовка окружения
1. Установите Python 3.12+.
2. Создайте виртуальное окружение:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows: .venv\Scripts\activate
   ```
3. Установите зависимости:
   ```bash
   pip install -e .
   # или, если используете uv:
   uv sync
   ```

## Переменные окружения
Добавьте файл `.env` в корне (пример в репозитории уже есть) со следующими значениями:

- `OPENROUTER_API_KEY` — ключ доступа к OpenRouter.
- `HUGGINGFACE_HUB_TOKEN` — токен для загрузки эмбеддингов.
- `TAVILY_API_KEY` — ключ для Tavily Search (необязателен, если веб-поиск не используется).

## Подготовка данных
1. Сложите исходные Markdown-материалы по детейлингу в `Data/cleaned`.
2. Однократно создайте векторную базу:
   ```bash
   python Notebooks/create_vector_db.py
   ```
   Путь к коллекции (`Data/ChromaDB`) и название (`VectorDB_deepvk_USER-bge-m3`) уже зашиты в скриптах.

## Запуск агента
### 1. Проверка ретривера
```bash
python Notebooks/test_retr.py
```
Скрипт выведет найденные фрагменты для запроса «Чем лучше чернить резину?».

### 2. Запуск графа LangGraph
- Через CLI:
  ```bash
  langgraph dev
  ```
  Конфигурация из `langgraph.json` подтянет граф `Notebooks/multystep_reasoning_agent.py:graph` и .env-файл.
- Через Python:
  ```python
  from langchain_core.messages import HumanMessage
  from Notebooks.multystep_reasoning_agent import graph

  initial_state = {"messages": [HumanMessage(content="Как удалить битум с кузова?")]}
  result = graph.invoke(initial_state)
  ```
  Во время выполнения возможны `interrupt`-события, требующие ручного решения.

## Тестирование
На данный момент предусмотрен ручной тест ретривера (`Notebooks/test_retr.py`). Дополнительные проверки можно добавлять в формате pytest или LangGraph unit-тестов.

## Структура репозитория
```
Data/                # Markdown-данные и ChromaDB
Notebooks/           # Основные модули агента и вспомогательные скрипты
langgraph.json       # Конфигурация LangGraph CLI
pyproject.toml       # Зависимости и метаданные проекта
main.py              # Заглушка для CLI-пакета
```
