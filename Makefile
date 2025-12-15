.PHONY: run run-backend run-frontend format lint check

APP_MODULE := lib.handlers.app:app
CODE_PATHS := lib tests
BACKEND_URL ?= http://127.0.0.1:8000

run:
	# Запускаем backend в фоне, затем стартуем Streamlit UI
	uv run --active uvicorn $(APP_MODULE) --reload --host 127.0.0.1 --port 8000 &
	sleep 10
	uv run --active env BACKEND_URL=$(BACKEND_URL) streamlit run streamlit_app_streamly.py --server.address 127.0.0.1 --server.port 8501

run-backend:
	# Быстрый запуск FastAPI-бэкенда (uvicorn + uv)
	uv run --active uvicorn $(APP_MODULE) --reload --host 127.0.0.1 --port 8000

run-frontend:
	# Запуск Streamlit UI (по умолчанию слушает локальный backend на 8000)
	uv run --active env BACKEND_URL=$(BACKEND_URL) streamlit run streamlit_app_streamly.py --server.address 127.0.0.1 --server.port 8501

format:
	# Форматирование исходников ruff'ом
	uv run --active ruff format $(CODE_PATHS)

lint:
	# Линтинг кода ruff'ом
	uv run --active ruff check $(CODE_PATHS)

check: lint
