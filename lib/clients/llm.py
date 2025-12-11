
from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model
import os

gpt_oss_20b = init_chat_model(
    model="openai/gpt-oss-20b:free",
    model_provider="openai",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    # # прокидывание провайдер-специфичных аргументов:
    extra_body={"temperature": 0}
)

gpt_oss_120b = init_chat_model(
    model="openai/gpt-oss-120b",
    model_provider="openai",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    # # прокидывание провайдер-специфичных аргументов:
    extra_body={"temperature": 0}
)