# import os
# from dotenv import load_dotenv
# from langchain.chat_models import init_chat_model


# class SimpleLLMFactory:
#     """Простой класс для инициализации LLM через OpenRouter."""

#     def __init__(
#         self,
#         api_key: str | None = None,
#         base_url: str = "https://openrouter.ai/api/v1",
#         temperature: float = 0.0,
#     ):
#         load_dotenv()
#         self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
#         self.base_url = base_url
#         self.temperature = temperature

#     def create(self, model: str):
#         return init_chat_model(
#             model=model,
#             model_provider="openai",
#             api_key=self.api_key,
#             base_url=self.base_url,
#             extra_body={"temperature": self.temperature},
#         )
    
# Пример вызова

# from lib.models.llm_factory import SimpleLLMFactory

# factory = SimpleLLMFactory(temperature=0)

# gpt_oss_20b = factory.create("openai/gpt-oss-20b:free")
# gpt_oss_120b = factory.create("openai/gpt-oss-120b")


# вариант для стриминга токенов

# lib/models/llm_factory.py
import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

class SimpleLLMFactory:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str = "https://openrouter.ai/api/v1",
        temperature: float = 0.0,
    ):
        load_dotenv()
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.temperature = temperature

    def create(self, model: str, **kwargs):
        # kwargs -> уйдут в __init__ конкретной chat model (OpenAI/совместимой)
        return init_chat_model(
            model=model,
            model_provider="openai",
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=self.temperature,
            **kwargs,
        )