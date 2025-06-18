from data_access import *
from .client import *


# Singleton instance
_CLIENT: LLMClient


class LLMClientManager:
    @staticmethod
    def set(client: LLMClient | dict, **kwargs):
        global _CLIENT
        if isinstance(client, dict):
            print(f'# LLM configuration: {client}')
            api_type: str = client.pop("api_type")
            creator_class = get_creator(api_type)
            client = creator_class().get_instance(client, **kwargs)
        _CLIENT = client

    @staticmethod
    def get() -> LLMClient:
        global _CLIENT
        return _CLIENT

    @staticmethod
    def name() -> str:
        global _CLIENT
        return _CLIENT.name

    @staticmethod
    def refresh(**kwargs):
        config = get_data_access().get("llm", "main.json")[0]
        LLMClientManager.set(config, **kwargs)


__all__ = [
    "LLMClientManager",
]
