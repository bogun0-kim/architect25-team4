from ._client import LLMClientCreator, LLMClient, register_creator
from langchain_openai import ChatOpenAI


class VllmLLMClientCreator(LLMClientCreator):
    api_type = "VLLM"

    def get_instance(self, config: dict, **kwargs) -> LLMClient:
        self._update_from_env(config)
        config.update(kwargs)
        return ChatOpenAI(**config)


register_creator(VllmLLMClientCreator)


__all__ = [
    "VllmLLMClientCreator",
]
