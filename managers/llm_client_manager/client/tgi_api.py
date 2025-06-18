from ._client import LLMClientCreator, LLMClient, register_creator
from langchain_openai import ChatOpenAI


class TgiLLMClientCreator(LLMClientCreator):
    api_type = "TGI"

    def get_instance(self, config: dict, **kwargs) -> LLMClient:
        self._update_from_env(config)
        config.update(kwargs)
        return ChatOpenAI(**config)


register_creator(TgiLLMClientCreator)


__all__ = [
    "TgiLLMClientCreator",
]
