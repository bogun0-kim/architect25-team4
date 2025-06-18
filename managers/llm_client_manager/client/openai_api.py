from langchain_openai import ChatOpenAI
from ._client import LLMClientCreator, LLMClient, register_creator


# TODO: delete
from .__debug import ChatOpenAI


class OpenAILLMClientCreator(LLMClientCreator):
    api_type = "OPENAI"
    default_model = "gpt-4.1-mini"

    def get_instance(self, config: dict, **kwargs) -> LLMClient:
        self._update_from_env(config)
        config.update(kwargs)
        return ChatOpenAI(**config)


register_creator(OpenAILLMClientCreator)


__all__ = [
    "OpenAILLMClientCreator",
]
