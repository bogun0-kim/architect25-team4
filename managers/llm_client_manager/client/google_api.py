from langchain_google_genai import ChatGoogleGenerativeAI
from ._client import LLMClientCreator, LLMClient, register_creator


# TODO: delete
from .__debug import ChatGoogleGenerativeAI


class GoogleLLMClientCreator(LLMClientCreator):
    api_type = "GOOGLE"

    def get_instance(self, config: dict, **kwargs) -> LLMClient:
        self._update_from_env(config)
        config.update(kwargs)
        return ChatGoogleGenerativeAI(**config)


register_creator(GoogleLLMClientCreator)


__all__ = [
    "GoogleLLMClientCreator",
]
