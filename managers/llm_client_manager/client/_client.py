import os
from abc import ABCMeta, abstractmethod
from langchain_core.language_models import BaseChatModel as LLMClient


class LLMClientCreator(metaclass=ABCMeta):
    api_type: str
    default_model: str = ""

    def __init__(self):
        assert self.api_type is not None

    @abstractmethod
    def get_instance(self, *args, **kwargs) -> LLMClient:
        raise NotImplementedError()

    def _update_from_env(self, config: dict):
        defaults = {"model": self.default_model}
        for k in ("api_key", "model", "base_url"):
            env = os.getenv(f"{self.api_type}_{k.upper()}", None)
            if env is not None:
                config[k] = env
            else:
                v: str = config.get(k, None)
                if v is None or v.strip() == '':
                    config[k] = defaults.get(k, '')
        return config


_CREATOR_MAP = {}


def register_creator(cls: type[LLMClientCreator]):
    if issubclass(cls, LLMClientCreator):
        _CREATOR_MAP[cls.api_type] = cls


def get_creator(api_type: str) -> type[LLMClientCreator]:
    return _CREATOR_MAP.get(api_type, None)


__all__ = [
    "LLMClient",
    "LLMClientCreator",
    "register_creator",
    "get_creator",
]
