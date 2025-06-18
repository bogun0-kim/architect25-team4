from abc import ABCMeta, abstractmethod
from langchain_core.tools import BaseTool as AgentClient


class AgentClientCreator(metaclass=ABCMeta):
    api_type: str = None

    def __init__(self):
        assert self.api_type is not None

    @abstractmethod
    def get_instance(self, *args, **kwargs) -> AgentClient:
        raise NotImplementedError()


_CREATOR_MAP = {}


def register_creator(cls: type[AgentClientCreator]):
    if issubclass(cls, AgentClientCreator):
        _CREATOR_MAP[cls.api_type] = cls


def get_creator(api_type: str) -> type[AgentClientCreator]:
    return _CREATOR_MAP.get(api_type, None)


__all__ = [
    "AgentClient",
    "AgentClientCreator",
    "register_creator",
    "get_creator",
]
