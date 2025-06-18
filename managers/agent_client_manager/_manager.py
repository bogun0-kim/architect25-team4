from data_access import *
from ..llm_client_manager import *
from .client import *


# Singleton instances
_CLIENTS: dict[str, AgentClient] = {}


class AgentClientManager:
    @staticmethod
    def set(client: AgentClient | dict, **kwargs):
        global _CLIENTS
        if isinstance(client, dict):
            print(f'# Agent configuration: {client}')
            api_type: str = client.pop("api_type")
            creator_class = get_creator(api_type)
            client = creator_class().get_instance(client, **kwargs)
        _CLIENTS[client.name] = client

    @staticmethod
    def get(client: AgentClient | str, default=None) -> AgentClient | None:
        global _CLIENTS
        return _CLIENTS.get(client.name if isinstance(client, AgentClient) else str(client), default)

    @staticmethod
    def pop(client: AgentClient | str) -> AgentClient | None:
        global _CLIENTS
        return _CLIENTS.pop(client.name if isinstance(client, AgentClient) else str(client), None)

    @staticmethod
    def list() -> list[AgentClient]:
        global _CLIENTS
        return list(_CLIENTS.values())

    @staticmethod
    def data() -> dict[str, AgentClient]:
        global _CLIENTS
        return dict(_CLIENTS)

    @staticmethod
    def refresh(**kwargs):
        configs = get_data_access().get("agent")
        for config in configs:
            AgentClientManager.set(config, **kwargs)


__all__ = [
    "AgentClientManager",
]
