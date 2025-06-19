from data_access import *
from .client import *


_CLIENTS: dict[str, AgentClient] = {}


class AgentClientManager:
    @staticmethod
    def set(client: AgentClient | dict, force: bool = False, **kwargs) -> bool:
        global _CLIENTS
        if isinstance(client, dict):
            if not force and client["name"] in _CLIENTS:
                return False
            api_type: str = client.pop("api_type")
            creator_class = get_creator(api_type)
            client = creator_class().get_instance(client, **kwargs)
        _CLIENTS[client.name] = client
        return True

    @staticmethod
    def get(client: AgentClient | str, default=None) -> AgentClient | None:
        global _CLIENTS
        return _CLIENTS.get(client.name if isinstance(client, AgentClient) else str(client), default)

    @staticmethod
    def pop(client: AgentClient | str) -> AgentClient | None:
        global _CLIENTS
        return _CLIENTS.pop(client.name if isinstance(client, AgentClient) else str(client), None)

    @staticmethod
    def data() -> dict[str, AgentClient]:
        global _CLIENTS
        return dict(_CLIENTS)

    @staticmethod
    def refresh(force: bool = False, **kwargs) -> tuple[list[dict], list[str]]:
        global _CLIENTS
        configs = get_data_access().get("agent")
        disconnected = []
        for k, v in _CLIENTS.items():
            if sum([1 if k == config["name"] else 0 for config in configs]) == 0:
                disconnected.append(k)
        connected = []
        for config in configs:
            if AgentClientManager.set(dict(config), force, **kwargs):
                connected.append(config)
        return connected, disconnected


__all__ = [
    "AgentClientManager",
]
