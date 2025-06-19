from data_access import *
from .client import *


_CONFIG: dict = {}
_CLIENT: LLMClient


class LLMClientManager:
    @staticmethod
    def set(client: LLMClient | dict, **kwargs) -> bool:
        global _CONFIG
        global _CLIENT
        if isinstance(client, dict):
            is_changed = False
            for k, v in client.items():
                if k not in _CONFIG or _CONFIG[k] != v:
                    is_changed = True
                    break
            if is_changed:
                _CONFIG = dict(client)
                api_type: str = client.pop("api_type")
                creator_class = get_creator(api_type)
                client = creator_class().get_instance(client, **kwargs)
                _CLIENT = client
            return is_changed
        else:
            _CONFIG = {}
            _CLIENT = client
            return True

    @staticmethod
    def get(key: str = None) -> LLMClient | str:
        global _CONFIG
        global _CLIENT
        return _CLIENT if key is None else _CONFIG.get(key, None)

    @staticmethod
    def name() -> str:
        global _CONFIG
        _name = _CONFIG.get("model", None)
        if _name is None or _name == '':
            _name = _CONFIG.get("base_url", None)
        return _name

    @staticmethod
    def refresh(**kwargs) -> bool:
        global _CONFIG
        config = get_data_access().get("llm", "main.json")[0]
        return config if LLMClientManager.set(dict(config), **kwargs) else None


__all__ = [
    "LLMClientManager",
]
