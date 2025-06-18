import importlib
from ._client import AgentClientCreator, AgentClient, register_creator


class FunctionAgentClientCreator(AgentClientCreator):
    api_type: str = "FUNCTION"

    def get_instance(self, config: dict, **kwargs) -> AgentClient:
        function_path = config["function"]
        module_path, function_name = function_path.rsplit('.', 1)
        module = importlib.import_module(module_path)
        function = getattr(module, function_name)
        return function(config, **kwargs)


register_creator(FunctionAgentClientCreator)


__all__ = [
    "FunctionAgentClientCreator",
]
