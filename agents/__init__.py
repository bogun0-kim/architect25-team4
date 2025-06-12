from .mcp_agent_client import get_agent_client as get_mcp_agent_client
from .rest_api_agent_client import get_agent_client as get_rest_api_agent_client


_CLIENTS = {
    "MCP": get_mcp_agent_client,
    "RESTAPI": get_rest_api_agent_client,
}


def get_agent_client(agent_type: str, config: dict, *args, **kwargs):
    return _CLIENTS[agent_type.upper()](config, *args, **kwargs)
