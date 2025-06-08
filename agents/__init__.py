from .mcp_agent_client import get_agent_client as get_mcp_agent_client
from .gemini_mcp_agent_client import get_agent_client as get_gemini_mcp_agent_client
from .rest_api_agent_client import get_agent_client as get_rest_api_agent_client



_DATA = {
    "mcp": get_mcp_agent_client,
    "gemini_mcp": get_gemini_mcp_agent_client,
    "restapi": get_rest_api_agent_client,
}


def get_agent_client(agent_type: str, config: dict, *args, **kwargs):
    return _DATA[agent_type.lower()](config, *args, **kwargs)
