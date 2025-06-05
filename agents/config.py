from dataclasses import dataclass


@dataclass
class AgentClientConfig:
    agent_type: str = ''
    name: str = ''
    description: str = ''


@dataclass
class McpAgentClientConfig(AgentClientConfig):
    transport: str = ''
    url: str = ''


@dataclass
class RestApiAgentClientConfig(AgentClientConfig):
    url: str = ''
