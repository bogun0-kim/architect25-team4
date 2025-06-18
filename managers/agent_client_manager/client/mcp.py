from typing import List, get_type_hints, Optional, Union
from pydantic import BaseModel, Field
import asyncio
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, FunctionMessage
from langchain_core.tools import BaseTool
from langchain_core.tools import StructuredTool
from langgraph.graph.graph import CompiledGraph
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.tools import tool
from ...llm_client_manager import *


def create_subagent_tool(
        mcp_agent: CompiledGraph,
        tool_name: str,
        tool_desc: str,
) -> BaseTool:
    # Define the input schema
    class SubAgentInput(BaseModel):
        input: str = Field(..., description="The input string to process through the sub-agent")
        context: Optional[Union[str, List[str]]] = Field(default=[], description="Optional context")

    # Define the tool function
    async def call_agent(input: str, context: Optional[Union[str, List[str]]] = None) -> str:
        content = input

        # You can optionally inject context if needed
        # TODO: context test
        if context is not None:
            if isinstance(context, (list, tuple)):
                context = ',\n'.join([v for c in context if (v := str(c).strip()) != ''])
            context = str(context).strip()
            if context != '':
                content = f'{content}\n\n<context>{context}</context>'

        # `mcp_agent` input_schema is
        # from langgraph.prebuilt.chat_agent_executor import AgentState
        agent_input = {"messages": [HumanMessage(content)]}
        agent_output = await mcp_agent.ainvoke(agent_input)
        output = None
        if isinstance(agent_output, dict) and "messages" in agent_output:
            output = []
            for msg in agent_output["messages"]:
                # TODO: AIMessage?
                if not isinstance(msg, (AIMessage, ToolMessage, FunctionMessage)):
                    continue
                content = msg.content.strip()
                if content != '' and content not in output:
                    output.append(content)
            if len(output) == 0:
                # TODO: no response? None vs error message
                # output = None
                output = f'[ERROR] No response from the agent: {tool_name}'
            else:
                output = '\n'.join(output)
        if output is None:
            output = str(agent_output)

        # print(f'@@ [call_agent] {tool_name}\n'
        #       f' >> input={input}, context={context}\n'
        #       f' >> agent_input={agent_input}\n'
        #       f' >> agent_output={agent_output}\n'
        #       f' >> output={output}\n')
        return output

    # Return as structured tool
    return StructuredTool.from_function(
        name=tool_name,
        description=tool_desc,
        coroutine=call_agent,
        args_schema=SubAgentInput,
    )


def generate_tool_description(tool: StructuredTool) -> str:
    # print(f"Generating description for tool: {tool.name}")

    sig = tool.args_schema if tool.args_schema else tool.func.__annotations__
    type_hints = get_type_hints(tool.func) if tool.func else get_type_hints(tool.coroutine)
    return_type = type_hints.get("return", "Unknown")
    doc = tool.description.strip() if tool.description else ""
    func_name = tool.name

    lines = []
    lines.append(f"{func_name}(...) -> {return_type.__name__ if hasattr(return_type, '__name__') else str(return_type)}:")
    lines.append(f" - {doc if doc else 'Performs the function defined by this tool.'}")

    # 인자 설명
    if hasattr(sig, "__annotations__"):
        for name, typ in sig.__annotations__.items():
            typename = typ.__name__ if hasattr(typ, '__name__') else str(typ)
            lines.append(f" - `{name}`: {typename} type input.")
            if name == "context" and "list" in typename.lower():
                lines.append(" - You can optionally provide a list of strings as `context` to help the tool operate correctly.")
    #else:
    #    lines.append(" - (No input schema found)")
    return "\n".join(lines)


@tool
def request_user_input_tool(question: str) -> str:
    """Request additional input from the user. This tool should be called when more information is needed from the user to complete the task."""
    return f"[HumanInTheLoop] {question}"

def generate_descriptions_for_tools(tools: List[BaseTool]) -> str:
    header = (
        "You are an agent equipped with a set of MCP tools. Use these tools to accurately fulfill user requests.\n\n"
        "Each tool has a defined function signature, specific input parameters, and a fixed output format. Read them carefully before selecting and invoking a tool.\n\n"
        "General guidelines:\n"
        "- Always choose the most appropriate tool based on the user's intent.\n"
        "- Strictly follow the input type and parameter names as defined in the tool description.\n"
        "- If `context` is provided, use it to improve your understanding or disambiguate the request.\n"
        "- Do not fabricate or infer tool outputs. Only return what the tool actually provides.\n"
        "- Avoid redundant tool calls. Call each tool only as needed and avoid repeating the same call with identical parameters.\n"
        "- If chaining values across multiple tools (e.g., using the output of one tool in another), you MUST explicitly pass those values through parameters or context.\n"
        "- Tool outputs are not retained implicitly. You must track and reuse them manually as needed.\n"
        "- NEVER pass raw outputs from search-type or exploratory tools directly into other tools unless explicitly designed to do so. If needed, extract relevant information and use it in `context`.\n"
        "- If any critical information is missing, or if there is ambiguity that requires user confirmation (e.g., multiple matches, unspecified targets), then:\n"
        "  👉 You MUST use `request_user_input_tool` to explicitly ask the user for the required information or clarification.\n"
        "- Always be precise in your requests. For example, instead of asking \"what is the value\", ask \"what is the temperature in Celsius\", or \"what is the email subject\".\n\n"
        "Your objective is to:\n"
        "- Execute tools accurately\n"
        "- Minimize unnecessary calls\n"
        "- Ensure the user remains in control when uncertainty arises\n"
    )
    tool_descriptions = [generate_tool_description(tool) for tool in tools]
    return header + "\n\n" + "============== Available Tool ==============\n" + "\n\n".join(tool_descriptions)


# def get_agent_client(config: dict, llm: BaseChatModel, *args, **kwargs) -> BaseTool:
#     name = config["name"]
#     mcp_config = config["mcp"]
#     client = MultiServerMCPClient({
#         name: {
#             "url": mcp_config["url"],
#             "transport": mcp_config.get("transport", "streamable-http")
#         },
#     })
#     tools = asyncio.run(client.get_tools())
#     tools.append(request_user_input_tool)
#
#     desc = generate_descriptions_for_tools(tools)
#     agent: CompiledGraph = create_react_agent(model=llm, tools=tools, prompt=desc)
#     # print(f'# agent: {name}')
#     # for n, t in enumerate(tools, 1):
#     #     print(f'  tool-#{n}: {t.name}, {t.description}')
#     # print(f'# ==== agent desc ====\n{desc}\n========')
#     return create_subagent_tool(agent, tool_name=name, tool_desc=config["description"])


def _agent_client(
        client: MultiServerMCPClient,
        name: str,
        description: str,
        **kwargs,
) -> BaseTool:
    llm: BaseChatModel = kwargs.get("llm", LLMClientManager.get())
    tools = asyncio.run(client.get_tools())
    tools.append(request_user_input_tool)
    desc = generate_descriptions_for_tools(tools)
    agent: CompiledGraph = create_react_agent(model=llm, tools=tools, prompt=desc)
    return create_subagent_tool(agent, tool_name=name, tool_desc=description)


from ._client import AgentClientCreator, AgentClient, register_creator


class McpAgentClientCreator(AgentClientCreator):
    api_type: str = "MCP"

    def get_instance(self, config: dict, **kwargs) -> AgentClient:
        client = MultiServerMCPClient({config["name"]: {**config["mcp"]}})
        return _agent_client(client, config["name"], config["description"], **kwargs)


register_creator(McpAgentClientCreator)


__all__ = [
    "McpAgentClientCreator",
]
