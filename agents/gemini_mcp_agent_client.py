import asyncio
from typing import List, Optional, get_type_hints
import threading

# New imports for creating a StructuredTool
from pydantic import BaseModel, Field
from langchain_core.tools import StructuredTool

from langchain_core.messages import SystemMessage, HumanMessage 
from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.prebuilt import create_react_agent
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_google_genai import ChatGoogleGenerativeAI


def get_agent_client(config: dict, llm: BaseChatModel=None) -> BaseTool:
    """
    Creates an agent client according to config.
    """
    if llm is None:
        raise RuntimeError("The argument LLM is None")
    with asyncio.Runner() as runner:
        return runner.run(_create_agent(config, llm))

async def _create_agent(config: dict, llm) -> BaseTool:
    """
    Create an react agent and turn it into an agent as tool.
    """
    name = config["name"]
    description = config["description"]
    mcp_config = config["mcp"]
    print("###########################")
    print(f"Creating an agent client {name}")
    print("###########################")

    # 1. Get tools
    print("###########################")
    print(f"Retrieving tools from MCP Server {name}")
    print("###########################")
    tools = await _get_tools(name, mcp_config)
    
    # 2. Generate prompt

    print("###########################")
    print(f"Generating the prompt for the agent {name}")
    print("###########################")
    prompt = _generate_prompt_for_tools(tools)
    print(f"\nAgent Prompt: \n{prompt}")

    # 3. Create an react agent
    print("###########################")
    print(f"Creating an react agent {name}")
    print("###########################")
    agent = create_react_agent(model=llm, tools=tools, prompt=prompt)

    # 4. Create an agent as a tool
    print("###########################")
    print(f"Creating an agent as a tool {name}")
    print("###########################")
    agent_tool = _create_agent_as_tool(agent, system_prompt=prompt, tool_name= name, tool_description=description)

    print("###########################")
    print(f"Created an agent client {name}")
    print("###########################")
    print(f"Tool Name: {agent_tool.name}")
    print(f"Tool Description: {agent_tool.description}")
    print(f"Tool Args: {list(agent_tool.args.keys())}")
    return agent_tool


# This function runs an async coroutine
def _async_to_sync_safe(coro):
    result = None
    error = None

    def runner():
        nonlocal result, error
        try:
            result = asyncio.run(coro)
        except Exception as e:
            error = e

    thread = threading.Thread(target=runner)
    thread.start()
    thread.join()

    if error:
        raise error
    return result


def _create_agent_as_tool(agent_executor, system_prompt: str, tool_name: str, tool_description: str) -> BaseTool:
    """
    Wraps a LangGraph agent executor in a StructuredTool.
    """
    # Define the input schema
    class SubAgentInput(BaseModel):
        """Defines the input argument for our agent tool."""
        input: str = Field(description="A detailed, natural language query for the agent.")

    # Define the tool function
    def run_agent(input: str) -> str:
        """The async function that will be executed when the tool is called."""

        agent_input = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=input)
            ]
        }

        print(f"\nFunction Call:\n\tinput={input}")
        result = _async_to_sync_safe(agent_executor.ainvoke(agent_input))
        print(f"Function Call: result: {result}")
        return result['messages'][-1].content

    # The parameter is changed from 'func' to 'coroutine' to correctly
    # handle the async function. This is the only change needed.
    return StructuredTool.from_function(
        name=tool_name,
        description=tool_description,
        coroutine=run_agent,  
        args_schema=SubAgentInput
    )

def _get_system_prompt(): 
    return (
        "You are an agent equipped with a set of MCP tools. Use these tools to accurately fulfill user requests.\n\n"
        "Each tool has a specific function signature, input requirements, and output format. Read them carefully before selecting and invoking a tool.\n\n"
        "- Always choose the most relevant tool based on the task.\n"
        "- Strictly follow the input type and parameter names as described.\n"
        "- If `context` is provided, use it to improve the accuracy of your answer.\n"
        "- Do not fabricate tool outputs. Only return what the tool provides.\n"
        "- You can optionally provide a list of strings as `context` to clarify any ambiguity (e.g., time of day, elevation, past weather).\n"
        "- This tool does NOT retain the output of previous calls. If chaining values (e.g., using temperature in math), you MUST explicitly pass prior outputs via `context`.\n"
    )


async def _get_tools(name: str, mcp_config: dict) -> BaseTool:
    """
    Gets the MCP tool list from the MCP server with mcp_config
    """
    return await MultiServerMCPClient({
        name: {
            "url": mcp_config["url"],
            "transport": mcp_config.get("transport", "streamable-http")
        },
    }).get_tools()

def _generate_tool_description(tool: StructuredTool) -> str:
    """
    Generate the descrtiption of the tool given as an argument.
    TODO: Check the failure to get the info of parameters from the tool. However LLM can call a tool 
    with proper arguments without its parameter info
    """
    print(f"Generating description for tool: {tool.name}")
    
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
    else:
        lines.append(" - (No input schema found)")

    return "\n".join(lines)


def _generate_prompt_for_tools(tools: List[BaseTool]):
    system_prompt = _get_system_prompt()
    tool_description = [_generate_tool_description(tool) for tool in tools]
    return system_prompt + "\n\n" + "\n\n".join(tool_description)






##################################################################################
# Test Code
##################################################################################

def _get_test_weather_agent_config():
    return {
        "name": "weather_agent",
        "description": "Useful for answering questions about the current weather, temperature, fine dust levels, and precipitation chance in a specific city.",
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp",
        }
    }

def _get_test_knox_mail_agent_config():
    return {
        "name": "knox_mail_agent",
        "description": "Useful for sending mail, listing unread mail, and read mail",
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8002/mcp",
        }
    }

def _test_weather_agent(agent):
    print("\n#############################################\n")
    print("\nWeather Agent Test")
    query = "What is the fine dust level in Busan?"
    
    tool_result = agent.invoke({"input": query})
    
    print(f"\nQuery: '{query}'")
    print(f"Result from tool: {tool_result}")

    query2 = "What is the weather like in Busan and what is the fine dust level there?"
    print("#############################################")
    print(f"Query 2: {query2}")
    tool_result2 = agent.invoke({"input": query2})

    print(f"Result from tool: {tool_result2}")


def _test_knox_mail_agent(agent):
    print("\n#############################################\n")
    print("\nKnox Mail Agent Test")

    print("\n--- Sending mail to a person ---")
    query = "Email at ionia.kim@samsung.com with the subject 'Hi,' the message I'm fine."
    
    # This ainvoke call was already correct. It will now work as expected.
    tool_result = agent.invoke({"input": query})
    
    print(f"\nQuery: '{query}'")
    print(f"Result from tool: {tool_result}")

    print("\n--- Composite Task")
    query = "Read the first mail from unread mail"
    tool_result = agent.invoke({"input": query})

    print(f"\nQuery: '{query}'")
    print(f"Result from tool: {tool_result}")



def _test():
    """
    Tests the mcp agent. The GEMINI_API_KEY environment variable must be set to work.
    """
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-lite", temperature=0)
    weather_agent_config = _get_test_weather_agent_config()
    weather_agent = get_agent_client(weather_agent_config, llm)
    knox_mail_agent_config = _get_test_knox_mail_agent_config()
    knox_mail_agent = get_agent_client(knox_mail_agent_config, llm)


    _test_weather_agent(weather_agent)
    _test_knox_mail_agent(knox_mail_agent)

if __name__ == '__main__':
    _test()