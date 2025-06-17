import os
try:
    from ..llm import get_llm_client
    from ..managers.llm_manager import LLM
    from ..managers.tool_manager import ToolManager
except ImportError:
    from llm import get_llm_client
    from managers.llm_manager import LLM
    from managers.tool_manager import ToolManager
try:
    from .tools.math_tool import get_math_tool
    from .tools.search_tool import get_search_tool
except ImportError:
    from _demo.tools.math_tool import get_math_tool
    from _demo.tools.search_tool import get_search_tool


def prepare(agents):
    openai_api_key = os.getenv("OPENAI_API_KEY", None)
    google_api_key = os.getenv("GOOGLE_API_KEY", None)
    if openai_api_key is not None:
        llm_type = "OPENAI"
        config = {
            "api_key": openai_api_key,
            "model": os.getenv("OPENAI_MODEL", 'gpt-4.1-mini'),
            "base_url": None,
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        # print(f'OpenAI: {config["model"]}')
    elif google_api_key is not None:
        llm_type = "GOOGLE"
        config = {
            "api_key": google_api_key,
            "model": os.getenv("GOOGLE_MODEL", 'gemini-2.0-flash'),
            "base_url": None,
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        # print(f'Google: {config["model"]}')
    else:
        llm_type = "OPENAI"
        config = {
            "api_key": "EMPTY",
            "model": '',
            "base_url": 'http://34.64.195.131:80/v1',
            "max_tokens": 1024,
            "temperature": 0.0,
        }
        # print(f'LLM: {config["base_url"]}')
    LLM.set(get_llm_client(llm_type, config))

    ToolManager.set(get_math_tool(LLM.get()))
    # ToolManager.set(get_search_tool())

    ToolManager.set(agents.get_agent_client("mcp", {
        "name": "mail_agent",
        "description": f"""
{{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
* This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
* The agent is equipped with multiple tools (e.g., list unread mails, read mail, send mail, etc.) and can autonomously choose the most suitable tool for the user's intent.
* `query` can be either a simple keyword (e.g. "latest email") or a natural language question (e.g. "when did I receive the last email from John?").
* You cannot handle multiple request in one call. For instance, `{{agent_name}}('get email from John, get email from Jane')` does not work. If you need to process multiple request, you need to call them separately like `{{agent_name}}('get email from John')` and then `{{agent_name}}('get email from Jane')`.
* Minimize the number of `mail` actions as much as possible. For instance, instead of calling 2. {{agent_name}}("what is the subject of $1") and then 3. {{agent_name}}("what is the sender of $1"), you MUST call 2. {{agent_name}}("what is the subject and sender of $1") instead, which will reduce the number of mail actions.
* You can optionally provide a list of strings as `context` to help the agent understand the query. If there are multiple contexts you need to answer the query, you can provide them as a list of strings.
* `{{agent_name}}` action will not see the output of the previous actions unless you provide it as `context`. You MUST provide the output of the previous actions as `context` if you need to refer to them.
* You MUST NEVER provide `search` type action's outputs as a variable in the `query` argument. This is because `search` returns a text blob, not a structured email object. Therefore, when you need to provide an output of `search` action, you MUST provide it as a `context` argument to `mail` action. For example, 1. search("John’s email") and then 2. {{agent_name}}("get sender of $1") is NEVER allowed. Use 2. {{agent_name}}("get sender of John’s email", context=["$1"]) instead.
* When you ask a question about `context`, specify the email fields explicitly. For instance, "what is the subject of this email?" or "who is the sender?" instead of vague questions like "what is this?"
* When sending an email, a **valid and correctly formatted email address** is required.
  Before asking the user directly, the agent MUST first try to find the email address using the contact agent.
  If the address still cannot be determined or if multiple results are returned ambiguously, you MUST use `request_user_input_tool` to explicitly ask the user for confirmation.
  NEVER guess or fabricate an email address under any circumstance.
        """.strip().format(agent_name='mail_agent'),
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8002/mcp",
        },
    }, LLM.get()))

    ToolManager.set(agents.get_agent_client("mcp", {
        "name": "weather_agent",
        "description": f"""
{{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
* This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
* The agent is equipped with multiple tools (e.g., math, weather queries, etc.) and can autonomously choose the most suitable tool for the user's intent.
* The `input` should be a plain English request describing what the user wants to know or compute.
* The `context` field is optional and can include supplemental information from previous steps or system memory to improve accuracy.
* The output is a final answer generated after the agent completes reasoning and tool execution.
* You should not assume the agent knows everything; it only knows what its tools allow it to observe or compute.
* `Seoul` is the default location when a location parameter is not provided.
* Do not include multiple unrelated questions in a single input. The agent processes one task per request.
* DO NOT guess the current date when the user does not specify a date.
* If a `date` parameter is required but not provided, you MUST call calendar agent first to obtain the date.
        """.strip().format(agent_name='weather_agent'),
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8001/mcp",
        },
    }, LLM.get()))

    ToolManager.set(agents.get_agent_client("mcp", {
        "name": "calendar_agent",
        "description": f"""
{{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
* This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
* The agent is equipped with multiple tools and can autonomously choose the most suitable tool for the user's intent.
* The `input` should be a plain English request describing what the user wants to know or compute.
* The `context` field is optional and can include supplemental information from previous steps or system memory to improve accuracy.
* The output is a final answer generated after the agent completes reasoning and tool execution.
* You should not assume the agent knows everything; it only knows what its tools allow it to observe or compute.
* Do not include multiple unrelated questions in a single input. The agent processes one task per request.
* DO prioritize answering date-related questions such as "What is today?" or "What day is it?" using the `get_today_date` tool.
* DO NOT attempt to compute or guess the date using reasoning alone. Always rely on the `get_today_date` tool for date-related answers.
        """.strip().format(agent_name='calendar_agent'),
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8003/mcp",
        },
    }, LLM.get()))

    ToolManager.set(agents.get_agent_client("mcp", {
        "name": "jira_agent",
        "description": f"""
{{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
* This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
* The agent is equipped with multiple tools and can autonomously choose the most suitable tool for the user's intent.
* The `input` should be a plain English request describing what the user wants to know or compute.
* The `context` field is optional and can include supplemental information from previous steps or system memory to improve accuracy.
* The output is a final answer generated after the agent completes reasoning and tool execution.
* You should not assume the agent knows everything; it only knows what its tools allow it to observe or compute.
* Do not include multiple unrelated questions in a single input. The agent processes one task per request.
    """.strip().format(agent_name='jira_agent'),
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8004/mcp",
        },
    }, LLM.get()))

    ToolManager.set(agents.get_agent_client("mcp", {
        "name": "contact_agent",
        "description": f"""
{{agent_name}}(input="input message for {{agent_name}}", context="context for {{agent_name}}") -> str:
* This is a unified interface to a multi-tool agent. It takes a natural language input, interprets the request, and uses internal MCP tools to execute the appropriate actions.
* The agent is equipped with multiple tools (e.g., math, weather queries, etc.) and can autonomously choose the most suitable tool for the user's intent.
* The `input` should be a plain English request describing what the user wants to know or compute.
* The `context` field is optional and can include supplemental information from previous steps or system memory to improve accuracy.
* The output is a final answer generated after the agent completes reasoning and tool execution.
* You should not assume the agent knows everything; it only knows what its tools allow it to observe or compute.
* Do not include multiple unrelated questions in a single input. The agent processes one task per request.
    """.strip().format(agent_name='contact_agent'),
        "mcp": {
            "transport": "streamable_http",
            "url": "http://localhost:8006/mcp",
        },
    }, LLM.get()))
