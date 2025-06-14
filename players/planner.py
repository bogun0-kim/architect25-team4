################################################################################
# Planner
################################################################################

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, ToolMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch
from langchain_core.tools import BaseTool
from .output_parser import LLMCompilerPlanParser


# TODO: prompt
start_number_description = """
Note:
* When listing the actions, start numbering from {next_task}.
* The starting point for action numbering has changed. When referencing previous actions, you'd better be sure you're using the correct IDs — especially for anything numbered below {next_task}. GETTING THIS WRONG IS NOT AN OPTION.
""".strip()


def build(
        model: BaseChatModel,
        tools: dict[str, BaseTool],
        prompt_template: ChatPromptTemplate,
        replanner_description: str,
        hitl_description: str,
) -> Runnable:
    num_tools = len(tools) + 1  # Add one because we're adding the join() tool at the end.
    tool_descriptions = '\n'.join(f'{n}. {tool.description}\n' for n, tool in enumerate(tools.values(), 1))
    planner_prompt = prompt_template.partial(
        num_tools=num_tools,
        tool_descriptions=tool_descriptions,
        continuing_plan='')
    replanner_prompt = prompt_template.partial(
        num_tools=num_tools,
        tool_descriptions=tool_descriptions,
        continuing_plan=replanner_description)
    hitl_prompt = prompt_template.partial(
        num_tools=num_tools,
        tool_descriptions=tool_descriptions,
        continuing_plan=hitl_description)

    def should_replan(messages: list):
        # Context is passed as a system message
        return isinstance(messages[-1], SystemMessage)

    def should_human_in_the_loop(messages: list):
        is_first_user_input = len([True for msg in messages if isinstance(msg, HumanMessage)]) == 1
        return not is_first_user_input and isinstance(messages[-1], HumanMessage)

    def wrap_messages(messages: list):
        return {"messages": messages}

    def wrap_and_get_last_index(messages: list):
        next_task = 0
        for message in messages[::-1]:
            if isinstance(message, ToolMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        if next_task > 1:
            desc = start_number_description.format(next_task=next_task)
            if isinstance(messages[-1], SystemMessage):
                messages[-1].content = messages[-1].content.strip() + '\n\n' + desc
            else:
                messages.append(SystemMessage(desc))
        return {"messages": messages}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            (should_human_in_the_loop, wrap_and_get_last_index | hitl_prompt),
            wrap_messages | planner_prompt,
        )
        | model
        | LLMCompilerPlanParser(tools=tools)
    )
