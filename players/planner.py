################################################################################
# Planner
################################################################################

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import FunctionMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableBranch
from langchain_core.tools import BaseTool
from .output_parser import LLMCompilerPlanParser


def build(
        model: BaseChatModel,
        tools: dict[str, BaseTool],
        prompt_template: ChatPromptTemplate,
        replanner_description: str,
) -> Runnable:
    num_tools = len(tools) + 1  # Add one because we're adding the join() tool at the end.
    tool_descriptions = '\n'.join(f'{n}. {tool.description}\n' for n, tool in enumerate(tools.values(), 1))
    planner_prompt = prompt_template.partial(
        replan='', num_tools=num_tools, tool_descriptions=tool_descriptions)
    replanner_prompt = prompt_template.partial(
        replan=replanner_description, num_tools=num_tools, tool_descriptions=tool_descriptions)

    print(f'@@@@ BUILDING @@@@')
    print('@@ <planner_prompt> @@')
    planner_prompt.pretty_print()
    print('@@ <replanner_prompt> @@')
    replanner_prompt.pretty_print()
    print('@@ <tool_descriptions> @@')
    print(tool_descriptions)

    def should_replan(messages: list):
        # Context is passed as a system message
        return isinstance(messages[-1], SystemMessage)

    def wrap_messages(messages: list):
        return {"messages": messages}

    def wrap_and_get_last_index(messages: list):
        next_task = 0
        for message in messages[::-1]:
            if isinstance(message, FunctionMessage):
                next_task = message.additional_kwargs["idx"] + 1
                break
        messages[-1].content = messages[-1].content + f' - Begin counting at : {next_task}'
        return {"messages": messages}

    return (
        RunnableBranch(
            (should_replan, wrap_and_get_last_index | replanner_prompt),
            wrap_messages | planner_prompt,
        )
        | model
        | LLMCompilerPlanParser(tools=tools)
    )
