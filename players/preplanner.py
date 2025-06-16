from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import BaseOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import chain as as_runnable
from langchain_core.runnables.base import Runnable


# TODO:
_PREPLANNER_PROMPT_TEMPLATE = """
<USER_REQUEST>
{user_request}
</USER_REQUEST>

Break down the tasks needed to fulfill the user request.
Each task must be a minimal unit of action that contributes to fulfilling the overall request.
Only decompose the original request — do not infer, elaborate, or add any new information that is not explicitly stated.
Avoid rephrasing or interpreting the user's intent beyond what is written.
Return only the list of task breakdowns and their brief descriptions.""".strip()



_SIMPLE_PLAN_TEMPLATE = "<TASK_LIST>\n{}\n</TASK_LIST>"


IS_PREPLAN = "is_preplan"



class FormattedOutputParser(BaseOutputParser[AIMessage]):
    template: str = '{}'

    def __init__(self, template: str):
        super().__init__()
        self.template = template

    def parse(self, content: str):
        return AIMessage(content=self.template.format(content), additional_kwargs={IS_PREPLAN: True})

    def get_format_instructions(self) -> str:
        return ''


def build(
        model: BaseChatModel,
        preplanner_prompt_template: str = _PREPLANNER_PROMPT_TEMPLATE,
        simple_plan_template: str = _SIMPLE_PLAN_TEMPLATE,
) -> Runnable:
    prompt = ChatPromptTemplate.from_messages([HumanMessagePromptTemplate.from_template(preplanner_prompt_template)])
    parser = FormattedOutputParser(simple_plan_template)
    _chain = prompt | model | parser

    @as_runnable
    def preplan(state):
        messages = state["messages"]
        is_first_user_input = len([True for msg in messages if isinstance(msg, HumanMessage)]) == 1
        if is_first_user_input:
            new_preplan = _chain.invoke({"user_request": state["user_request"]})
            messages = messages + [new_preplan]
        else:
            old_preplan_indices_to_remove = [
                i for i, msg in enumerate(messages)
                if IS_PREPLAN in msg.additional_kwargs]
            for index in old_preplan_indices_to_remove[::-1]:
                messages.pop(index)
        return {"messages": messages}

    return preplan
