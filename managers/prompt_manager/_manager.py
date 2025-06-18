# import os
import json
from langchain_core.load.load import loads
from langchain_core.prompts import ChatPromptTemplate
from data_access import *


# def load_from_json(data: str | dict | list) -> ChatPromptTemplate:
#     import json
#     if isinstance(data, str) and os.path.isfile(data):
#         with open(data, 'r', encoding='utf-8') as f:
#             json_obj = json.load(f)
#     elif isinstance(data, str):
#         json_obj = json.loads(data)
#     else:
#         json_obj = data
#     json_str = json.dumps(json_obj, ensure_ascii=True)
#     return loads(json_str)


# plan_prompt_template_json = os.path.abspath(os.path.join(os.path.dirname(__file__), 'plan-hitl.json'))
# join_prompt_template_json = os.path.abspath(os.path.join(os.path.dirname(__file__), 'join-hitl.json'))


# _replan: str = """* Replan (under "Current Plan"):
#   - You are given "Previous Plan" which is the plan that the previous agent created along with the execution results (given as Observation) of each plan and a general thought (given as Thought) about the executed results. You MUST use these information to create the next plan under "Current Plan".
#   - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.
#   - In the Current Plan, you should NEVER repeat the actions that are already executed in the Previous Plan.
#   - You must continue the action index from the end of the previous one. Do not repeat action indices.""".strip()
#
# _hitl_plan: str = """
# * Ongoing Plan (under "Current Plan"):
#   - You are given the "Previous Plan", which is provided between human messages. This includes the plan that the previous agent created, along with the execution results for each step (provided as "Observation") and a general assessment (provided as "Thought") about those results. You MUST use these information to create the next plan under "Current Plan".
#   - When starting the Current Plan, you should start with "Thought" that outlines the strategy for the next plan.
#   - You must continue the action index from the end of the previous one. Do not repeat action indices.
#   - Additional user input is provided to help address the USER_REQUEST. You must create the "Current Plan" based on this updated input.
#   - When referencing previous action outputs, you may refer to actions from the "Previous Plan", but make sure it is truly necessary before doing so.
# """.strip()

# TODO: simple memory DB
_default_key = "default"
_DATA: dict[str, dict[str, ChatPromptTemplate | str]] = {}


class PromptManager:
    @staticmethod
    def get(key: str) -> ChatPromptTemplate | str | None:
        global _DATA
        return _DATA.get(key, _DATA[_default_key])

    @staticmethod
    def refresh():
        global _DATA
        dao = get_data_access()
        replan_json = dao.get("prompt", "replan.json")[0]
        plan_prompt_template_json = dao.get("prompt", "plan-hitl.json")[0]
        join_prompt_template_json = dao.get("prompt", "join-hitl.json")[0]
        _plan: ChatPromptTemplate = loads(json.dumps(plan_prompt_template_json, ensure_ascii=True))
        _join: ChatPromptTemplate = loads(json.dumps(join_prompt_template_json, ensure_ascii=True))
        _DATA[_default_key] = {
            "plan": _plan,
            "replan": replan_json["replan"],
            "hitl_plan": replan_json["hitl_plan"],
            "join": _join,
        }
