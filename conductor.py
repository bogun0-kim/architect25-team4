################################################################################
# Conductor: LangGraph
################################################################################

from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from players.planner import build as build_planner
from players.scheduler import build as build_scheduler
from players.joiner import build as build_joiner


class State(TypedDict):
    messages: Annotated[list, add_messages]


def build(
        model: BaseChatModel,
        tools: list[BaseTool],
        prompts: dict[str, ChatPromptTemplate],
):
    planner: Runnable = build_planner(model, tools, prompts["plan"], prompts["replan"])
    plan_and_execute: Runnable = build_scheduler(planner)
    join: Runnable = build_joiner(model, prompts["join"].partial(examples=''))

    graph = StateGraph(State)

    # Define vertices
    # We defined plan_and_execute above already.
    # Assign each node to a state variable to update.
    planner_executor_node = "plan_and_execute"
    joiner_node = "join"
    graph.add_node(planner_executor_node, plan_and_execute)
    graph.add_node(joiner_node, join)

    # Define edges
    graph.add_edge(planner_executor_node, joiner_node)

    # This condition determines looping logic
    def should_continue(state):
        return END if isinstance(state["messages"][-1], AIMessage) else planner_executor_node

    # Next, we pass in the function that will determine which node is called next.
    graph.add_conditional_edges(joiner_node, should_continue)

    graph.add_edge(START, planner_executor_node)
    return graph.compile()
