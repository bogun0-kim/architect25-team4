################################################################################
# Task Fetching Unit: plan and schedule
################################################################################

import re
import time
import itertools
from typing import Any, Dict, Iterator, List, Union, Optional
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, wait
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_core.runnables import chain as as_runnable
from langchain_core.runnables.base import Runnable
from .output_parser import Task


# TODO:
_PRE_PLANNER_PROMPT_TEMPLATE = """
<USER_REQUEST>
{user_query}
</USER_REQUEST>

Break down the tasks needed to fulfill the user request.
List only the task breakdown and their descriptions.""".strip()

_SIMPLE_PLAN_TEMPLATE = "<TASK_LIST>\n{simple_plan}\n</TASK_LIST>"

_RESPONSE_REFINER_PROMPT_TEMPLATE = """
<QUERY>{query}<QUERY>
<RESPONSE>{response}</RESPONSE>

Find the correct answer corresponding to the QUERY from the RESPONSE.
Do not include any additional explanations, ONLY provide the answer.""".strip()


class SchedulerInput(TypedDict):
    model: BaseChatModel
    user_query: str
    messages: List[BaseMessage]
    tasks: Iterator[Task]


class ExecutorInput(TypedDict):
    model: Optional[BaseChatModel]
    user_query: str
    task: Task
    observations: Dict[int, ToolMessage]


def _get_observations(messages: List[BaseMessage]) -> Dict[int, ToolMessage]:
    # Get all previous tool responses
    return {int(msg.additional_kwargs["idx"]): msg for msg in messages[::-1] if isinstance(msg, ToolMessage)}


def _resolve_arg(
        arg: Union[str, Any],
        observations: Dict[int, ToolMessage],
        model: BaseChatModel,
        user_query: str,
):
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [_resolve_arg(v, observations, model, user_query) for v in arg]
    if isinstance(arg, dict):
        return {k: _resolve_arg(v, observations, model, user_query) for k, v in arg.items()}
    if not isinstance(arg, str):
        return arg

    _ID_PATTERN = r'\$\{?(\d+)\}?'  # $1 or ${1} -> 1

    # TODO: test
#     _test_mode = False
#     if _test_mode:
#         var_idx_ref = []
#         for m in re.compile(_ID_PATTERN).finditer(arg):
#             variable = m.group(0)
#             index = int(m.group(1))
#             observation: ToolMessage = observations.get(index, None)
#             if observation is None:
#                 reference = variable
#             else:
#                 reference = observation.additional_kwargs.get("result", None)
#                 if reference is None:
#                     reference = observation.content
#             var_idx_ref.append([variable, index, reference])
#         if len(var_idx_ref) == 0:
#             return arg
#
#         _RESOLVER_PROMPT_TEMPLATE = f"""
# <QUERY>{user_query}</QUERY>
# {{references}}
# <PROBLEM>{{problem}}</PROBLEM>
#
# PROBLEM contains one or more variables in the format of $ID.
# Each variable $ID must be replaced with a constant value extracted exclusively from REFERENCE-ID that matches the same number.
# For example, $1 must be filled using information from REFERENCE-1, and $3 must be filled using information from REFERENCE-3.
# First, understand the contents of the QUERY, and based on that, determine which information from each REFERENCE should be used.
# Then, replace each $ID with the appropriate constant value to complete the PROBLEM.
# Only submit the completed PROBLEM as your answer. Do not include any explanations, comments, or additional text.
# Now, complete the PROBLEM in its final form.
#         """.strip()
#         _REFERENCE_TEMPLATE = '<REFERENCE-{idx}>{content}</REFERENCE-{idx}>'
#
#         print(f'@@ {__file__} >> _resolve_arg2: arg={arg}')
#         try:
#             resolver_prompt = _RESOLVER_PROMPT_TEMPLATE.format(
#                 references='\n'.join([_REFERENCE_TEMPLATE.format(idx=i, content=r) for v, i, r in var_idx_ref]),
#                 problem=arg)
#             resolved_arg = model.invoke(ChatPromptValue(messages=[HumanMessage(resolver_prompt)]))
#             print(f'@@ {__file__} >> _resolve_arg2: resolved_arg={resolved_arg}')
#             return resolved_arg.content
#         except Exception as e:
#             for v, i, r in var_idx_ref:
#                 arg = arg.replace(v, r)
#             print(f'@@ {__file__} >> _resolve_arg2: replaced={arg} / {e}')
#             return arg

    for m in re.compile(_ID_PATTERN).finditer(arg):
        index = int(m.group(1))
        observation: ToolMessage = observations.get(index, None)
        if observation is None:
            continue
        reference = observation.additional_kwargs.get("result", None)
        if reference is None:
            reference = observation.content
        variable = m.group(0)
        arg = arg.replace(variable, reference)
    arg = arg.strip()
    for quote in ['"', "'"]:
        if arg.startswith(quote) and arg.endswith(quote) and quote not in arg[1:-1]:
            arg = arg[1:-1].strip()
    return arg


def _execute_with_retry(func, *args, max_attempts: int = 3, retry_delay_seconds: float = 0.2, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            print(f'# [{attempt}/{max_attempts}] Failed to call: {func} (Exception={e})')
            if max_attempts <= attempt:
                return None, e
            time.sleep(retry_delay_seconds)


def _execute_task(
        task: Task,
        observations: Dict[int, ToolMessage],
        config,
        model: BaseChatModel,
        user_query: str,
) -> List[str]:
    tool = task["tool"]
    args = task["args"]
    print(f'@@ {__file__} >> _execute_task: tool={type(tool)}, {tool}')
    if isinstance(tool, str):
        return [tool]

    ################################################################################
    # TODO: Resolve arguments
    ################################################################################
    print(f'@@ {__file__} >> _execute_task: args={type(args)}, {args}')
    try:
        resolved_args = _resolve_arg(args, observations, model, user_query)
    except Exception as e:
        print(f'@@ {__file__} >> _execute_task: failed to resolve args. {e}')
        error = (
            f'ERROR'
            f' (Failed to call {tool.name} with args {args}.'
            f' Args could not be resolved. Error: {repr(e)})')
        return [error]
    print(f'@@ {__file__} >> _execute_task: resolved_args={type(resolved_args)}, {resolved_args}')
    ################################################################################

    ################################################################################
    # Execute tool
    ################################################################################
    print(f'@@ {__file__} >> _execute_task: execute> {type(tool)}.invoke({resolved_args}, config={config})')
    output, error = _execute_with_retry(tool.invoke, resolved_args, config)
    if error is not None:
        error = (
            f'ERROR'
            f' (Failed to call {tool.name} with args {args}.'
            f' Args resolved to {resolved_args}. Error: {repr(error)})')
        return [error]
    print(f'@@ {__file__} >> _execute_task: tool_output> {output}')
    ################################################################################

    ################################################################################
    # TODO: Refine the response of tool
    ################################################################################
    query = None
    if isinstance(resolved_args, str):
        if resolved_args.strip() != '':
            query = resolved_args
    elif isinstance(resolved_args, (list, tuple)):
        if len(resolved_args) > 0:
            query = '\n'.join([str(v) for v in resolved_args])
    elif isinstance(resolved_args, dict):
        if len(resolved_args) == 1:
            query = resolved_args[list(resolved_args)[0]]
        elif len(resolved_args) > 1:
            query = '\n'.join([f'{k}={v}' for k, v in resolved_args.items()])
    if query is not None:
        print(f'@@ {__file__} >> _execute_task: refine> query={query}, response={output}')
        try:
            refiner_prompt = _RESPONSE_REFINER_PROMPT_TEMPLATE.format(query=query, response=str(output))
            refined_output = model.invoke(ChatPromptValue(messages=[HumanMessage(refiner_prompt)]))
            print(f'@@ {__file__} >> _execute_task: refined_output> {refined_output}')
            return [output, refined_output.content]
        except Exception as e:
            print(f'@@ {__file__} >> _execute_task: not refined. {e}')
            return [output]

    return [output]


@as_runnable
def _schedule_task(executor_input: ExecutorInput, config):
    task: Task = executor_input["task"]
    observations: Dict[int, ToolMessage] = executor_input["observations"]
    try:
        outputs = _execute_task(
            task, observations, config,
            executor_input.get("model", None),
            executor_input.get("user_query", None))
    except Exception as e:
        import traceback
        outputs = [traceback.format_exception(e)]

    additional_kwargs = {"idx": task["idx"], "args": task["args"]}
    if len(outputs) > 1:
        additional_kwargs["result"] = outputs[1]
    observations[task["idx"]] = ToolMessage(
        name=task["tool"] if isinstance(task["tool"], str) else task["tool"].name,
        content=str(outputs[0]),
        additional_kwargs=additional_kwargs,
        tool_call_id=task["idx"])


def _schedule_pending_task(
        task: Task,
        observations: Dict[int, ToolMessage],
        model: BaseChatModel,
        user_query: str,
        retry_delay_seconds: float = 0.2,
):
    while True:
        # Dependencies not yet satisfied
        dependencies = task["dependencies"]
        if dependencies and (any([d not in observations for d in dependencies])):
            time.sleep(retry_delay_seconds)
            continue

        _schedule_task.invoke(ExecutorInput(
            task=task, observations=observations,
            model=model, user_query=user_query))
        break


@as_runnable
def _schedule_tasks(scheduler_input: SchedulerInput) -> List[ToolMessage]:
    """Group the tasks into a DAG schedule."""

    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure.
    user_query = scheduler_input["user_query"]
    messages = scheduler_input["messages"]
    tasks = scheduler_input["tasks"]
    model = scheduler_input["model"]
    _task_names = {}
    _task_args = {}

    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    original_keys = set(observations)

    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_delay_seconds = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            _task_names[task["idx"]] = task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            _task_args[task["idx"]] = task["args"]

            # Depends on other tasks
            dependencies = task["dependencies"]
            if dependencies and (any([d not in observations for d in dependencies])):
                futures.append(executor.submit(
                    _schedule_pending_task, task, observations, model, user_query, retry_delay_seconds))

            # No dependencies or all dependencies satisfied, can schedule now
            else:
                _schedule_task.invoke(
                    ExecutorInput(task=task, observations=observations, model=model, user_query=user_query))
                # futures.append(executor.submit(
                #     _schedule_task.invoke,
                #     ExecutorInput(task=task, observations=observations, model=model, user_query=user_query)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)

    # Convert observations to new tool messages to add to the state
    tool_messages = []
    for idx in sorted(observations.keys() - original_keys):
        tool_messages.append(AIMessage(
            content='', additional_kwargs={
                "tool_calls": [
                    {"id": str(idx), "function": {"name": _task_names[idx], "arguments": str(_task_args[idx])}}
                ]
            }
        ))
        tool_messages.append(observations[idx])

    return tool_messages


@as_runnable
def _pre_plan_without_descriptions(inputs):
    messages = inputs["messages"]
    model = inputs["model"]
    user_query: str = ''
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_query = msg.content
            break
    pre_planner = _PRE_PLANNER_PROMPT_TEMPLATE.format(user_query=user_query)
    simple_plan = model.invoke(ChatPromptValue(messages=[HumanMessage(pre_planner)]))
    messages.append(AIMessage(
        content=_SIMPLE_PLAN_TEMPLATE.format(simple_plan=simple_plan.content),
        additional_kwargs={"TEST": True}))
    return messages


def build(planner: Runnable, model: BaseChatModel) -> Runnable:

    @as_runnable
    def plan_and_execute(state):
        # Get messages and extract user_query
        messages = state["messages"]
        user_query = None
        for msg in messages:
            if user_query is None and isinstance(msg, HumanMessage):
                user_query = msg.content
            print(f'@@ {__file__} >> plan_and_execute[1]: messages={msg.__class__} {msg}')

        if len(messages) == 1:
            messages = _pre_plan_without_descriptions.invoke({"messages": messages, "model": model})
            for msg in messages:
                print(f'@@ {__file__} >> plan_and_execute[1a]: messages={msg.__class__} {msg}')

        tasks: Iterator[Task] = planner.stream(messages)
        print(f'@@ {__file__} >> plan_and_execute[2]: tasks={tasks}')
        # Begin executing the planner immediately
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            # Handle the case where tasks is empty.
            tasks = iter([])
        print(f'@@ {__file__} >> plan_and_execute[3] tasks={tasks}')

        # Execute tasks
        executed_tasks = _schedule_tasks.invoke(SchedulerInput(
            messages=messages, tasks=tasks,
            model=model, user_query=user_query))
        print(f'@@ {__file__} >> plan_and_execute[4] executed_tasks={executed_tasks}')

        return {"messages": executed_tasks}

    return plan_and_execute
