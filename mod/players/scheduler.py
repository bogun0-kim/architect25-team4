################################################################################
# Task Fetching Unit: plan and schedule
################################################################################

import re
import time
import itertools
from typing import Any, Dict, Iterator, List, Union
from typing_extensions import TypedDict
from concurrent.futures import ThreadPoolExecutor, wait
from langchain_core.messages import BaseMessage, FunctionMessage
from langchain_core.runnables import chain as as_runnable
from langchain_core.runnables.base import Runnable
from .output_parser import Task


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    print(f'@@ {__file__} >> _get_observations: messages={messages}')
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    print(f'@@ {__file__} >> _get_observations: results={results}')
    return results


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterator[Task]


def _execute_task(task: Task, observations, config):
    tool_to_use = task["tool"]
    if isinstance(tool_to_use, str):
        return tool_to_use
    print(f'# <_execute_task> tool={tool_to_use.name}, args={task["args"]}')
    args = task["args"]
    try:
        if isinstance(args, str):
            resolved_args = _resolve_arg(args, observations)
        elif isinstance(args, dict):
            resolved_args = {k: _resolve_arg(v, observations) for k, v in args.items()}

        # TODO: test
        elif isinstance(args, (list, tuple)):
            resolved_args = _resolve_arg(args, observations)

        else:
            # This will likely fail
            resolved_args = args
    except Exception as e:
        return (
            f'ERROR'
            f' (Failed to call {tool_to_use.name} with args {args}.'
            f' Args could not be resolved. Error: {repr(e)})')
    try:
        return tool_to_use.invoke(resolved_args, config)
    except Exception as e:
        return (
            f'ERROR'
            f' (Failed to call {tool_to_use.name} with args {args}.'
            f' Args resolved to {resolved_args}. Error: {repr(e)})')


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    _ID_PATTERN = r'\$\{?(\d+)\}?'  # $1 or ${1} -> 1

    def replace_match(match):
        # If the string is ${123}, match.group(0) is ${123}, and match.group(1) is 123.

        # Return the match group, in this case the index, from the string. This is the index
        # number we get back.
        idx = int(match.group(1))
        return str(observations.get(idx, match.group(0)))

    # For dependencies on other tasks
    if arg is None:
        return None
    elif isinstance(arg, str):
        return re.sub(_ID_PATTERN, replace_match, arg)
    elif isinstance(arg, list):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


@as_runnable
def _schedule_task(task_inputs, config):
    task: Task = task_inputs["task"]
    observations: Dict[int, Any] = task_inputs["observations"]
    try:
        observation = _execute_task(task, observations, config)
    except Exception as e:
        import traceback
        observation = traceback.format_exception(e)
    observations[task["idx"]] = observation


def _schedule_pending_task(
        task: Task,
        observations: Dict[int, Any],
        retry_after: float = 0.2,
):
    while True:
        dependencies = task["dependencies"]

        # Dependencies not yet satisfied
        if dependencies and (any([d not in observations for d in dependencies])):
            time.sleep(retry_after)
            continue

        _schedule_task.invoke({"task": task, "observations": observations})
        break


@as_runnable
def _schedule_tasks(scheduler_input: SchedulerInput) -> List[FunctionMessage]:
    """Group the tasks into a DAG schedule."""

    # For streaming, we are making a few simplifying assumption:
    # 1. The LLM does not create cyclic dependencies
    # 2. That the LLM will not generate tasks with future deps
    # If this ceases to be a good assumption, you can either
    # adjust to do a proper topological sort (not-stream)
    # or use a more complicated data structure.
    messages = scheduler_input["messages"]
    tasks = scheduler_input["tasks"]
    args_for_tasks = {}

    # If we are re-planning, we may have calls that depend on previous
    # plans. Start with those.
    observations = _get_observations(messages)
    originals = set(observations)
    task_names = {}

    # ^^ We assume each task inserts a different key above to
    # avoid race conditions...
    futures = []
    retry_after = 0.25  # Retry every quarter second
    with ThreadPoolExecutor() as executor:
        for task in tasks:
            dependencies = task["dependencies"]
            task_names[task["idx"]] = task["tool"] if isinstance(task["tool"], str) else task["tool"].name
            args_for_tasks[task["idx"]] = task["args"]

            # Depends on other tasks
            if dependencies and (any([d not in observations for d in dependencies])):
                futures.append(executor.submit(_schedule_pending_task, task, observations, retry_after))

            # No dependencies or all dependencies satisfied, can schedule now
            else:
                _schedule_task.invoke(dict(task=task, observations=observations))
                # futures.append(executor.submit(schedule_task.invoke, dict(task=task, observations=observations)))

        # All tasks have been submitted or enqueued
        # Wait for them to complete
        wait(futures)

    # Convert observations to new tool messages to add to the state
    new_observations = {
        k: (task_names[k], args_for_tasks[k], observations[k])
        for k in sorted(observations.keys() - originals)
    }
    tool_messages = [
        FunctionMessage(
            name=name,
            content=str(obs),
            additional_kwargs={"idx": k, "args": task_args},
            tool_call_id=k,
        )
        for k, (name, task_args, obs) in new_observations.items()
    ]
    return tool_messages


def build(planner: Runnable) -> Runnable:

    @as_runnable
    def plan_and_execute(state):
        messages = state["messages"]
        for msg in messages:
            print(f'# <plan_and_execute> {msg.__class__} {msg}')

        tasks: Iterator[Task] = planner.stream(messages)
        # Begin executing the planner immediately
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            # Handle the case where tasks is empty.
            tasks = iter([])
        executed_tasks = _schedule_tasks.invoke({"messages": messages, "tasks": tasks})
        return {"messages": executed_tasks}

    return plan_and_execute
