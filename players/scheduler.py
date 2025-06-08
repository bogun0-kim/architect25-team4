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
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue, PromptValue, StringPromptValue
from langchain_core.runnables import chain as as_runnable
from langchain_core.runnables.base import Runnable
from .output_parser import Task


class SchedulerInput(TypedDict):
    model: BaseChatModel
    user_query: str
    messages: List[BaseMessage]
    tasks: Iterator[Task]


class ExecutorInput(TypedDict):
    model: Optional[BaseChatModel]
    user_query: str
    task: Task
    observations: Dict[int, Any]


def _get_observations(messages: List[BaseMessage]) -> Dict[int, Any]:
    # Get all previous tool responses
    results = {}
    for message in messages[::-1]:
        if isinstance(message, FunctionMessage):
            results[int(message.additional_kwargs["idx"])] = message.content
    return results


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, Any]):
    # $1 or ${1} -> 1
    _id_pattern = r"\$\{?(\d+)\}?"

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
        return re.sub(_id_pattern, replace_match, arg)
    elif isinstance(arg, (list, tuple)):
        return [_resolve_arg(a, observations) for a in arg]
    else:
        return str(arg)


def _resolve_arg2(arg: Union[str, Any], observations: Dict[int, Any], model: BaseChatModel, user_query: str):
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [_resolve_arg2(v, observations, model, user_query) for v in arg]
    if isinstance(arg, dict):
        return {k: _resolve_arg2(v, observations, model, user_query) for k, v in arg.items()}
    if not isinstance(arg, str):
        return arg

    # TODO: test
    _ID_PATTERN = r'\$\{?(\d+)\}?'  # $1 or ${1} -> 1
    for m in re.compile(_ID_PATTERN).finditer(arg):
        arg = arg.replace(m.group(0), observations.get(int(m.group(1)), m.group(0)))
    return arg

    var_idx_ref: list[list[str, str, str]] = [
        [m.group(0), m.group(1), observations.get(int(m.group(1)), m.group(0))]
        for m in re.compile(_ID_PATTERN).finditer(arg)]
    if len(var_idx_ref) == 0:
        return arg

    _RESOLVER_PROMPT_TEMPLATE = f"""
<QUERY>{user_query}</QUERY>
{{references}}
<PROBLEM>{{problem}}</PROBLEM>

PROBLEM contains one or more variables in the format of $ID.
Each variable $ID must be replaced with a constant value extracted exclusively from REFERENCE-ID that matches the same number.
For example, $1 must be filled using information from REFERENCE-1, and $3 must be filled using information from REFERENCE-3.
First, understand the contents of the QUERY, and based on that, determine which information from each REFERENCE should be used.
Then, replace each $ID with the appropriate constant value to complete the PROBLEM.
Only submit the completed PROBLEM as your answer. Do not include any explanations, comments, or additional text.
Now, complete the PROBLEM in its final form.
    """.strip()
    _REFERENCE_TEMPLATE = '<REFERENCE-{idx}>{content}</REFERENCE-{idx}>'

    print(f'@@ {__file__} >> _resolve_arg2: arg={arg}')
    try:
        resolver_prompt = _RESOLVER_PROMPT_TEMPLATE.format(
            references='\n'.join([_REFERENCE_TEMPLATE.format(idx=i, content=r) for v, i, r in var_idx_ref]),
            problem=arg)
        resolved_arg = model.invoke(ChatPromptValue(messages=[HumanMessage(resolver_prompt)]))
        print(f'@@ {__file__} >> _resolve_arg2: resolved_arg={resolved_arg}')
        return resolved_arg.content
    except Exception as e:
        for v, i, r in var_idx_ref:
            arg = arg.replace(v, r)
        print(f'@@ {__file__} >> _resolve_arg2: replaced={arg} / {e}')
        return arg


def _execute_task(
        task: Task,
        observations,
        config,
        model: BaseChatModel = None,
        user_query: str = None,
):
    tool = task["tool"]
    args = task["args"]
    print(f'@@ {__file__} >> _execute_task: tool={type(tool)}, {tool}')
    if isinstance(tool, str):
        return tool

    ################################################################################
    # TODO: Resolve arguments
    ################################################################################
    print(f'@@ {__file__} >> _execute_task: args={type(args)}, {args}')
    try:
        # if args is None:
        #     resolved_args = None
        # elif isinstance(args, str):
        #     resolved_args = _resolve_arg(args, observations)
        # elif isinstance(args, (list, tuple)):
        #     resolved_args = [_resolve_arg(v, observations) for v in args]
        # elif isinstance(args, dict):
        #     resolved_args = {k: _resolve_arg(v, observations) for k, v in args.items()}
        # else:
        #     # This will likely fail
        #     resolved_args = args
        resolved_args = _resolve_arg2(args, observations, model, user_query)
    except Exception as e:
        print(f'@@ {__file__} >> _execute_task: failed to resolve args. {e}')
        return (
            f'ERROR'
            f' (Failed to call {tool.name} with args {args}.'
            f' Args could not be resolved. Error: {repr(e)})')
    print(f'@@ {__file__} >> _execute_task: resolved_args={type(resolved_args)}, {resolved_args}')
    ################################################################################

    ################################################################################
    # Execute tool
    ################################################################################
    print(f'@@ {__file__} >> _execute_task: execute> {type(tool)}.invoke({resolved_args}, config={config})')
    try:
        output = tool.invoke(resolved_args, config)
    except Exception as e:
        print(f'@@ {__file__} >> _execute_task: failed to invoke. {e}')
        return (
            f'ERROR'
            f' (Failed to call {tool.name} with args {args}.'
            f' Args resolved to {resolved_args}. Error: {repr(e)})')
    print(f'@@ {__file__} >> _execute_task: tool_output> {output}')
    # return output
    ################################################################################

    ################################################################################
    # TODO: Clean the result of tool
    ################################################################################
    query = None
    try:
        if isinstance(resolved_args, str):
            if resolved_args.strip() != '':
                query = resolved_args
        elif isinstance(resolved_args, (list, tuple)):
            if len(resolved_args) > 0:
                query = '\n'.join(resolved_args)
        elif isinstance(resolved_args, dict):
            if len(resolved_args) == 1:
                query = resolved_args[list(resolved_args)[0]]
            elif len(resolved_args) > 1:
                query = '\n'.join([f'{k}={v}' for k, v in resolved_args.items()])
    except Exception:
        return output
    if query is None:
        return output

    print(f'@@ {__file__} >> _execute_task: clean> query={query}, response={output}')
    try:
        cleaner_prompt_template = """
<QUERY>{query}<QUERY>
<RESPONSE>{response}</RESPONSE>

Find the correct answer corresponding to the QUERY from the RESPONSE.
Do not include any additional explanations, ONLY provide the answer.
        """.strip()
        cleaned_output = model.invoke(ChatPromptValue(messages=[
            HumanMessage(cleaner_prompt_template.format(query=query, response=str(output)))]))
        print(f'@@ {__file__} >> _execute_task: cleaned_output> {cleaned_output}')
        return cleaned_output.content
    except Exception as e:
        print(f'@@ {__file__} >> _execute_task: not cleaned. {e}')
        return output


@as_runnable
def _schedule_task(executor_input: ExecutorInput, config):
    task: Task = executor_input["task"]
    observations: Dict[int, Any] = executor_input["observations"]
    try:
        observation = _execute_task(
            task, observations, config,
            executor_input.get("model", None),
            executor_input.get("user_query", None))
    except Exception as e:
        import traceback
        observation = traceback.format_exception(e)
    observations[task["idx"]] = observation


def _schedule_pending_task(
        task: Task,
        observations: Dict[int, Any],
        model: BaseChatModel,
        user_query: str,
        retry_after: float = 0.2,
):
    while True:
        # Dependencies not yet satisfied
        dependencies = task["dependencies"]
        if dependencies and (any([d not in observations for d in dependencies])):
            time.sleep(retry_after)
            continue

        _schedule_task.invoke(ExecutorInput(
            task=task, observations=observations,
            model=model, user_query=user_query))
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
    user_query = scheduler_input["user_query"]
    messages = scheduler_input["messages"]
    tasks = scheduler_input["tasks"]
    model = scheduler_input["model"]
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
                futures.append(executor.submit(
                    _schedule_pending_task, task, observations, model, user_query, retry_after))

            # No dependencies or all dependencies satisfied, can schedule now
            else:
                _schedule_task.invoke(ExecutorInput(
                    task=task, observations=observations,
                    model=model, user_query=user_query))
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
