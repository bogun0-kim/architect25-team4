################################################################################
# Task Fetching Unit: plan and schedule
################################################################################

import re
import itertools
import asyncio
from typing import Any, Dict, Iterator, List, Union, Optional
from typing_extensions import TypedDict
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate
from langchain_core.runnables import chain as as_runnable
from langchain_core.runnables.base import Runnable
from .output_parser import Task


# TODO:
_RESPONSE_RESOLVER_PROMPT_TEMPLATE = """
<QUERY>{query}<QUERY>
<REFERENCE>{reference}</REFERENCE>

Find the correct answer corresponding to the QUERY from the REFERENCE.
Do not include any additional explanations, ONLY provide the answer.""".strip()


class SchedulerInput(TypedDict):
    messages: List[BaseMessage]
    tasks: Iterator[Task]
    model: BaseChatModel


class ExecutorInput(TypedDict):
    task: Task
    observations: Dict[int, ToolMessage]
    model: Optional[BaseChatModel]


def _get_observations(messages: List[BaseMessage]) -> Dict[int, ToolMessage]:
    # Get all previous tool responses
    return {int(msg.additional_kwargs["idx"]): msg for msg in messages[::-1] if isinstance(msg, ToolMessage)}


def _resolve_arg(arg: Union[str, Any], observations: Dict[int, ToolMessage]):
    if arg is None:
        return None
    if isinstance(arg, (list, tuple)):
        return [_resolve_arg(v, observations) for v in arg]
    if isinstance(arg, dict):
        return {k: _resolve_arg(v, observations) for k, v in arg.items()}
    if not isinstance(arg, str):
        return arg

    _ID_PATTERN = r'\$\{?(\d+)\}?'  # $1 or ${1} -> 1
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


async def _execute_with_retry(coroutine, *args, max_attempts: int = 3, retry_delay_seconds: float = 0.2, **kwargs):
    for attempt in range(1, max_attempts + 1):
        try:
            return await coroutine(*args, **kwargs), None
        except Exception as e:
            print(f'# [{attempt}/{max_attempts}] Failed to call: {coroutine} (Exception={e})')
            if max_attempts <= attempt:
                return None, e
            await asyncio.sleep(retry_delay_seconds)


async def _resolve_response(resolved_args, response, model: BaseChatModel):
    # TODO: to Runnable?

    def _query(value, sep=',') -> Union[str, None]:
        if value is None:
            return None
        if isinstance(value, str):
            return None if (q := value.strip()) == '' else q
        if isinstance(value, (list, tuple)):
            value = [q for v in value if (q := _query(v, sep)) is not None]
            return None if len(value) == 0 else sep.join(value)
        if isinstance(value, dict):
            value = {k: q for k, v in value.items() if (q := _query(v, sep)) is not None}
            if len(value) == 0:
                return None
            elif len(value) == 1:
                return str(value[list(value)[0]])
            else:
                return sep.join([f'{k}={v}' for k, v in value.items()])
        return _query(str(value))

    query = _query(resolved_args)
    response = str(response).strip()
    if query is not None and response != '':
        try:
            print(f'@@ [resolve_response] query={query}, response={response}')
            prompt = ChatPromptTemplate.from_messages([
                HumanMessagePromptTemplate.from_template(_RESPONSE_RESOLVER_PROMPT_TEMPLATE)])
            parser = StrOutputParser()
            _chain = prompt | model | parser
            resolved = await _chain.ainvoke({"query": query, "reference": response})
            print(f'@@ [resolve_response] resolved={resolved}')
            return resolved
        except Exception as e:
            print(f'@@ [resolve_response] failed: {e}')
    return None


async def _execute_task(
        task: Task,
        observations: Dict[int, ToolMessage],
        config,
        model: BaseChatModel,
) -> List[str]:
    tool = task["tool"]
    print(f'@@ [execute_task] tool={type(tool)}, {tool}')
    if isinstance(tool, str):
        return [tool]

    args = task["args"]
    print(f'@@ [execute_task] args={type(args)}, {args}')
    try:
        resolved_args = _resolve_arg(args, observations)
    except Exception as e:
        return [
            f'ERROR'
            f' (Failed to call {tool.name} with args {args}.'
            f' Args could not be resolved. Error: {repr(e)})']
    print(f'@@ [execute_task] resolved_args={type(resolved_args)}, {resolved_args}')

    print(f'@@ [execute_task] execute: {type(tool)}.ainvoke({resolved_args}, config={config})')
    response, error = await _execute_with_retry(tool.ainvoke, resolved_args, config)
    if error is not None:
        return [
            f'ERROR'
            f' (Failed to call {tool.name} with args {args}.'
            f' Args resolved to {resolved_args}. Error: {repr(error)})']
    print(f'@@ [execute_task] tool_response={response}')

    resolved_response = await _resolve_response(resolved_args, response, model)
    return [response] if resolved_response is None else [response, resolved_response]


@as_runnable
async def _schedule_task(executor_input: ExecutorInput, config):
    task: Task = executor_input["task"]
    observations: Dict[int, ToolMessage] = executor_input["observations"]
    model: BaseChatModel = executor_input["model"]
    try:
        outputs = await _execute_task(task, observations, config, model)
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


async def _schedule_pending_task(
        task: Task,
        observations: Dict[int, ToolMessage],
        model: BaseChatModel,
        retry_delay_seconds: float = 0.2,
        max_wait_seconds_per_dependency: float = 3.0,
):
    dependencies = task["dependencies"]
    if dependencies:
        max_wait_seconds = max_wait_seconds_per_dependency * len(dependencies)
        waited_seconds = 0.0
        while max_wait_seconds < 0.0 or waited_seconds < max_wait_seconds:
            # Dependencies not yet satisfied
            if any([d not in observations for d in dependencies]):
                await asyncio.sleep(retry_delay_seconds)
                waited_seconds += retry_delay_seconds
                continue

    await _schedule_task.ainvoke(
        ExecutorInput(task=task, observations=observations, model=model))


# 제한 병렬도 (최적값은 실험 필요, 일반적으로 5~10)
MAX_PARALLEL = 5
semaphore = asyncio.Semaphore(MAX_PARALLEL)


@as_runnable
async def _schedule_tasks(scheduler_input: SchedulerInput) -> List[ToolMessage]:
    messages = scheduler_input["messages"]
    model = scheduler_input["model"]
    _task_names = {}
    _task_args = {}

    observations = _get_observations(messages)
    original_keys = set(observations)

    retry_after = 0.2
    tasks = []

    async def schedule_with_semaphore(coro):
        async with semaphore:
            return await coro

    for task in scheduler_input["tasks"]:
        deps = task["dependencies"]
        _task_names[task["idx"]] = (
            task["tool"] if isinstance(task["tool"], str) else task["tool"].name
        )
        _task_args[task["idx"]] = task["args"]

        if deps and any([dep not in observations for dep in deps]):
            print('??????????????????????????????????????????????????????????????')
            task_coro = _schedule_pending_task(task, observations, model, retry_after)
        else:
            task_coro = _schedule_task.ainvoke(dict(task=task, observations=observations, model=model))

        # 병렬 스케줄링
        tasks.append(asyncio.create_task(schedule_with_semaphore(task_coro)))

    await asyncio.gather(*tasks)

    # 결과 정리
    tool_messages = []
    for idx in sorted(observations.keys() - original_keys):
        try:
            tool_call = {"id": str(idx), "function": {"name": _task_names[idx], "arguments": str(_task_args[idx])}}
            msg = AIMessage(content='', additional_kwargs={"tool_calls": [tool_call]})
        except Exception:
            tool_call = {"id": str(idx), "function": {"name": _task_names[idx], "arguments": [str(_task_args[idx])]}}
            msg = AIMessage(content='', additional_kwargs={"tool_calls": [tool_call]})
        tool_messages.append(msg)
        tool_messages.append(observations[idx])
    return tool_messages


def build(planner: Runnable, model: BaseChatModel) -> Runnable:

    @as_runnable
    async def plan_and_execute(state):
        for msg in state["messages"]:
            print(f'@@ [plan_and_execute] {msg.__class__.__name__}={msg}')

        messages = state["messages"]
        tasks: Iterator[Task] = planner.stream(messages)
        # Begin executing the planner immediately
        try:
            tasks = itertools.chain([next(tasks)], tasks)
        except StopIteration:
            # Handle the case where tasks is empty.
            tasks = iter([])
        print(f'@@ [plan_and_execute] tasks={tasks}')

        # Execute tasks
        executed_tasks = await _schedule_tasks.ainvoke(
            SchedulerInput(messages=messages, tasks=tasks, model=model))
        print(f'@@ [plan_and_execute] executed_tasks={executed_tasks}')

        return {"messages": executed_tasks}

    return plan_and_execute
