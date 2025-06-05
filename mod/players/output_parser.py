import ast
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
from typing_extensions import TypedDict
from langchain_core.exceptions import OutputParserException
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers.transform import BaseTransformOutputParser
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool


THOUGHT_PATTERN = r"Thought: ([^\n]*)"
ACTION_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
# ACTION_LIKE_PATTERN = r"\n*(\d+)\. (\w+)\((.*)\)(\s*#\w+\n)?"
ID_PATTERN = r"\$\{?(\d+)\}?"  # $1 or ${1} -> 1
END_OF_PLAN = ""
JOINER_TOOL_NAME = "join"


def _parse_string_arguments(raw_args: str, tool: BaseTool) -> Dict[str, Any]:
    """Parse arguments from a string."""

    # TODO: test
    if raw_args is None or raw_args == '':
        return {}
    if isinstance(tool, str):
        return {}
    arg_names = list(tool.args.keys())
    return {arg_names[0]: raw_args}

    if raw_args is None or raw_args == '':
        return {}

    tokens = []
    _current = []
    _bracket = 0
    for c in raw_args:
        if c in '{[(':
            _bracket += 1
        elif c in ')]}':
            _bracket -= 1
        if c == ',' and _bracket == 0:
            token = ''.join(_current).strip()
            if token:
                tokens.append(token)
            _current = []
        else:
            _current.append(c)
    if _current:
        tokens.append(''.join(_current).strip())

    kwargs = {}
    arg_names = list(tool.args.keys())
    for token in tokens:
        if '=' in token and not token.startswith('{') and not token.startswith('[') and not token.startswith('('):
            key, value = token.split('=', 1)
            key = key.strip()
            for k in [key, key.lower(), key.upper(), f'_{key}']:
                if k in arg_names:
                    arg_names.pop(arg_names.index(k))
                    key = k
                    break
            value = value.strip()
        else:
            key = arg_names.pop(0)
            value = token.strip()
        try:
            kwargs[key] = ast.literal_eval(value)
        except:  # noqa
            kwargs[key] = value
    return kwargs


def default_dependency_rule(idx: int, args: str):
    matches = re.findall(ID_PATTERN, args)
    numbers = [int(match) for match in matches]
    return idx in numbers


def _get_dependencies_from_graph(idx: int, tool_name: str, args: Dict[str, Any]) -> List[int]:
    """Get dependencies from a graph."""
    return list(range(1, idx)) if tool_name == JOINER_TOOL_NAME else [
        i for i in range(1, idx) if default_dependency_rule(i, str(args))]


class Task(TypedDict):
    idx: int
    tool: BaseTool
    args: Dict[str, Any]
    dependencies: list[int]
    thought: Optional[str]


def instantiate_task(
        idx: int,
        tool_name: str,
        tools: Dict[str, BaseTool],
        raw_args: Union[str, Any],
        thought: Optional[str] = None,
) -> Task:
    if tool_name == JOINER_TOOL_NAME:
        tool = JOINER_TOOL_NAME
        tool_args = {}
    else:
        try:
            tool = tools[tool_name]
        except KeyError as e:
            raise OutputParserException(
                f'Tool "{tool_name}" not found. (available={list(tools.keys())}') from e
        tool_args = _parse_string_arguments(raw_args, tool)
    dependencies = _get_dependencies_from_graph(idx, tool_name, tool_args)
    return Task(idx=idx, tool=tool, args=tool_args, dependencies=dependencies, thought=thought)


class LLMCompilerPlanParser(BaseTransformOutputParser[dict], extra="allow"):
    """Planning output parser."""

    tools: Dict[str, BaseTool]

    def stream(
            self,
            input_: str | BaseMessage,
            config: RunnableConfig | None = None,
            **kwargs: Any | None,
    ) -> Iterator[Task]:
        yield from self.transform([input_], config, **kwargs)

    def parse(self, text: str) -> List[Task]:
        return list(self._transform([text]))

    def _transform(self, input_: Iterator[Union[str, BaseMessage]]) -> Iterator[Task]:
        texts = []
        # TODO: Cleanup tuple state tracking here.
        thought = None
        for chunk in input_:
            # Assume input is str. TODO: support vision/other formats
            text = chunk if isinstance(chunk, str) else str(chunk.content)
            for task, thought in self._ingest_token(text, texts, thought):
                yield task
        # Final possible task
        if texts:
            task, _ = self._parse_task(''.join(texts), thought)
            if task:
                yield task

    def _ingest_token(
            self,
            token: str,
            buffer: List[str],
            thought: Optional[str],
    ) -> Iterator[Tuple[Optional[Task], str]]:
        buffer.append(token)
        if '\n' in token:
            buffer_ = ''.join(buffer).split("\n")
            suffix = buffer_[-1]
            for line in buffer_[:-1]:
                task, thought = self._parse_task(line, thought)
                if task:
                    yield task, thought
            buffer.clear()
            buffer.append(suffix)

    def _parse_task(self, line: str, thought: Optional[str] = None) -> tuple[Task, str]:
        task = None

        # Optionally, action can be preceded by a thought
        if match := re.match(THOUGHT_PATTERN, line):
            thought = match.group(1)
            print(f'# <_parse_task> THOUGHT: thought={thought}')

        # If action is parsed, return the task, and clear the buffer
        elif match := re.match(ACTION_PATTERN, line):
            idx, tool_name, raw_args, _ = match.groups()
            print(f'# <_parse_task> ACTION: idx={idx}, tool_name={tool_name}, args="{raw_args}"')

            task = instantiate_task(
                idx=int(idx),
                tool_name=tool_name,
                tools=self.tools,
                raw_args=raw_args,
                thought=thought)
            thought = None

        # Else it is just dropped
        else:
            print(f'# <_parse_task> NOTHING: line={line}')

        return task, thought
