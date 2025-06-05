from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field
from typing import Literal, Optional, List

# ----------------------------------
# 입력 스키마 정의
# ----------------------------------

class WeatherInput(BaseModel):
    location: str = Field(..., description="The location to get the weather info for")
    func: Literal["temperature", "particular", "precipitation"] = Field(
        ..., description="The type of weather info to retrieve: 'temperature', 'particular' (fine dust), or 'precipitation'"
    )
    context: Optional[List[str]] = Field(default=[], description="Optional context from other tools")

# ----------------------------------
# 단일 툴 함수 정의
# ----------------------------------

def get_weather_info(location: str, func: str, context: Optional[List[str]] = []) -> int:
    if func == "temperature":
        return 23  # 섭씨
    elif func == "particular":
        return 42  # µg/m³
    elif func == "precipitation":
        return 20  # %
    else:
        raise ValueError(f"Invalid func value: {func}. Must be one of: temperature, particular, precipitation.")

# ----------------------------------
# StructuredTool 생성
# ----------------------------------

weather_tool = StructuredTool.from_function(
    name="get_weather_info",
    func=get_weather_info,
    description=(
        "get_weather_info(location: str, func: Literal['temperature', 'particular', 'precipitation'], context: Optional[list[str]]) -> int:\n"
        " - Retrieves mock weather information for the given location.\n"
        " - `func` determines what to retrieve: 'temperature' for current temperature in Celsius (int), "
        "'particular' for fine dust level in µg/m³ (int), or 'precipitation' for chance of rain in % (int).\n"
        " - You MUST call this tool only once per type of weather data. For example, you cannot call `get_weather_info('Seoul', 'temperature, precipitation')`. "
        "Instead, call `get_weather_info('Seoul', 'temperature')` and then `get_weather_info('Seoul', 'precipitation')` separately.\n"
        " - Minimize the number of `get_weather_info` calls by grouping what you need logically. For example, if all values are needed, call them individually but only once per type.\n"
        " - You can optionally provide a list of strings as `context` to clarify any ambiguity (e.g., time of day, elevation, past weather).\n"
        " - This tool does NOT retain the output of previous calls. If chaining values (e.g., using temperature in math), you MUST explicitly pass prior outputs via `context`.\n"
        " - You MUST NEVER treat `search`-type tool outputs as inputs for `get_weather_info`. If needed, extract values or use them in `context` only.\n"
        " - Always specify the units you expect when asking about weather. For example, ask 'what is the temperature in Celsius' instead of just 'what is the temperature'.\n"
    ),
    args_schema=WeatherInput,
)

def get_weather_tool() -> StructuredTool:
    return weather_tool
