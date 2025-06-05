# weather_server.py
from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather", port=8001)

@mcp.tool()
async def get_weather(location: str) -> str:
    """현재 날씨를 조회합니다."""
    return "It's always sunny in " + location

@mcp.tool()
async def get_temperature(location: str) -> str:
    """현재 온도를 조회합니다."""
    return f"The current temperature in {location} is 23°C."

@mcp.tool()
async def get_fine_dust_level(location: str) -> str:
    """미세먼지 농도를 조회합니다."""
    return f"The fine dust level in {location} is 42 µg/m³, which is considered moderate."

@mcp.tool()
async def get_precipitation_chance(location: str) -> str:
    """강수 확률을 조회합니다."""
    return f"The chance of precipitation in {location} today is 20%."

if __name__ == "__main__":
    mcp.run(transport="streamable-http")