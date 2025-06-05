################################################################################
# Tools
################################################################################

from langchain_core.tools import BaseTool


# TODO: simple memory DB
_DATA: dict[str, BaseTool] = {}


class ToolManager:
    @staticmethod
    def set(tool: BaseTool, name: str = None):
        global _DATA
        _DATA[name or tool.name] = tool

    @staticmethod
    def get(tool: str | BaseTool, default=None) -> BaseTool | None:
        global _DATA
        return _DATA.get(tool.name if isinstance(tool, BaseTool) else str(tool), default)

    @staticmethod
    def pop(tool: str | BaseTool) -> BaseTool | None:
        global _DATA
        return _DATA.pop(tool.name if isinstance(tool, BaseTool) else str(tool), None)

    @staticmethod
    def data() -> dict[str, BaseTool]:
        global _DATA
        return dict(_DATA)
