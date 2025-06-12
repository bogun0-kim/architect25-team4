################################################################################
# LLM
################################################################################

from langchain_core.language_models import BaseChatModel


# TODO: simple singleton instance
_LLM: BaseChatModel


class LLM:
    @staticmethod
    def set(llm: BaseChatModel):
        global _LLM
        _LLM = llm

    @staticmethod
    def get() -> BaseChatModel:
        global _LLM
        return _LLM

    @staticmethod
    def name() -> str:
        global _LLM
        return _LLM.name
