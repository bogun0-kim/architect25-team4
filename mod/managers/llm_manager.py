################################################################################
# LLM
################################################################################

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI


# TODO: simple singleton instance
_LLM: BaseChatModel


class LLM:
    @staticmethod
    def set(*args, **kwargs):
        global _LLM
        _LLM = ChatOpenAI(*args, **kwargs)

    @staticmethod
    def get():
        global _LLM
        return _LLM

    @staticmethod
    def name():
        global _LLM
        return _LLM.name
