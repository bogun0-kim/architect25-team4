################################################################################
# LLM
################################################################################

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

# TODO: simple singleton instance
_LLM: BaseChatModel


class LLM:
    @staticmethod
    def set(llm_type: str, *args, **kwargs):
        global _LLM
        if llm_type == "GEMINI":
            _LLM = ChatGoogleGenerativeAI(*args, **kwargs)
        else:
            _LLM = ChatOpenAI(*args, **kwargs)

    @staticmethod
    def get():
        global _LLM
        return _LLM

    @staticmethod
    def name():
        global _LLM
        return _LLM.name
