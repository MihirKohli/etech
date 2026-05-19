"""
LLM factory — returns a LangChain chat model based on config.

Supports OpenAI and Groq out of the box.  Add more providers
by extending the if/elif chain.
"""

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from config import get_settings

def get_openai_llm() -> BaseChatModel:
    settings = get_settings()
    return ChatOpenAI(
        model=settings.LLM_MODEL,
        temperature=settings.LLM_TEMPERATURE,
        api_key=settings.OPENAI_API_KEY,
    )


