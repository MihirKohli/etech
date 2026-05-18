"""
LLM factory — returns a LangChain chat model based on config.

Supports OpenAI and Groq out of the box.  Add more providers
by extending the if/elif chain.
"""

from langchain_core.language_models import BaseChatModel

from config import get_settings
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI


def get_llm() -> BaseChatModel:
    """Return a configured LangChain chat model."""
    settings = get_settings()

    if settings.LLM_PROVIDER == "openai":
        return ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.OPENAI_API_KEY,
        )
    elif settings.LLM_PROVIDER == "groq":
        return ChatGroq(
            model=settings.LLM_MODEL,
            temperature=settings.LLM_TEMPERATURE,
            api_key=settings.GROQ_API_KEY,
        )
    else:
        raise ValueError(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")