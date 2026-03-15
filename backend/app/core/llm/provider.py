from langchain.chat_models import init_chat_model
from langchain_core.language_models import BaseChatModel

from app.config import settings


def get_chat_model(
    model: str | None = None,
    temperature: float | None = None,
) -> BaseChatModel:
    """Initialize LLM with runtime provider switching.

    Model format: 'provider:model_name'
    Examples:
        - 'openai:gpt-4.1'
        - 'anthropic:claude-sonnet-4-6'
        - 'ollama:llama3'
    """
    model = model or settings.default_model
    temperature = temperature if temperature is not None else settings.default_temperature
    return init_chat_model(model=model, temperature=temperature)
