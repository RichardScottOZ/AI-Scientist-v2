from . import backend_anthropic, backend_openai, backend_openrouter
from .utils import FunctionSpec, OutputType, PromptType, compile_prompt_to_md

# Global variable to track which backend to use
_current_backend = None

def query(
    system_message: PromptType | None,
    user_message: PromptType | None,
    model: str,
    temperature: float | None = None,
    max_tokens: int | None = None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> OutputType:
    """
    General LLM query for various backends with a single system and user message.
    Supports function calling for some backends.

    Args:
        system_message (PromptType | None): Uncompiled system message (will generate a message following the OpenAI/Anthropic format)
        user_message (PromptType | None): Uncompiled user message (will generate a message following the OpenAI/Anthropic format)
        model (str): string identifier for the model to use (e.g. "gpt-4-turbo")
        temperature (float | None, optional): Temperature to sample at. Defaults to the model-specific default.
        max_tokens (int | None, optional): Maximum number of tokens to generate. Defaults to the model-specific max tokens.
        func_spec (FunctionSpec | None, optional): Optional FunctionSpec object defining a function call. If given, the return value will be a dict.

    Returns:
        OutputType: A string completion if func_spec is None, otherwise a dict with the function call details.
    """

    global _current_backend

    model_kwargs = model_kwargs | {
        "model": model,
        "temperature": temperature,
    }

    # Handle models with beta limitations
    # ref: https://platform.openai.com/docs/guides/reasoning/beta-limitations
    if model.startswith("o1"):
        if system_message and user_message is None:
            user_message = system_message
        elif system_message is None and user_message:
            pass
        elif system_message and user_message:
            system_message["Main Instructions"] = {}
            system_message["Main Instructions"] |= user_message
            user_message = system_message
        system_message = None
        # model_kwargs["temperature"] = 0.5
        model_kwargs["reasoning_effort"] = "high"
        model_kwargs["max_completion_tokens"] = 100000  # max_tokens
        # remove 'temperature' from model_kwargs
        model_kwargs.pop("temperature", None)
    else:
        model_kwargs["max_tokens"] = max_tokens

    # Choose the appropriate backend based on the model
    # First check if it's an OpenRouter model
    if model in backend_openrouter.MODEL_MAPPING or any(
        model.startswith(prefix) for prefix in ["claude-", "gpt-", "llama", "deepseek"]
    ):
        query_func = backend_openrouter.query
    elif "claude-" in model:
        query_func = backend_anthropic.query
    elif model.startswith("gpt-") or model.startswith("o1"):
        query_func = backend_openai.query
    else:
        raise ValueError(f"Model {model} not supported.")

    output, req_time, in_tok_count, out_tok_count, info = query_func(
        system_message, user_message, func_spec, **model_kwargs
    )

    return output
