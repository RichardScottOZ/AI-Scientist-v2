import time
import os
import logging
import json
from typing import Optional

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
from rich import print

logger = logging.getLogger("ai-scientist")

_client: openai.OpenAI = None  # type: ignore

OPENROUTER_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

# Model name mapping for OpenRouter
MODEL_MAPPING = {
    "llama3.1-405b": "meta-llama/llama-3.1-405b-instruct",
    "gpt-4o": "openai/gpt-4o",
    "gpt-4": "openai/gpt-4",
    "gpt-4-turbo": "openai/gpt-4-turbo-preview",
    "gpt-3.5-turbo": "openai/gpt-3.5-turbo",
    "gpt-4o-2024-11-20": "openai/gpt-4o-2024-11-20",
    "o3-mini": "openai/o3-mini",
    "claude-3-opus": "anthropic/claude-3-opus",
    "claude-3-sonnet": "anthropic/claude-3-sonnet",
    "claude-3-haiku": "anthropic/claude-3-haiku",
    "claude-3-5-sonnet-OR": "anthropic/claude-3.5-sonnet",
    # DeepSeek models
    "deepseek-coder": "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-coder-33b": "deepseek-ai/deepseek-coder-33b-instruct",
    "deepseek-coder-6.7b": "deepseek-ai/deepseek-coder-6.7b-instruct",
    "deepseek-coder-1.3b": "deepseek-ai/deepseek-coder-1.3b-instruct",
    "deepcoder-14b-preview": "agentica-org/deepcoder-14b-preview",
    "deepcoder-14b-preview:free": "agentica-org/deepcoder-14b-preview:free",
    "deepseek-llm": "deepseek-ai/deepseek-llm-67b-chat",
    "deepseek-llm-67b": "deepseek-ai/deepseek-llm-67b-chat",
    "deepseek-llm-7b": "deepseek-ai/deepseek-llm-7b-chat",
    "deepseek-chat-v3-0324": "deepseek/deepseek-chat-v3-0324",
    # Google Gemma models
    "gemma-7b": "google/gemma-7b-it",
    "gemma-2b": "google/gemma-2b-it",
    "gemma-7b-instruct": "google/gemma-7b-it",
    "gemma-2b-instruct": "google/gemma-2b-it"
    # Add more mappings as needed
}

@once
def _setup_openrouter_client():
    global _client
    # OpenRouter uses the OpenAI client with a different base URL and headers
    _client = openai.OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        max_retries=0,
        default_headers={
            "HTTP-Referer": os.getenv("OPENROUTER_REFERRER", "http://localhost:3000"),
            "X-Title": os.getenv("OPENROUTER_TITLE", "AI Scientist")
        }
    )

def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    _setup_openrouter_client()
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    # Map the model name to OpenRouter's format
    if "model" in filtered_kwargs:
        model_name = filtered_kwargs["model"]
        if model_name in MODEL_MAPPING:
            filtered_kwargs["model"] = MODEL_MAPPING[model_name]
        else:
            logger.warning(f"Model {model_name} not found in mapping, using as-is")

    messages = opt_messages_to_list(system_message, user_message)
    
    # Ensure messages are properly formatted for OpenAI API
    formatted_messages = []
    for msg in messages:
        if isinstance(msg, dict):
            # Convert dict to proper message format
            formatted_msg = {
                "role": msg.get("role", "user"),
                "content": msg.get("content", "")
            }
            formatted_messages.append(formatted_msg)
        else:
            # If it's already a string, assume it's user content
            formatted_messages.append({
                "role": "user",
                "content": str(msg)
            })

    # If function calling is requested but not supported, modify the prompt to get JSON output
    if func_spec is not None:
        try:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict
        except Exception as e:
            logger.warning(f"Tool use not supported, falling back to JSON completion: {e}")
            # Add instruction to output JSON
            if user_message:
                user_message = f"{user_message}\n\nPlease respond with a JSON object that matches this schema: {json.dumps(func_spec.json_schema)}"
            else:
                user_message = f"Please respond with a JSON object that matches this schema: {json.dumps(func_spec.json_schema)}"
            messages = opt_messages_to_list(system_message, user_message)
            func_spec = None

    t0 = time.time()
    try:
        # Log the full request details
        logger.info(f"Sending request to OpenRouter with messages: {formatted_messages}")
        logger.info(f"Request kwargs: {filtered_kwargs}")
        logger.info(f"Model being used: {filtered_kwargs.get('model', 'unknown')}")
        
        # Ensure messages are properly formatted
        for msg in formatted_messages:
            if not isinstance(msg.get('content', ''), str):
                msg['content'] = str(msg['content'])
            if not isinstance(msg.get('role', ''), str):
                msg['role'] = str(msg['role'])
        
        completion = backoff_create(
            _client.chat.completions.create,
            OPENROUTER_TIMEOUT_EXCEPTIONS,
            messages=formatted_messages,
            **filtered_kwargs,
        )
    except Exception as e:
        logger.error(f"OpenRouter API call failed with error: {str(e)}")
        if hasattr(e, 'response'):
            try:
                error_content = e.response.content
                if isinstance(error_content, bytes):
                    error_content = error_content.decode('utf-8')
                logger.error(f"Response content: {error_content}")
                # Try to parse error details
                try:
                    error_json = json.loads(error_content)
                    if isinstance(error_json, dict):
                        error_msg = error_json.get('error', {}).get('message', 'Unknown error')
                        error_type = error_json.get('error', {}).get('type', 'Unknown type')
                        error_code = error_json.get('error', {}).get('code', 'Unknown code')
                        logger.error(f"Error details - Type: {error_type}, Code: {error_code}, Message: {error_msg}")
                except json.JSONDecodeError:
                    pass
            except Exception as parse_error:
                logger.error(f"Error parsing response: {str(parse_error)}")
        raise

    req_time = time.time() - t0

    if completion is None:
        logger.error("OpenRouter API call returned None")
        raise ValueError("OpenRouter API call failed - no response received")

    # Add detailed logging of the response
    logger.info(f"OpenRouter response type: {type(completion)}")
    logger.info(f"OpenRouter response attributes: {dir(completion)}")
    logger.info(f"OpenRouter response: {completion}")
    
    # Check for error in response
    if hasattr(completion, 'error'):
        error_msg = completion.error.get('message', 'Unknown error')
        error_code = completion.error.get('code', 'Unknown code')
        error_type = completion.error.get('type', 'Unknown type')
        logger.error(f"OpenRouter returned error: {error_msg} (code: {error_code}, type: {error_type})")
        if hasattr(completion.error, 'metadata') and 'raw' in completion.error.metadata:
            try:
                raw_error = completion.error.metadata['raw']
                if isinstance(raw_error, str):
                    error_json = json.loads(raw_error)
                    if isinstance(error_json, dict):
                        provider_error = error_json.get('error', {})
                        logger.error(f"Provider error details: {provider_error}")
            except Exception as e:
                logger.error(f"Error parsing raw error: {str(e)}")
        raise ValueError(f"OpenRouter API error: {error_msg}")

    # Handle different response formats
    if hasattr(completion, 'choices'):
        choices = completion.choices
        logger.info(f"Found choices in response: {choices}")
    elif hasattr(completion, 'completion'):
        # Handle Claude's format
        logger.info(f"Found completion in response: {completion.completion}")
        choices = [{'message': {'content': completion.completion}}]
    else:
        logger.error(f"Unexpected response format: {completion}")
        raise ValueError("OpenRouter API call failed - unexpected response format")

    if not choices:
        logger.error("Response has empty choices array")
        raise ValueError("OpenRouter API call failed - empty choices array")

    choice = choices[0] if isinstance(choices, list) else choices
    logger.info(f"Processing choice: {choice}")
    
    # Handle different choice formats
    if isinstance(choice, dict):
        message = choice.get('message', {})
        logger.info(f"Found message in choice dict: {message}")
        if not message:
            logger.error(f"Invalid choice format: {choice}")
            raise ValueError("OpenRouter API call failed - invalid choice format")
    else:
        if not hasattr(choice, 'message'):
            logger.error(f"Choice missing 'message' attribute: {choice}")
            raise ValueError("OpenRouter API call failed - choice missing 'message' attribute")
        message = choice.message
        logger.info(f"Found message in choice object: {message}")

    # Get content from message
    if isinstance(message, dict):
        content = message.get('content', '')
        logger.info(f"Content from message dict: {content}")
    else:
        content = getattr(message, 'content', '')
        logger.info(f"Content from message object: {content}")

    # Check for function calls/tool calls
    if isinstance(message, dict):
        tool_calls = message.get('tool_calls', [])
    else:
        tool_calls = getattr(message, 'tool_calls', [])
    
    logger.info(f"Found tool calls: {tool_calls}")

    # Get token counts
    if hasattr(completion, 'usage'):
        in_tokens = completion.usage.prompt_tokens
        out_tokens = completion.usage.completion_tokens
    else:
        # Estimate tokens if not provided
        in_tokens = len(str(formatted_messages)) // 4
        out_tokens = len(str(content)) // 4

    info = {
        "system_fingerprint": getattr(completion, 'system_fingerprint', None),
        "model": getattr(completion, 'model', 'unknown'),
        "created": getattr(completion, 'created', None),
    }

    # If we have tool calls, use those instead of content
    if tool_calls:
        logger.info("Using tool calls instead of content")
        try:
            # Get the first tool call's function arguments
            if isinstance(tool_calls[0], dict):
                func_args = tool_calls[0].get('function', {}).get('arguments', '{}')
            else:
                func_args = tool_calls[0].function.arguments
            
            logger.info(f"Function arguments: {func_args}")
            output = json.loads(func_args)
            return output, req_time, in_tokens, out_tokens, info
        except Exception as e:
            logger.error(f"Error parsing function arguments: {e}")
            raise

    # Only check for empty content if we don't have tool calls
    if not content and not tool_calls:
        logger.error("Empty content in response and no tool calls")
        logger.error(f"Full message structure: {message}")
        raise ValueError("OpenRouter API call failed - empty content in response")

    # Handle function calls if needed
    if func_spec is not None:
        if not tool_calls:
            logger.error("Function call requested but no tool_calls in response")
            raise ValueError("OpenRouter API call failed - no tool_calls in response")
            
        try:
            if isinstance(tool_calls[0], dict):
                output = json.loads(tool_calls[0].get('function', {}).get('arguments', '{}'))
            else:
                output = json.loads(tool_calls[0].function.arguments)
        except Exception as e:
            logger.error(f"Error parsing function arguments: {e}")
            raise
    else:
        output = content
        try:
            output = json.loads(output)
        except json.JSONDecodeError:
            pass

    return output, req_time, in_tokens, out_tokens, info 