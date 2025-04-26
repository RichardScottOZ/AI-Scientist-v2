import json
import os
import re
from typing import Any
from ai_scientist.utils.token_tracker import track_token_usage

import anthropic
import backoff
import openai
import google.generativeai as genai

MAX_NUM_TOKENS = 4096

AVAILABLE_LLMS = [
    "claude-3-5-sonnet-20240620",
    "claude-3-5-sonnet-20241022",
    # OpenAI models
    "gpt-4o-mini",
    "gpt-4o-mini-2024-07-18",
    "gpt-4o",
    "gpt-4o-2024-05-13",
    "gpt-4o-2024-08-06",
    "gpt-4o-2024-11-20",
    "gpt-4.1",
    "gpt-4.1-2025-04-14",
    "gpt-4.1-mini",
    "gpt-4.1-mini-2025-04-14",
    "o1",
    "o1-2024-12-17",
    "o1-preview-2024-09-12",
    "o1-mini",
    "o1-mini-2024-09-12",
    "o3-mini",
    "o3-mini-2025-01-31",
    # Google Gemini models
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-pro-OR",
    "gemini-1.5-pro-latest-OR",
    # DeepSeek Models
    "deepseek-coder-v2-0724",
    "deepcoder-14b",
    # Llama 3 models
    "llama3.1-405b",
    # Anthropic Claude models via Amazon Bedrock
    "bedrock/anthropic.claude-3-sonnet-20240229-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
    "bedrock/anthropic.claude-3-5-sonnet-20241022-v2:0",
    "bedrock/anthropic.claude-3-haiku-20240307-v1:0",
    "bedrock/anthropic.claude-3-opus-20240229-v1:0",
    # Anthropic Claude models Vertex AI
    "vertex_ai/claude-3-opus@20240229",
    "vertex_ai/claude-3-5-sonnet@20240620",
    "vertex_ai/claude-3-5-sonnet@20241022",
    "vertex_ai/claude-3-sonnet@20240229",
    "vertex_ai/claude-3-haiku@20240307",
]


# Get N responses from a single message, used for ensembling.
@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
@track_token_usage
def get_batch_responses_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
    n_responses=1,
) -> tuple[list[str], list[list[dict[str, Any]]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    if "claude" in model and "-OR" in model:
        # Handle Claude models through OpenRouter using chat completions
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model.replace("-OR", ""),
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif "claude" in model:
        new_msg_history = msg_history + [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": msg,
                    }
                ],
            }
        ]
        response = client.messages.create(
            model=model,
            max_tokens=MAX_NUM_TOKENS,
            temperature=temperature,
            system=system_message,
            messages=new_msg_history,
        )
        content = [response.content[0].text]
        new_msg_history = [
            new_msg_history + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": c,
                        }
                    ],
                }
            ]
            for c in content
        ]
    elif "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
            seed=0,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    elif model == "llama-3-1-405b-instruct":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=n_responses,
            stop=None,
        )
        content = [r.message.content for r in response.choices]
        new_msg_history = [
            new_msg_history + [{"role": "assistant", "content": c}] for c in content
        ]
    else:
        content, new_msg_history = [], []
        for _ in range(n_responses):
            c, hist = get_response_from_llm(
                msg,
                client,
                model,
                system_message,
                print_debug=False,
                msg_history=None,
                temperature=temperature,
            )
            content.append(c)
            new_msg_history.append(hist)

    if print_debug:
        # Just print the first one.
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history[0]):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


@track_token_usage
def make_llm_call(client, model, temperature, system_message, prompt):
    if model.startswith("gemini") and "-OR" in model:
        print(f"Debug: Using OpenRouter API for Gemini model: {model}")
        # Remove the -OR suffix for the actual model name
        model = model.replace("-OR", "")
        # Add the google/ prefix for OpenRouter
        if not model.startswith("google/"):
            model = f"google/{model}"
        print(f"Debug: Final model ID: {model}")
        
        # Build message history in OpenAI format
        new_msg_history = []
        if system_message:
            new_msg_history.append({"role": "system", "content": system_message})
        for msg in prompt:
            new_msg_history.append({"role": msg["role"], "content": msg["content"]})
        new_msg_history.append({"role": "user", "content": prompt[-1]["content"]})
        
        # Make the API call
        response = client.chat.completions.create(
            model=model,
            messages=new_msg_history,
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
        )
        return type('obj', (object,), {'choices': [type('obj', (object,), {'message': type('obj', (object,), {'content': response.choices[0].message.content})()})()]})
    elif "claude" in model and "-OR" in model:
        # Handle Claude models through OpenRouter using chat completions
        model_id = "anthropic/claude-3.5-sonnet"
        print(f"Debug: Using model ID in make_llm_call: {model_id}")
        return client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
        )
    elif "gpt" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_message},
                *prompt,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
            seed=0,
        )
    elif "o1" in model or "o3" in model:
        return client.chat.completions.create(
            model=model,
            messages=[
                {"role": "user", "content": system_message},
                *prompt,
            ],
            temperature=1,
            n=1,
            seed=0,
        )
    else:
        raise ValueError(f"Model {model} not supported.")


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.APITimeoutError,
        openai.InternalServerError,
        anthropic.RateLimitError,
    ),
)
def get_response_from_llm(
    prompt,
    client,
    model,
    system_message,
    print_debug=False,
    msg_history=None,
    temperature=0.7,
) -> tuple[str, list[dict[str, Any]]]:
    msg = prompt
    if msg_history is None:
        msg_history = []

    print(f"Debug: Model type: {model}")
    print(f"Debug: Client type: {type(client)}")

    if model.startswith("gemini"):
        print(f"Debug: Using OpenRouter API for Gemini model: {model}")
        # Remove any -OR suffix if present
        model = model.replace("-OR", "")
        # Use the correct OpenRouter model ID format
        model = "google/gemini-2.5-pro-preview-03-25"
        print(f"Debug: Final model ID: {model}")
        
        # Build message history in OpenAI format
        new_msg_history = []
        if system_message:
            print("Debug: Adding system message")
            new_msg_history.append({"role": "system", "content": system_message})
        
        if msg_history:
            print(f"Debug: Processing {len(msg_history)} existing messages")
            for i, msg in enumerate(msg_history):
                print(f"Debug: Message {i}: role={msg['role']}, content length={len(msg['content'])}")
                new_msg_history.append({"role": msg["role"], "content": msg["content"]})
        
        print(f"Debug: Adding current user message: {len(msg)} characters")
        new_msg_history.append({"role": "user", "content": msg})
        
        print(f"Debug: Final message count: {len(new_msg_history)}")
        print("Debug: Message structure:")
        for i, msg in enumerate(new_msg_history):
            print(f"  {i}. {msg['role']}: {len(msg['content'])} chars")
        
        try:
            # Make the API call
            print("Debug: Making API call to OpenRouter")
            response = client.chat.completions.create(
                model=model,
                messages=new_msg_history,
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
            )
            print("Debug: API call completed successfully")
            content = response.choices[0].message.content
            print(f"Debug: Response length: {len(content)} characters")
            
            # Update message history
            new_msg_history.append({"role": "assistant", "content": content})
            print("Debug: Returning response and updated message history")
            return content, new_msg_history
        except Exception as e:
            print(f"Debug: Error in API call: {str(e)}")
            raise
    elif "claude" in model:
        # Handle all Claude models (including OpenRouter) through chat completions
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        if isinstance(client, openai.OpenAI):  # OpenRouter case
            print("Debug: Using OpenRouter for Claude model")
            # Use the correct OpenRouter model ID format
            model_id = "anthropic/claude-3.5-sonnet"
            print(f"Debug: Using model ID: {model_id}")
            response = client.chat.completions.create(
                model=model_id,
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
            )
            content = response.choices[0].message.content
            new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
        else:  # Direct Anthropic case
            print("Debug: Using direct Anthropic client")
            new_msg_history = msg_history + [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": msg,
                        }
                    ],
                }
            ]
            response = client.messages.create(
                model=model,
                max_tokens=MAX_NUM_TOKENS,
                temperature=temperature,
                system=system_message,
                messages=new_msg_history,
            )
            content = response.content[0].text
            new_msg_history = new_msg_history + [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": content,
                        }
                    ],
                }
            ]
    elif "gpt" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif "o1" in model or "o3" in model:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = make_llm_call(
            client,
            model,
            temperature,
            system_message=system_message,
            prompt=new_msg_history,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepseek-coder-v2-0724":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="deepseek-coder",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model == "deepcoder-14b":
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        try:
            response = client.chat.completions.create(
                model="agentica-org/DeepCoder-14B-Preview",
                messages=[
                    {"role": "system", "content": system_message},
                    *new_msg_history,
                ],
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
                n=1,
                stop=None,
            )
            content = response.choices[0].message.content
        except Exception as e:
            # Fallback to direct API call if OpenAI client doesn't work with HuggingFace
            import requests
            headers = {
                "Authorization": f"Bearer {os.environ['HUGGINGFACE_API_KEY']}",
                "Content-Type": "application/json"
            }
            payload = {
                "inputs": {
                    "system": system_message,
                    "messages": [{"role": m["role"], "content": m["content"]} for m in new_msg_history]
                },
                "parameters": {
                    "temperature": temperature,
                    "max_new_tokens": MAX_NUM_TOKENS,
                    "return_full_text": False
                }
            }
            response = requests.post(
                "https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
                headers=headers,
                json=payload
            )
            if response.status_code == 200:
                content = response.json()["generated_text"]
            else:
                raise ValueError(f"Error from HuggingFace API: {response.text}")

        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    elif model in ["meta-llama/llama-3.1-405b-instruct", "llama-3-1-405b-instruct"]:
        new_msg_history = msg_history + [{"role": "user", "content": msg}]
        response = client.chat.completions.create(
            model="meta-llama/llama-3.1-405b-instruct",
            messages=[
                {"role": "system", "content": system_message},
                *new_msg_history,
            ],
            temperature=temperature,
            max_tokens=MAX_NUM_TOKENS,
            n=1,
            stop=None,
        )
        content = response.choices[0].message.content
        new_msg_history = new_msg_history + [{"role": "assistant", "content": content}]
    else:
        raise ValueError(f"Model {model} not supported.")

    if print_debug:
        print()
        print("*" * 20 + " LLM START " + "*" * 20)
        for j, msg in enumerate(new_msg_history):
            print(f'{j}, {msg["role"]}: {msg["content"]}')
        print(content)
        print("*" * 21 + " LLM END " + "*" * 21)
        print()

    return content, new_msg_history


def extract_json_between_markers(llm_output: str) -> dict | None:
    # Regular expression pattern to find JSON content between ```json and ```
    json_pattern = r"```json(.*?)```"
    matches = re.findall(json_pattern, llm_output, re.DOTALL)

    if not matches:
        # Fallback: Try to find any JSON-like content in the output
        json_pattern = r"\{.*?\}"
        matches = re.findall(json_pattern, llm_output, re.DOTALL)

    for json_string in matches:
        json_string = json_string.strip()
        try:
            parsed_json = json.loads(json_string)
            return parsed_json
        except json.JSONDecodeError:
            # Attempt to fix common JSON issues
            try:
                # Remove invalid control characters
                json_string_clean = re.sub(r"[\x00-\x1F\x7F]", "", json_string)
                parsed_json = json.loads(json_string_clean)
                return parsed_json
            except json.JSONDecodeError:
                continue  # Try next match

    return None  # No valid JSON found


def create_client(model) -> tuple[Any, str]:
    if model.startswith("claude-") and "-OR" in model:
        print(f"Using OpenRouter API with model {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            model.replace("-OR", ""),
        )
    elif model.startswith("claude-"):
        print(f"Using Anthropic API with model {model}.")
        return anthropic.Anthropic(), model
    elif model.startswith("gemini") and "-OR" in model:
        print(f"Using OpenRouter API with Gemini model {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            model.replace("-OR", ""),
        )
    elif model.startswith("gemini"):
        print(f"Debug: Using OpenRouter API for Gemini model: {model}")
        # Remove any -OR suffix if present
        model = model.replace("-OR", "")
        # Use the correct OpenRouter model ID format
        model = "google/gemini-2.5-pro-preview-03-25"
        print(f"Debug: Final model ID: {model}")
        
        # Build message history in OpenAI format
        new_msg_history = []
        if system_message:
            print("Debug: Adding system message")
            new_msg_history.append({"role": "system", "content": system_message})
        
        if msg_history:
            print(f"Debug: Processing {len(msg_history)} existing messages")
            for i, msg in enumerate(msg_history):
                print(f"Debug: Message {i}: role={msg['role']}, content length={len(msg['content'])}")
                new_msg_history.append({"role": msg["role"], "content": msg["content"]})
        
        print(f"Debug: Adding current user message: {len(msg)} characters")
        new_msg_history.append({"role": "user", "content": msg})
        
        print(f"Debug: Final message count: {len(new_msg_history)}")
        print("Debug: Message structure:")
        for i, msg in enumerate(new_msg_history):
            print(f"  {i}. {msg['role']}: {len(msg['content'])} chars")
        
        try:
            # Make the API call
            print("Debug: Making API call to OpenRouter")
            response = client.chat.completions.create(
                model=model,
                messages=new_msg_history,
                temperature=temperature,
                max_tokens=MAX_NUM_TOKENS,
            )
            print("Debug: API call completed successfully")
            content = response.choices[0].message.content
            print(f"Debug: Response length: {len(content)} characters")
            
            # Update message history
            new_msg_history.append({"role": "assistant", "content": content})
            print("Debug: Returning response and updated message history")
            return content, new_msg_history
        except Exception as e:
            print(f"Debug: Error in API call: {str(e)}")
            raise
    elif model.startswith("bedrock") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Amazon Bedrock with model {client_model}.")
        return anthropic.AnthropicBedrock(), client_model
    elif model.startswith("vertex_ai") and "claude" in model:
        client_model = model.split("/")[-1]
        print(f"Using Vertex AI with model {client_model}.")
        return anthropic.AnthropicVertex(), client_model
    elif "gpt" in model and "-OR" in model:
        print(f"Using OpenRouter API with model {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            model.replace("-OR", ""),
        )
    elif "gpt" in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif "o1" in model or "o3" in model:
        print(f"Using OpenAI API with model {model}.")
        return openai.OpenAI(), model
    elif model == "deepseek-coder-v2-0724":
        print(f"Using OpenAI API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["DEEPSEEK_API_KEY"],
                base_url="https://api.deepseek.com",
            ),
            model,
        )
    elif model == "deepcoder-14b":
        print(f"Using HuggingFace API with {model}.")
        # Using OpenAI client with HuggingFace API
        if "HUGGINGFACE_API_KEY" not in os.environ:
            raise ValueError("HUGGINGFACE_API_KEY environment variable not set")
        return (
            openai.OpenAI(
                api_key=os.environ["HUGGINGFACE_API_KEY"],
                base_url="https://api-inference.huggingface.co/models/agentica-org/DeepCoder-14B-Preview",
            ),
            model,
        )
    elif model == "llama3.1-405b":
        print(f"Using OpenRouter API with {model}.")
        return (
            openai.OpenAI(
                api_key=os.environ["OPENROUTER_API_KEY"],
                base_url="https://openrouter.ai/api/v1",
            ),
            "meta-llama/llama-3.1-405b-instruct",
        )
    else:
        raise ValueError(f"Model {model} not supported.")
