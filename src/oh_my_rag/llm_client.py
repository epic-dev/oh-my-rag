import re

import requests

from oh_my_rag.config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT

# mlx-lm hands back the model's raw output, so the stop tokens and Qwen3's
# reasoning block arrive verbatim and have to be cleaned off here.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
# A reasoning model that hits max_tokens mid-thought never emits the closing
# tag, so the paired pattern above leaves the whole block in place.
_OPEN_THINK_RE = re.compile(r"<think>.*\Z", re.DOTALL)
_CONTROL_TOKENS = ("</s>", "<|im_end|>")


def strip_control_tokens(content: str) -> str:
    content = _THINK_RE.sub("", content)
    content = _OPEN_THINK_RE.sub("", content)
    for token in _CONTROL_TOKENS:
        content = content.replace(token, "")
    return content.strip()


def chat(
    messages: list[dict],
    *,
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
    timeout: float = LLM_TIMEOUT,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """POST to an OpenAI-compatible /chat/completions endpoint, non-streaming,
    and return the reply with control tokens stripped."""
    payload = {"model": model, "messages": messages, "stream": False}
    # Only sent when explicitly asked for, so the answering path's payload stays
    # byte-identical to what it has always been.
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens

    response = requests.post(base_url, json=payload, timeout=timeout)
    response.raise_for_status()

    content = response.json()["choices"][0]["message"]["content"]
    return strip_control_tokens(content)
