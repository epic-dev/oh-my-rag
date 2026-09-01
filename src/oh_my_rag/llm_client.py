import json
import re
import time

import requests

from oh_my_rag.config import LLM_BASE_URL, LLM_MODEL, LLM_TIMEOUT
from oh_my_rag.usage_log import LlmCall, get_usage_log

# mlx-lm hands back the model's raw output, so the stop tokens and Qwen3's
# reasoning block arrive verbatim and have to be cleaned off here.
# Reasoning models emit their scratchpad inline, and a 4-bit local model emits it
# sloppily. Three shapes have all been seen in practice, so all three are handled:
#
#   <think>a</think>answer   a matched pair — the common case
#   a</think>answer          an ORPHAN CLOSE: the opener never arrived (the chat
#                            template pre-filled it, or the model simply skipped
#                            it). Everything before the tag is scratchpad.
#   <think>a                 an orphan open: max_tokens cut the thought short.
#
# Tags are matched loosely (whitespace inside the angle brackets, any case)
# because a quantized model does not reliably reproduce them exactly.
_THINK_TAG = r"<\s*{}think\s*>"
_THINK_BLOCK_RE = re.compile(
    _THINK_TAG.format("") + r".*?" + _THINK_TAG.format(r"/\s*"), re.DOTALL | re.IGNORECASE
)
# Greedy from the start: with several orphan closes, everything up to the LAST
# one is scratchpad.
_ORPHAN_CLOSE_RE = re.compile(r"\A.*" + _THINK_TAG.format(r"/\s*"), re.DOTALL | re.IGNORECASE)
_ORPHAN_OPEN_RE = re.compile(_THINK_TAG.format("") + r".*\Z", re.DOTALL | re.IGNORECASE)

_CONTROL_TOKENS = ("</s>", "<|im_end|>")


def strip_control_tokens(content: str) -> str:
    """Remove the model's reasoning scratchpad and stop tokens from a reply.

    Order matters: complete blocks first, so that a well-formed pair is not
    mistaken for an orphan by the looser patterns that follow.
    """
    content = _THINK_BLOCK_RE.sub("", content)
    content = _ORPHAN_CLOSE_RE.sub("", content)
    content = _ORPHAN_OPEN_RE.sub("", content)
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
    reasoning_effort: str | None = None,
    kind: str = "other",
) -> str:
    """POST to an OpenAI-compatible /chat/completions endpoint, non-streaming,
    and return the reply with control tokens stripped.

    Every call — including one that raises — is appended to the usage log.
    """
    payload = {"model": model, "messages": messages, "stream": False}
    # Only sent when explicitly asked for, so the answering path's payload stays
    # byte-identical to what it has always been.
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    # Ollama honours this on its OpenAI-compatible endpoint; "none" disables a
    # reasoning model's scratchpad. Sent only when asked for, since other
    # backends may reject an unknown field.
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    started = time.perf_counter()

    def elapsed() -> float:
        return (time.perf_counter() - started) * 1000

    try:
        response = requests.post(base_url, json=payload, timeout=timeout)
        response.raise_for_status()
        body = response.json()
    except Exception as exc:
        # A timeout or a refused connection is exactly the traffic worth having
        # a record of, so failures are logged before the exception propagates.
        get_usage_log().record(
            LlmCall(
                kind=kind, model=model, base_url=base_url, streamed=False,
                messages_sent=len(messages), prompt_chars=prompt_chars,
                duration_ms=elapsed(), ok=False,
                error=f"{type(exc).__name__}: {exc}"[:500],
            )
        )
        raise

    choice = (body.get("choices") or [{}])[0]
    content = (choice.get("message") or {}).get("content") or ""
    usage = body.get("usage") or {}

    get_usage_log().record(
        LlmCall(
            kind=kind, model=model, base_url=base_url, streamed=False,
            messages_sent=len(messages), prompt_chars=prompt_chars,
            duration_ms=elapsed(),
            prompt_tokens=usage.get("prompt_tokens"),
            completion_tokens=usage.get("completion_tokens"),
            total_tokens=usage.get("total_tokens"),
            reply_chars=len(content),
            finish_reason=choice.get("finish_reason"),
        )
    )
    return strip_control_tokens(content)


class StreamScrubber:
    """Removes a reasoning scratchpad from a stream, one delta at a time.

    Streaming makes this harder than the batch case. Two hazards:

    1. **A tag can be split across deltas.** So text is never emitted past a
       trailing `<` that might be the start of one — which also means ordinary
       prose (no `<` anywhere) streams with zero delay.
    2. **An orphan `</think>` invalidates text already sent.** Nothing can
       un-print it. `probation` holds back the first N characters so a leading
       orphan can still be caught; with `probation=0` the scrubber instead
       reports it via `saw_late_orphan`.

    Callers that know there is no scratchpad — `reasoning_effort="none"`, or a
    backend like Ollama that returns reasoning in its own field — can safely use
    `probation=0` and stream from the first token.
    """

    _TAIL = 12  # longest thing we might need to re-join: "<|im_end|>", "</think >"
    _OPEN = re.compile(r"<\s*think\s*>", re.IGNORECASE)
    _CLOSE = re.compile(r"<\s*/\s*think\s*>", re.IGNORECASE)

    def __init__(self, probation: int = 0):
        self._buf = ""
        self._in_think = False
        self._emitted = False
        self._probation = probation
        self._held = ""
        self.saw_late_orphan = False

    def _safe_cut(self, text: str) -> int:
        """How much of `text` can be released without splitting a tag."""
        idx = text.rfind("<")
        if idx != -1 and (len(text) - idx) <= self._TAIL:
            return idx
        return len(text)

    def feed(self, delta: str) -> str:
        self._buf += delta
        out = []

        while True:
            if self._in_think:
                m = self._CLOSE.search(self._buf)
                if not m:
                    # Scratchpad: discard, but keep a possible partial close tag.
                    self._buf = self._buf[self._safe_cut(self._buf):]
                    break
                self._buf = self._buf[m.end():]
                self._in_think = False
                continue

            opening = self._OPEN.search(self._buf)
            closing = self._CLOSE.search(self._buf)

            # A close with no open before it: everything so far was scratchpad.
            if closing and (not opening or closing.start() < opening.start()):
                if self._emitted:
                    self.saw_late_orphan = True
                out.clear()
                self._held = ""
                self._buf = self._buf[closing.end():]
                continue

            if opening:
                out.append(self._buf[: opening.start()])
                self._buf = self._buf[opening.end():]
                self._in_think = True
                continue
            break

        if not self._in_think:
            cut = self._safe_cut(self._buf)
            out.append(self._buf[:cut])
            self._buf = self._buf[cut:]

        return self._release("".join(out))

    def _release(self, text: str) -> str:
        """Apply the probation window, then strip stop tokens."""
        if self._probation:
            self._held += text
            if len(self._held) < self._probation:
                return ""
            text, self._held, self._probation = self._held, "", 0

        for token in _CONTROL_TOKENS:
            text = text.replace(token, "")
        if text:
            self._emitted = True
        return text

    def flush(self) -> str:
        """Emit whatever is left once the stream ends."""
        text = "" if self._in_think else self._buf
        self._buf = ""
        held, self._held, self._probation = self._held, "", 0
        text = held + text
        for token in _CONTROL_TOKENS:
            text = text.replace(token, "")
        return text


def chat_stream(
    messages: list[dict],
    *,
    base_url: str = LLM_BASE_URL,
    model: str = LLM_MODEL,
    timeout: float = LLM_TIMEOUT,
    temperature: float | None = None,
    max_tokens: int | None = None,
    reasoning_effort: str | None = None,
    scrub: bool = True,
    probation: int = 0,
    kind: str = "other",
):
    """Yield answer text as it is generated, over SSE.

    Only `delta.content` is yielded. Ollama also streams a `delta.reasoning`
    field for reasoning models; it is ignored rather than shown, which keeps the
    behaviour identical to the non-streaming path.
    """
    payload = {"model": model, "messages": messages, "stream": True}
    # An SSE stream carries no usage block unless it is asked for; without this
    # every streamed call would land in the usage log with NULL token counts.
    if get_usage_log().enabled:
        payload["stream_options"] = {"include_usage": True}
    if temperature is not None:
        payload["temperature"] = temperature
    if max_tokens is not None:
        payload["max_tokens"] = max_tokens
    if reasoning_effort is not None:
        payload["reasoning_effort"] = reasoning_effort

    scrubber = StreamScrubber(probation=probation) if scrub else None
    prompt_chars = sum(len(m.get("content") or "") for m in messages)
    started = time.perf_counter()
    usage: dict = {}
    finish_reason = None
    reply_chars = 0
    error = None

    try:
        with requests.post(
            base_url, json=payload, timeout=timeout, stream=True
        ) as response:
            response.raise_for_status()
            for raw in response.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw[len("data:"):].strip() if raw.startswith("data:") else raw.strip()
                if not line or line == "[DONE]":
                    continue
                try:
                    chunk = json.loads(line)
                except ValueError:
                    continue

                # The usage block arrives in its own final chunk, which carries
                # no choices at all.
                if chunk.get("usage"):
                    usage = chunk["usage"]

                choices = chunk.get("choices") or []
                if not choices:
                    continue
                if choices[0].get("finish_reason"):
                    finish_reason = choices[0]["finish_reason"]
                piece = (choices[0].get("delta") or {}).get("content") or ""
                if not piece:
                    continue

                text = scrubber.feed(piece) if scrubber else piece
                if text:
                    reply_chars += len(text)
                    yield text

        if scrubber:
            tail = scrubber.flush()
            if tail:
                reply_chars += len(tail)
                yield tail
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"[:500]
        raise
    finally:
        # `finally`, so a consumer that abandons the generator early still
        # leaves a record of the traffic it caused.
        get_usage_log().record(
            LlmCall(
                kind=kind, model=model, base_url=base_url, streamed=True,
                messages_sent=len(messages), prompt_chars=prompt_chars,
                duration_ms=(time.perf_counter() - started) * 1000,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                total_tokens=usage.get("total_tokens"),
                reply_chars=reply_chars,
                finish_reason=finish_reason,
                ok=error is None,
                error=error,
            )
        )
