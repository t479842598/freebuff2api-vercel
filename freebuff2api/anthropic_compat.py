from __future__ import annotations

import json
import time
import uuid as uuid_mod
from typing import Any

from .codebuff import FreebuffSession
from .models import resolve_model
from .openai_compat import normalize_chat_messages


# ── Anthropic → OpenAI parameter mapping ──────────────────────────────
# Keys that can pass through to the upstream OpenAI-style payload.
_ANTHROPIC_UPSTREAM_KEYS = frozenset(
    {
        "frequency_penalty",
        "logit_bias",
        "logprobs",
        "max_tokens",
        "metadata",
        "modalities",
        "parallel_tool_calls",
        "presence_penalty",
        "reasoning_effort",
        "seed",
        "service_tier",
        "stream_options",
        "temperature",
        "tool_choice",
        "top_logprobs",
        "top_p",
        "top_k",
        "user",
    }
)

# ── Stop reason mapping ───────────────────────────────────────────────

_OPENAI_TO_ANTHROPIC_STOP: dict[str | None, str] = {
    None: "end_turn",
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "end_turn",
    "function_call": "tool_use",
}


def _map_stop_reason(openai_reason: str | None) -> str:
    return _OPENAI_TO_ANTHROPIC_STOP.get(openai_reason, "end_turn")


# ── 2.1 Request conversion ────────────────────────────────────────────


def _anthropic_system_to_openai_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Anthropic top-level ``system`` field into an OpenAI system message.

    Anthropic ``system`` can be:
      - a plain string
      - a list of ``{"type": "text", "text": "..."}`` blocks
    Returns **one** system message dict suitable for ``normalize_chat_messages``.
    """
    system = body.get("system")
    if system is None:
        return []
    if isinstance(system, str):
        return [{"role": "system", "content": system}]
    if isinstance(system, list):
        text_parts: list[str] = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(str(block.get("text", "")))
        if text_parts:
            return [{"role": "system", "content": "\n".join(text_parts)}]
    return []


def _anthropic_content_to_openai(content: Any) -> str | list[dict[str, Any]]:
    """Flatten Anthropic message ``content`` to an OpenAI-compatible equivalent.

    Returns a plain string when every block is text; returns a multimodal list
    otherwise (for image / file blocks).
    """
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return str(content)

    # If every block is text, return a simple string.
    if all(
        isinstance(b, dict) and b.get("type") == "text" for b in content
    ):
        return "".join(str(b.get("text", "")) for b in content)

    # Mixed / multimodal blocks – map to OpenAI content-parts list.
    parts: list[dict[str, Any]] = []
    for block in content:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            parts.append({"type": "text", "text": str(block.get("text", ""))})
        elif block_type == "image":
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": _data_uri(block),
                        "detail": "auto",
                    },
                }
            )
        elif block_type == "tool_result":
            # tool_result will be handled separately; skip here.
            pass
        elif block_type == "tool_use":
            # tool_use will be handled separately; skip here.
            pass
        elif block_type == "document":
            # Anthropic document -> base64 content
            source = block.get("source", {})
            if isinstance(source, dict) and source.get("type") == "base64":
                parts.append(
                    {
                        "type": "file",
                        "file": {
                            "filename": source.get("media_type", "application/octet-stream"),
                            "file_data": source.get("data", ""),
                        },
                    }
                )
    return parts if parts else ""


def _data_uri(block: dict[str, Any]) -> str:
    source = block.get("source", {})
    if isinstance(source, dict):
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return f"data:{media_type};base64,{data}"
    return ""


def anthropic_to_openai_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    """Convert Anthropic Messages-API body into OpenAI-style chat messages.

    * Top-level ``system`` → ``{"role": "system", ...}``
    * ``tool_use`` blocks → ``assistant`` with ``tool_calls``
    * ``tool_result`` blocks → ``tool`` role messages
    * Text / image blocks → ``user`` / ``assistant`` with appropriate content
    """
    # Collect tool-use → call-id mappings so we can link tool_result messages.
    tool_use_to_call_id: dict[str, str] = {}
    messages: list[dict[str, Any]] = []

    # Extract system first.
    system_msgs = _anthropic_system_to_openai_messages(body)
    messages.extend(system_msgs)

    for msg in body.get("messages") or []:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "user")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            messages.append({"role": role, "content": str(content or "")})
            continue

        # Separate blocks into text, tool_use, and tool_result groups.
        text_blocks: list[dict[str, Any]] = []
        tool_use_blocks: list[dict[str, Any]] = []
        tool_result_blocks: list[dict[str, Any]] = []

        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "tool_use":
                tool_use_blocks.append(block)
            elif bt == "tool_result":
                tool_result_blocks.append(block)
            else:
                text_blocks.append(block)

        # Emit text content as one message (role preserved).
        if text_blocks:
            openai_content = _anthropic_content_to_openai(text_blocks)
            messages.append({"role": role, "content": openai_content})

        # Emit each tool_use as an assistant message with tool_calls.
        for tb in tool_use_blocks:
            tu_id = tb.get("id") or f"toolu_{uuid_mod.uuid4().hex[:24]}"
            call_id = f"call_{uuid_mod.uuid4().hex[:24]}"
            tool_use_to_call_id[tu_id] = call_id

            input_val = tb.get("input", {})
            if isinstance(input_val, str):
                arguments = input_val
            else:
                arguments = json.dumps(input_val, ensure_ascii=False)

            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": call_id,
                            "type": "function",
                            "function": {
                                "name": tb.get("name", ""),
                                "arguments": arguments,
                            },
                        }
                    ],
                }
            )

        # Emit tool_results as tool-role messages.
        for tr in tool_result_blocks:
            tu_id = tr.get("tool_use_id", "")
            call_id = tool_use_to_call_id.get(
                tu_id, f"call_{uuid_mod.uuid4().hex[:24]}"
            )
            result_content = tr.get("content", "")
            if isinstance(result_content, list):
                # Anthropic tool_result content can be a list of text blocks.
                text_parts = [
                    str(b.get("text", ""))
                    for b in result_content
                    if isinstance(b, dict) and b.get("type") == "text"
                ]
                result_content = "\n".join(text_parts) if text_parts else ""
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": str(result_content),
                }
            )

    return messages


def anthropic_tools_to_openai(
    tools: list[dict[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    """Convert Anthropic ``tools`` array (``input_schema``) to OpenAI format
    (``function.parameters``)."""
    if not tools:
        return None
    openai_tools: list[dict[str, Any]] = []
    for tool in tools:
        if not isinstance(tool, dict):
            continue
        params = tool.get("input_schema", {})
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.get("name", ""),
                    "description": tool.get("description", ""),
                    "parameters": params,
                },
            }
        )
    return openai_tools


def anthropic_tool_choice_to_openai(tool_choice: Any) -> Any:
    """Map Anthropic ``tool_choice`` to OpenAI ``tool_choice``."""
    if tool_choice is None:
        return None
    if isinstance(tool_choice, str):
        # "auto" / "any" / "none"
        if tool_choice == "any":
            return "required"
        if tool_choice in {"auto", "none"}:
            return tool_choice
        return "auto"
    if isinstance(tool_choice, dict):
        tc_type = tool_choice.get("type")
        if tc_type == "tool" and tool_choice.get("name"):
            return {
                "type": "function",
                "function": {"name": tool_choice["name"]},
            }
        if tc_type == "auto":
            return "auto"
        if tc_type == "any":
            return "required"
    return "auto"


def build_anthropic_upstream_payload(
    body: dict[str, Any],
    *,
    session: FreebuffSession,
    run_id: str,
    client_id: str,
    trace_session_id: str | None = None,
    upstream_model_id: str | None = None,
    top_k: int | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Build the upstream OpenAI-style payload from an Anthropic request body."""
    # Resolve model.
    model_id = upstream_model_id or body.get("model", "")
    try:
        upstream_id = resolve_model(model_id).upstream_id
    except ValueError:
        upstream_id = model_id

    # Convert tools.
    openai_tools = anthropic_tools_to_openai(body.get("tools"))

    # Convert messages (including tool_use/tool_result remapping).
    raw_messages = anthropic_to_openai_messages(body)

    # Apply Buffy prompt injection via the existing normalizer.
    messages = normalize_chat_messages(raw_messages, system_prompt=system_prompt)

    # Build payload from allowed keys.
    payload: dict[str, Any] = {
        key: body[key]
        for key in _ANTHROPIC_UPSTREAM_KEYS
        if key in body and body[key] is not None
    }
    payload["model"] = upstream_id
    payload["messages"] = messages
    payload["stream"] = True
    payload.setdefault("stop", ['"cb_easp"'])

    # Map Anthropic stop_sequences → stop (merge with existing stop).
    stop_sequences = body.get("stop_sequences")
    if isinstance(stop_sequences, list):
        existing_stop: list = payload.setdefault("stop", [])
        if isinstance(existing_stop, list):
            payload["stop"] = list(set(existing_stop + stop_sequences))

    # Map top_k.
    if top_k is not None:
        payload["top_k"] = top_k
    elif "top_k" in body and body["top_k"] is not None:
        payload["top_k"] = body["top_k"]

    # Map tools.
    if openai_tools:
        payload["tools"] = openai_tools

    # Map tool_choice.
    tc = anthropic_tool_choice_to_openai(body.get("tool_choice"))
    if tc is not None:
        payload["tool_choice"] = tc

    # Metadata.
    payload["provider"] = {"data_collection": "deny"}
    payload["codebuff_metadata"] = {
        "freebuff_instance_id": session.instance_id,
        "trace_session_id": trace_session_id or str(uuid_mod.uuid4()),
        "run_id": run_id,
        "client_id": client_id,
        "cost_mode": "free",
    }
    return payload


# ── 2.2 Non-streaming accumulator ─────────────────────────────────────


class AnthropicCompletionAccumulator:
    """Collect OpenAI SSE chunks and produce an Anthropic Messages response."""

    def __init__(
        self,
        model: str,
        system_fingerprint: str | None = None,
    ) -> None:
        self.model = model
        self.id: str | None = None
        self.created: int | None = None
        self.usage: dict[str, Any] | None = None
        self.system_fingerprint: str | None = system_fingerprint

        # Content blocks are built in-order: text_blocks followed by tool_use blocks.
        self._text: str = ""
        self._tool_uses: list[dict[str, Any]] = []  # accumulated tool_use blocks
        self._tool_call_index_map: dict[int, int] = {}  # openai_index → anthropic_index
        self._next_block_index: int = 0

        self._stop_reason: str | None = None

    @property
    def _stop_reason_or_default(self) -> str:
        return _map_stop_reason(self._stop_reason)

    def add(self, chunk: dict[str, Any]) -> None:
        """Ingest one OpenAI SSE chunk."""
        self.id = chunk.get("id") or self.id
        self.created = chunk.get("created") or self.created
        # Keep original model — upstream chunk model may differ
        self.usage = chunk.get("usage") or self.usage
        self.system_fingerprint = (
            chunk.get("system_fingerprint") or self.system_fingerprint
        )

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")

            if isinstance(content, str):
                self._text += content

            if tool_calls:
                for tc in tool_calls or []:
                    idx = tc.get("index", 0)
                    func = tc.get("function") or {}
                    name = func.get("name", "")
                    args = func.get("arguments", "")

                    if idx not in self._tool_call_index_map:
                        # New tool call → new anthropic tool_use block.
                        tool_use_id = tc.get("id") or f"toolu_{uuid_mod.uuid4().hex[:24]}"
                        self._tool_call_index_map[idx] = self._next_block_index
                        self._next_block_index += 1
                        self._tool_uses.append(
                            {
                                "type": "tool_use",
                                "id": tool_use_id,
                                "name": name,
                                "input": {"_partial": args},
                            }
                        )
                    else:
                        anthro_idx = self._tool_call_index_map[idx]
                        tu = self._tool_uses[anthro_idx]
                        if name:
                            tu["name"] = name
                        if args:
                            existing: dict[str, Any] = tu.setdefault("input", {})
                            partial = existing.pop("_partial", "") + args
                            existing["_partial"] = partial

            if choice.get("finish_reason"):
                self._stop_reason = choice["finish_reason"]

    def final_response(self) -> dict[str, Any]:
        """Compose the Anthropic Messages response from accumulated chunks."""
        msg_id = self.id or f"msg_{uuid_mod.uuid4().hex[:24]}"

        # Build content array.
        content: list[dict[str, Any]] = []
        if self._text:
            content.append({"type": "text", "text": self._text})

        # Finalize tool_use input: parse partial JSON.
        for tu in self._tool_uses:
            inp: dict[str, Any] = tu.get("input", {})
            partial: str = inp.pop("_partial", "")
            if partial:
                try:
                    tu["input"] = json.loads(partial)
                except (json.JSONDecodeError, TypeError):
                    tu["input"] = partial  # fallback: keep as raw string
            else:
                tu["input"] = inp
            content.append(tu)

        # Usage.
        usage: dict[str, Any] = self.usage or {
            "input_tokens": 0,
            "output_tokens": 0,
        }
        # Normalize usage keys for Anthropic.
        usage_out: dict[str, int] = {}
        if "prompt_tokens" in usage:
            usage_out["input_tokens"] = usage["prompt_tokens"]
        elif "input_tokens" in usage:
            usage_out["input_tokens"] = usage["input_tokens"]
        else:
            usage_out["input_tokens"] = 0
        if "completion_tokens" in usage:
            usage_out["output_tokens"] = usage["completion_tokens"]
        elif "output_tokens" in usage:
            usage_out["output_tokens"] = usage["output_tokens"]
        else:
            usage_out["output_tokens"] = 0

        response: dict[str, Any] = {
            "id": msg_id,
            "type": "message",
            "role": "assistant",
            "content": content,
            "model": self.model,
            "stop_reason": self._stop_reason_or_default,
            "stop_sequence": None,
            "usage": usage_out,
        }
        return response


# ── 2.3 Streaming state machine ───────────────────────────────────────


class AnthropicStreamState:
    """Tracks the state of an Anthropic streaming response while consuming
    OpenAI SSE chunks."""

    def __init__(
        self,
        model: str,
        system_fingerprint: str | None = None,
    ) -> None:
        self.model = model
        self.message_id: str | None = None
        self.usage: dict[str, Any] | None = None
        self.system_fingerprint: str | None = system_fingerprint

        # Current state per content block.
        self._text: str = ""
        self._current_tool_index: int | None = None  # openai tool index
        self._tool_use_ids: dict[int, str] = {}  # openai idx → tool_use id
        self._tool_names: dict[int, str] = {}  # openai idx → tool name
        self._tool_arg_bufs: dict[int, str] = {}  # openai idx → partial args
        self._next_anthro_index: int = 0
        self._openai_to_anthro_index: dict[int, int] = {}  # openai → anthro block index
        self._active_block_type: str | None = None  # "text" | "tool_use"

        self._stop_reason: str | None = None
        self._message_started: bool = False
        self._text_block_started: bool = False
        self._text_block_closed: bool = False

    @property
    def _stop_reason_or_default(self) -> str:
        return _map_stop_reason(self._stop_reason)

    def consume_chunk(
        self, chunk: dict[str, Any]
    ) -> list[tuple[str, dict[str, Any]]]:
        """Process one OpenAI chunk, return a list of (event_type, data) tuples
        ready for SSE encoding."""
        events: list[tuple[str, dict[str, Any]]] = []

        self.message_id = chunk.get("id") or self.message_id
        # Keep original model — upstream chunk model may differ
        self.usage = chunk.get("usage") or self.usage
        self.system_fingerprint = (
            chunk.get("system_fingerprint") or self.system_fingerprint
        )

        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            tool_calls = delta.get("tool_calls")
            finish_reason = choice.get("finish_reason")
            index = choice.get("index", 0)

            # Start message if not already started.
            if not self._message_started:
                self._message_started = True
                self.message_id = self.message_id or f"msg_{uuid_mod.uuid4().hex[:24]}"
                usage_event: dict[str, int] = {}
                if self.usage:
                    usage_event["input_tokens"] = self.usage.get("prompt_tokens", 0)
                    usage_event["output_tokens"] = self.usage.get("completion_tokens", 1)
                else:
                    usage_event = {"input_tokens": 0, "output_tokens": 1}
                events.append(
                    (
                        "message_start",
                        {
                            "type": "message_start",
                            "message": {
                                "id": self.message_id,
                                "type": "message",
                                "role": "assistant",
                                "content": [],
                                "model": self.model,
                                "stop_reason": None,
                                "stop_sequence": None,
                                "usage": usage_event,
                            },
                        },
                    )
                )

            # ── Text delta ──
            if isinstance(content, str):
                if not self._text_block_started:
                    # First text delta → start text block.
                    self._text_block_started = True
                    self._active_block_type = "text"
                    events.append(
                        (
                            "content_block_start",
                            {
                                "type": "content_block_start",
                                "index": self._next_anthro_index,
                                "content_block": {"type": "text", "text": ""},
                            },
                        )
                    )
                self._text += content
                events.append(
                    (
                        "content_block_delta",
                        {
                            "type": "content_block_delta",
                            "index": self._next_anthro_index,
                            "delta": {"type": "text_delta", "text": content},
                        },
                    )
                )

            # ── Tool call delta ──
            if tool_calls:
                for tc in tool_calls:
                    oai_idx = tc.get("index", 0)
                    func = tc.get("function") or {}
                    name = func.get("name", "")
                    args = func.get("arguments", "")

                    if oai_idx not in self._openai_to_anthro_index:
                        # New tool_use block.
                        anthro_idx = self._next_anthro_index
                        self._next_anthro_index += 1
                        self._openai_to_anthro_index[oai_idx] = anthro_idx
                        tu_id = tc.get("id") or f"toolu_{uuid_mod.uuid4().hex[:24]}"
                        self._tool_use_ids[oai_idx] = tu_id

                        # Close text block if it was open (shouldn't be, but safety).
                        if (
                            self._text_block_started
                            and not self._text_block_closed
                            and self._active_block_type == "text"
                        ):
                            self._text_block_closed = True
                            text_anthro_idx = 0  # text is always index 0 in our impl
                            events.append(
                                (
                                    "content_block_stop",
                                    {
                                        "type": "content_block_stop",
                                        "index": text_anthro_idx,
                                    },
                                )
                            )

                        self._active_block_type = "tool_use"
                        events.append(
                            (
                                "content_block_start",
                                {
                                    "type": "content_block_start",
                                    "index": anthro_idx,
                                    "content_block": {
                                        "type": "tool_use",
                                        "id": tu_id,
                                        "name": name,
                                        "input": {},
                                    },
                                },
                            )
                        )

                    anthro_idx = self._openai_to_anthro_index[oai_idx]
                    if name:
                        self._tool_names[oai_idx] = name
                    if args:
                        buf = self._tool_arg_bufs.get(oai_idx, "")
                        self._tool_arg_bufs[oai_idx] = buf + args
                        events.append(
                            (
                                "content_block_delta",
                                {
                                    "type": "content_block_delta",
                                    "index": anthro_idx,
                                    "delta": {
                                        "type": "input_json_delta",
                                        "partial_json": args,
                                    },
                                },
                            )
                        )

            # ── Finish reason ──
            if finish_reason:
                self._stop_reason = finish_reason

        return events

    def finalize_events(self) -> list[tuple[str, dict[str, Any]]]:
        """Called after all chunks are consumed to emit closing events."""
        events: list[tuple[str, dict[str, Any]]] = []

        if not self._message_started:
            # No content at all — edge case.
            self.message_id = self.message_id or f"msg_{uuid_mod.uuid4().hex[:24]}"
            events.append(
                (
                    "message_start",
                    {
                        "type": "message_start",
                        "message": {
                            "id": self.message_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                            "model": self.model,
                            "stop_reason": None,
                            "stop_sequence": None,
                            "usage": {"input_tokens": 0, "output_tokens": 1},
                        },
                    },
                )
            )

        # Close text block if open.
        if self._text_block_started and not self._text_block_closed:
            self._text_block_closed = True
            if self._active_block_type == "text":
                text_anthro_idx = 0  # text is always first block (index 0)
                events.append(
                    (
                        "content_block_stop",
                        {
                            "type": "content_block_stop",
                            "index": text_anthro_idx,
                        },
                    )
                )

        # Close any open tool_use blocks.
        for oai_idx in sorted(self._tool_use_ids):
            anthro_idx = self._openai_to_anthro_index.get(oai_idx)
            if anthro_idx is not None:
                events.append(
                    (
                        "content_block_stop",
                        {
                            "type": "content_block_stop",
                            "index": anthro_idx,
                        },
                    )
                )

        # message_delta.
        usage_delta: dict[str, int] = {}
        if self.usage:
            output_tokens = self.usage.get(
                "completion_tokens",
                self.usage.get("output_tokens", 0),
            )
            usage_delta["output_tokens"] = output_tokens
        else:
            usage_delta["output_tokens"] = 0

        events.append(
            (
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {
                        "stop_reason": self._stop_reason_or_default,
                        "stop_sequence": None,
                    },
                    "usage": usage_delta,
                },
            )
        )

        # message_stop.
        events.append(
            (
                "message_stop",
                {"type": "message_stop"},
            )
        )

        return events


# ── 2.4 SSE encoding ──────────────────────────────────────────────────


def anthropic_sse_encode(
    event_type: str,
    data: dict[str, Any] | str,
) -> bytes:
    """Encode an Anthropic SSE event.

    Format::

        event: {event_type}
        data: {json}

    """
    payload: str
    if isinstance(data, str):
        payload = data
    else:
        payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    return f"event: {event_type}\ndata: {payload}\n\n".encode("utf-8")


def anthropic_sse_ping() -> bytes:
    """Return a ping heartbeat event (``event: ping`` with empty data)."""
    return b"event: ping\ndata: {}\n\n"


# ── 2.5 Error response ────────────────────────────────────────────────


def anthropic_error_payload(
    message: str,
    *,
    error_type: str = "api_error",
    status_code: int = 500,
) -> dict[str, Any]:
    """Build an Anthropic-compatible error payload."""
    return {
        "type": "error",
        "error": {
            "type": error_type,
            "message": message,
        },
    }
