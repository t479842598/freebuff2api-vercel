from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FreebuffModel:
    id: str
    agent_id: str
    owned_by: str = "freebuff"
    upstream_model_id: str | None = None
    session_model_id: str | None = None
    parent_agent_id: str | None = None

    @property
    def upstream_id(self) -> str:
        return self.upstream_model_id or self.id

    @property
    def session_id(self) -> str:
        return self.session_model_id or self.upstream_id


FREEBUFF_MODELS: tuple[FreebuffModel, ...] = (
    FreebuffModel("deepseek/deepseek-v4-flash", "base2-free-deepseek-flash"),
    FreebuffModel("deepseek/deepseek-v4-pro", "base2-free-deepseek"),
    FreebuffModel("moonshotai/kimi-k2.6", "base2-free-kimi"),
    FreebuffModel("minimax/minimax-m2.7", "base2-free"),
    FreebuffModel("minimax/minimax-m3", "base2-free-minimax-m3"),
    FreebuffModel("mimo/mimo-v2.5", "base2-free-mimo"),
    FreebuffModel("mimo/mimo-v2.5-pro", "base2-free-mimo-pro"),
)

DEFAULT_MODEL = FREEBUFF_MODELS[0]
CONTEXT_PRUNER_AGENT_ID = "context-pruner"
GEMINI_THINKER_AGENT_ID = "thinker-with-files-gemini"
GEMINI_THINKER_PARENT_AGENT_ID = "base2-free-kimi"
GEMINI_THINKER_PARENT_MODEL_ID = "moonshotai/kimi-k2.6"
GEMINI_FLASH_LITE_SESSION_MODEL_ID = DEFAULT_MODEL.id

GEMINI_FREE_MODELS: tuple[FreebuffModel, ...] = (
    FreebuffModel(
        "google/gemini-2.5-flash-lite",
        "file-picker",
        owned_by="google",
        session_model_id=GEMINI_FLASH_LITE_SESSION_MODEL_ID,
        parent_agent_id=DEFAULT_MODEL.agent_id,
    ),
    FreebuffModel(
        "google/gemini-3.1-flash-lite-preview",
        "file-picker-max",
        owned_by="google",
        session_model_id=GEMINI_FLASH_LITE_SESSION_MODEL_ID,
        parent_agent_id=DEFAULT_MODEL.agent_id,
    ),
    FreebuffModel(
        "google/gemini-3.1-pro-preview",
        GEMINI_THINKER_AGENT_ID,
        owned_by="google",
        session_model_id=GEMINI_THINKER_PARENT_MODEL_ID,
        parent_agent_id=GEMINI_THINKER_PARENT_AGENT_ID,
    ),
)

ALL_MODELS = FREEBUFF_MODELS + GEMINI_FREE_MODELS

# ── Anthropic model aliases ──────────────────────────────────────────
# Maps Anthropic-style model IDs to Freebuff model IDs.
# Clients using the /v1/messages endpoint can use either:
#   1. The native freebuff model id (e.g. "deepseek/deepseek-v4-flash")
#   2. One of these Anthropic aliases (e.g. "claude-sonnet-4-20250514")
ANTHROPIC_MODEL_ALIASES: dict[str, str] = {
    "claude-sonnet-4-20250514": DEFAULT_MODEL.id,
    "claude-3.5-sonnet": DEFAULT_MODEL.id,
    "claude-3-5-sonnet-20241022": DEFAULT_MODEL.id,
    "claude-3-haiku-20240307": DEFAULT_MODEL.id,
    "claude-opus-4-20250514": "deepseek/deepseek-v4-pro",
}


def _resolve_alias(requested: str) -> str | None:
    """Resolve an Anthropic model alias to a native model id.

    Returns the native id if the alias is known, *None* otherwise.
    """
    return ANTHROPIC_MODEL_ALIASES.get(requested)


def resolve_model(requested: str | None) -> FreebuffModel:
    if not requested:
        return DEFAULT_MODEL

    # Direct match.
    for model in ALL_MODELS:
        if model.id == requested:
            return model

    # Anthropic alias resolution.
    aliased = _resolve_alias(requested)
    if aliased is not None:
        for model in ALL_MODELS:
            if model.id == aliased:
                return model

    raise ValueError(f"Unsupported Freebuff model: {requested}")


def models_response() -> dict[str, object]:
    data: list[dict[str, object]] = []
    for model in ALL_MODELS:
        data.append({
            "id": model.id,
            "object": "model",
            "created": 0,
            "owned_by": model.owned_by,
        })
    for alias, native_id in ANTHROPIC_MODEL_ALIASES.items():
        owned_by = "anthropic"
        for model in ALL_MODELS:
            if model.id == native_id:
                owned_by = model.owned_by
                break
        data.append({
            "id": alias,
            "object": "model",
            "created": 0,
            "owned_by": owned_by,
        })
    return {"object": "list", "data": data}


def model_response(model_id: str) -> dict[str, object] | None:
    for model in ALL_MODELS:
        if model.id == model_id:
            return {
                "id": model.id,
                "object": "model",
                "created": 0,
                "owned_by": model.owned_by,
            }
    return None


def agent_validation_payload() -> dict[str, object]:
    models_by_agent: dict[str, FreebuffModel] = {}
    spawnable_by_agent: dict[str, set[str]] = {}
    for model in ALL_MODELS:
        models_by_agent.setdefault(model.agent_id, model)
        spawnable_by_agent.setdefault(model.agent_id, set()).add(CONTEXT_PRUNER_AGENT_ID)
        if model.parent_agent_id:
            spawnable_by_agent.setdefault(model.parent_agent_id, set()).add(model.agent_id)

    definitions = [
        _agent_definition(
            agent_id=model.agent_id,
            model_id=model.upstream_id,
            display_name=f"Freebuff {model.upstream_id}",
            spawnable_agents=sorted(spawnable_by_agent.get(model.agent_id, set())),
        )
        for model in models_by_agent.values()
    ]
    definitions.append(
        _agent_definition(
            agent_id=CONTEXT_PRUNER_AGENT_ID,
            model_id=DEFAULT_MODEL.id,
            display_name="Context Pruner",
            spawnable_agents=[],
        )
    )

    return {"agentDefinitions": definitions}


def _agent_definition(
    *,
    agent_id: str,
    model_id: str,
    display_name: str,
    spawnable_agents: list[str],
) -> dict[str, object]:
    return {
        "id": agent_id,
        "publisher": "codebuff",
        "model": model_id,
        "displayName": display_name,
        "spawnerPrompt": "Freebuff OpenAI-compatible orchestrator",
        "inputSchema": {
            "prompt": {
                "type": "string",
                "description": "A coding task to complete",
            },
            "params": {"type": "object", "properties": {}, "required": []},
        },
        "outputMode": "last_message",
        "includeMessageHistory": True,
        "toolNames": ["spawn_agents"] if spawnable_agents else [],
        "spawnableAgents": spawnable_agents,
        "systemPrompt": "Act as a helpful coding assistant.",
    }
