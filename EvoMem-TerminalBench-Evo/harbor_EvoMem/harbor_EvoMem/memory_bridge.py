"""Small shared helpers for Terminus2 + EvoMem."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


DEFAULT_HOST_ROOT = Path("~/.harbor_evomem").expanduser()


@dataclass
class LLMConfig:
    """Resolved LLM triple (agent + EvoMem summariser identical)."""

    model: Optional[str] = None
    api_key: Optional[str] = None
    base_url: Optional[str] = None


class MissingLLMConfigError(RuntimeError):
    """Raised when ``LLM_MODEL`` / ``LLM_API_KEY`` / ``LLM_BASE_URL`` are not all set."""

    pass


def is_openrouter_base_url(base_url: str | None) -> bool:
    """Return whether an OpenAI-compatible base URL is OpenRouter."""

    return "openrouter.ai" in str(base_url or "").lower()


def litellm_model_for_config(cfg: LLMConfig) -> str | None:
    """Return the LiteLLM routing model for callers that go through LiteLLM.

    ``LLM_MODEL`` remains the provider-facing model ID from
    ``terminus2_llm.env``. For OpenRouter, LiteLLM needs an additional
    ``openrouter/`` routing prefix; LiteLLM strips that prefix before sending
    the request to OpenRouter.
    """

    model = str(cfg.model or "").strip()
    if not model:
        return cfg.model
    if is_openrouter_base_url(cfg.base_url) and not model.startswith("openrouter/"):
        return f"openrouter/{model}"
    return model


def ensure_llm_resolved(cfg: LLMConfig) -> LLMConfig:
    """Require complete triple — no silent model/key/URL defaults anywhere."""

    missing: list[str] = []
    if not (cfg.model and str(cfg.model).strip()):
        missing.append("LLM_MODEL")
    if not (cfg.api_key and str(cfg.api_key).strip()):
        missing.append("LLM_API_KEY")
    if not (cfg.base_url and str(cfg.base_url).strip()):
        missing.append("LLM_BASE_URL")
    if missing:
        raise MissingLLMConfigError(
            f"Missing required LLM settings ({', '.join(missing)}) in "
            f"{terminus2_llm_env_path()}: set LLM_MODEL, LLM_API_KEY, and LLM_BASE_URL."
        )
    if any(
        str(value or "").startswith("REPLACE_WITH_")
        for value in (cfg.model, cfg.api_key, cfg.base_url)
    ):
        raise MissingLLMConfigError(
            f"{terminus2_llm_env_path()} still contains placeholder LLM values."
        )
    return cfg


def repo_root_from_package() -> Path:
    """Harbor-EvoMem checkout root (parent of ``harbor_EvoMem/`` package)."""

    return Path(__file__).resolve().parent.parent


def default_dataset_root() -> Path:
    """Default Terminal-Bench-Evo dataset directory (sibling of harbor_EvoMem checkout)."""

    return repo_root_from_package().parent / "Terminal-Bench-Evo"


DEFAULT_TERMINUS2_LLM_ENV_PATH = repo_root_from_package() / "scripts" / "terminus2_llm.env"
# Backward-compatible exported name. Runtime resolution happens in
# terminus2_llm_env_path() so HARBOR_EVOMEM_LLM_ENV can override it.
TERMINUS2_LLM_ENV_PATH = DEFAULT_TERMINUS2_LLM_ENV_PATH


def terminus2_llm_env_path() -> Path:
    """Return the authoritative local LLM env path for this checkout."""

    override = os.environ.get("HARBOR_EVOMEM_LLM_ENV")
    if override:
        return Path(override).expanduser()
    return DEFAULT_TERMINUS2_LLM_ENV_PATH


_ENV_LINE_RE = re.compile(
    r"^\s*(?:export\s+)?(?P<key>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*(?P<val>.*)\s*$"
)


def parse_env_file_assignments(path: Path) -> dict[str, str]:
    """Subset of shell env files: ``KEY=...`` and ``export KEY=...``, ``#`` comments."""

    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return {}

    out: dict[str, str] = {}

    def _strip_inline_comment(raw: str) -> str:
        quote = ""
        escaped = False
        chars: list[str] = []
        for ch in raw.strip():
            if escaped:
                chars.append(ch)
                escaped = False
                continue
            if quote and ch == "\\":
                chars.append(ch)
                escaped = True
                continue
            if ch in ("'", '"'):
                if not quote:
                    quote = ch
                elif quote == ch:
                    quote = ""
                chars.append(ch)
                continue
            if ch == "#" and not quote:
                break
            chars.append(ch)
        return "".join(chars).strip()

    def _strip_val(raw: str) -> str:
        s = _strip_inline_comment(raw)
        if not s:
            return ""
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            return s[1:-1]
        return s

    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _ENV_LINE_RE.match(stripped)
        if not m:
            continue
        out[m.group("key")] = _strip_val(m.group("val"))

    return out


def load_llm_from_env_file(path: Path) -> LLMConfig:
    """Parse ``LLM_MODEL`` / ``LLM_API_KEY`` / ``LLM_BASE_URL`` from a dotenv-format file."""

    data = parse_env_file_assignments(path)

    raw_base = (data.get("LLM_BASE_URL") or "").strip()
    api_raw = data.get("LLM_API_KEY")
    api_clean = api_raw.strip().strip('"').strip("'") if api_raw else None

    base_clean: Optional[str] = raw_base or None
    if raw_base == '""' or raw_base == "''":
        base_clean = None

    model_raw = data.get("LLM_MODEL")
    model_clean = model_raw.strip() if model_raw else None

    return LLMConfig(
        model=model_clean or None,
        api_key=api_clean or None,
        base_url=base_clean,
    )


def authoritative_llm_config() -> LLMConfig:
    """The only LLM triple: read the local env file — no Harbor LLM overrides."""

    path = terminus2_llm_env_path().resolve()
    if not path.is_file():
        raise MissingLLMConfigError(
            "Required LLM config file is missing:\n"
            f"  {path}\n"
            "Copy scripts/terminus2_llm.env.example to scripts/terminus2_llm.env "
            "and populate LLM_MODEL, LLM_API_KEY, and LLM_BASE_URL."
        )
    cfg = load_llm_from_env_file(path)
    return ensure_llm_resolved(cfg)


def resolve_llm_config(*_: Any, **__: Any) -> LLMConfig:
    """Backward-compatible alias — identical to :func:`authoritative_llm_config`."""

    return authoritative_llm_config()


def chain_root(chain_id: str, host_root: Path | str | None = None) -> Path:
    root = Path(host_root).expanduser() if host_root else DEFAULT_HOST_ROOT
    if "HARBOR_EVOMEM_HOST_ROOT" in os.environ and host_root is None:
        root = Path(os.environ["HARBOR_EVOMEM_HOST_ROOT"]).expanduser()
    safe = chain_id.replace("/", "_").replace("..", "_")
    target = root / safe
    target.mkdir(parents=True, exist_ok=True)
    return target


def _flatten_atif_content(value: Any) -> str:
    """Coerce an ATIF content field into plain text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text)
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(value)


def load_terminus2_history(trajectory_json: Path | str) -> list[dict[str, Any]]:
    """Translate a Terminus2 ATIF trajectory into EvoMem history events.

    EvoMem's extractor expects a compact message/tool event list. Terminus2
    writes ATIF ``trajectory.json`` instead, so we synthesize a
    compatible event stream from its steps:

    - agent text/reasoning -> ``source=agent`` message events
    - ``bash_command`` tool calls -> ``source=agent`` terminal actions
    - observations -> ``source=environment`` terminal results
    """
    path = Path(trajectory_json)
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    steps = data.get("steps") if isinstance(data, dict) else None
    if not isinstance(steps, list):
        return []

    history: list[dict[str, Any]] = []
    for step in steps:
        if not isinstance(step, dict):
            continue

        timestamp = step.get("timestamp")
        source = str(step.get("source") or "").lower()
        message = _flatten_atif_content(step.get("message"))
        reasoning = _flatten_atif_content(step.get("reasoning_content"))

        if source in {"system", "user"}:
            if message:
                history.append(
                    {
                        "timestamp": timestamp,
                        "source": source,
                        "type": "message",
                        "content": message,
                    }
                )
            continue

        if source != "agent":
            continue

        if message or reasoning:
            event: dict[str, Any] = {
                "timestamp": timestamp,
                "source": "agent",
                "type": "message",
                "content": message,
            }
            if reasoning:
                event["thought"] = reasoning
            history.append(event)

        bash_commands: list[str] = []
        for tool_call in step.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            fn = str(tool_call.get("function_name") or "")
            args = tool_call.get("arguments") if isinstance(tool_call.get("arguments"), dict) else {}
            if fn == "bash_command":
                command = str(args.get("keystrokes") or "").strip()
                if not command:
                    continue
                bash_commands.append(command)
                history.append(
                    {
                        "timestamp": timestamp,
                        "source": "agent",
                        "tool_name": "terminal",
                        "type": "tool_call",
                        "content": command,
                        "action": {"command": command},
                    }
                )
            elif fn == "mark_task_complete":
                history.append(
                    {
                        "timestamp": timestamp,
                        "source": "agent",
                        "type": "message",
                        "content": "task_complete",
                    }
                )

        observation = step.get("observation")
        if not isinstance(observation, dict):
            continue
        results = observation.get("results")
        if not isinstance(results, list):
            continue
        rendered_results = [
            _flatten_atif_content(result.get("content"))
            for result in results
            if isinstance(result, dict)
        ]
        rendered_results = [text for text in rendered_results if text]
        if not rendered_results:
            continue
        combined_output = "\n\n".join(rendered_results)
        history.append(
            {
                "timestamp": timestamp,
                "source": "environment",
                "tool_name": "terminal",
                "type": "tool_result",
                "content": combined_output,
                "observation": {
                    "command": " && ".join(bash_commands) if bash_commands else None,
                    "content": [{"text": combined_output}],
                    "exit_code": None,
                    "is_error": False,
                },
            }
        )
    return history


__all__ = [
    "DEFAULT_HOST_ROOT",
    "DEFAULT_TERMINUS2_LLM_ENV_PATH",
    "TERMINUS2_LLM_ENV_PATH",
    "LLMConfig",
    "MissingLLMConfigError",
    "is_openrouter_base_url",
    "litellm_model_for_config",
    "ensure_llm_resolved",
    "repo_root_from_package",
    "default_dataset_root",
    "terminus2_llm_env_path",
    "parse_env_file_assignments",
    "load_llm_from_env_file",
    "authoritative_llm_config",
    "resolve_llm_config",
    "chain_root",
    "load_terminus2_history",
]
