"""Shared Terminus2 LLM wiring — always :func:`~harbor_EvoMem.memory_bridge.authoritative_llm_config`."""

from __future__ import annotations

import logging

from harbor.agents.utils import get_api_key_var_names_from_model_name
from harbor.llms.utils import split_provider_model_name
from harbor.llms.lite_llm import LiteLLM
from litellm.litellm_core_utils.get_supported_openai_params import (
    get_supported_openai_params,
)

from ..memory_bridge import (
    LLMConfig,
    authoritative_llm_config,
    litellm_model_for_config,
)

_RESPONSES_API_MODEL_PREFIXES = (
    "gpt-5",
    "openai/gpt-5",
)


def apply_authoritative_llm_to_terminus2(agent: object) -> None:
    cfg: LLMConfig = authoritative_llm_config()

    setattr(agent, "model_name", cfg.model)
    if hasattr(agent, "_model_name"):
        setattr(agent, "_model_name", cfg.model)

    llm_kw = dict(getattr(agent, "_llm_kwargs", None) or {})
    llm_kw["api_key"] = cfg.api_key
    setattr(agent, "_llm_kwargs", llm_kw)

    llm = getattr(agent, "_llm", None)
    if llm is not None:
        inner_kw = dict(getattr(llm, "_llm_kwargs", None) or {})
        inner_kw["api_key"] = cfg.api_key
        setattr(llm, "_llm_kwargs", inner_kw)  # type: ignore[attr-defined]

    merged_env = dict(getattr(agent, "_extra_env", None) or {})
    merged_env["LLM_API_KEY"] = cfg.api_key
    merged_env["LLM_MODEL"] = cfg.model
    merged_env["LLM_BASE_URL"] = cfg.base_url
    model_names = {str(cfg.model or ""), str(litellm_model_for_config(cfg) or "")}
    try:
        for model_name_str in model_names:
            for env_var in get_api_key_var_names_from_model_name(model_name_str):
                merged_env[env_var] = cfg.api_key
    except Exception:
        merged_env.setdefault("OPENAI_API_KEY", cfg.api_key)

    setattr(agent, "_extra_env", merged_env)
    if llm is not None:
        _sync_litellm_request_target(llm, cfg)
        _assert_litellm_uses_authoritative_config(llm, cfg)


def _sync_litellm_request_target(llm: object, cfg: LLMConfig) -> None:
    """Force the actual LiteLLM request object to use the authoritative triple."""

    litellm_request_model = litellm_model_for_config(cfg)
    setattr(llm, "_model_name", litellm_request_model)
    if hasattr(llm, "_llm_kwargs"):
        inner_kw = dict(getattr(llm, "_llm_kwargs", None) or {})
        inner_kw["api_key"] = cfg.api_key
        setattr(llm, "_llm_kwargs", inner_kw)
    setattr(llm, "_api_base", cfg.base_url)

    try:
        provider_prefix, canonical_model_name = split_provider_model_name(
            str(litellm_request_model)
        )
        setattr(llm, "_provider_prefix", provider_prefix)
        setattr(llm, "_canonical_model_name", canonical_model_name)
        litellm_model_name = (
            canonical_model_name
            if provider_prefix == "hosted_vllm"
            else litellm_request_model
        )
        setattr(llm, "_litellm_model_name", litellm_model_name)
        supported_params = get_supported_openai_params(str(litellm_model_name))
        setattr(llm, "_supported_params", supported_params)
        setattr(
            llm,
            "_supports_response_format",
            bool(supported_params and "response_format" in supported_params),
        )
        setattr(
            llm,
            "_supports_temperature",
            bool(supported_params and "temperature" in supported_params),
        )
    except Exception:
        # The request path still uses _model_name/api_base/api_key. These cached
        # lookup fields only improve parameter support detection.
        pass


def _assert_litellm_uses_authoritative_config(llm: object, cfg: LLMConfig) -> None:
    expected_model = litellm_model_for_config(cfg)
    model = getattr(llm, "_model_name", None)
    api_base = getattr(llm, "_api_base", None)
    llm_kwargs = getattr(llm, "_llm_kwargs", None) or {}
    api_key = llm_kwargs.get("api_key") if isinstance(llm_kwargs, dict) else None

    mismatches: list[str] = []
    if model != expected_model:
        mismatches.append(f"litellm_model={model!r}, expected={expected_model!r}")
    if api_base != cfg.base_url:
        mismatches.append(f"api_base={api_base!r}")
    if api_key != cfg.api_key:
        mismatches.append("api_key=<not authoritative>")
    if mismatches:
        raise RuntimeError(
            "Terminus2 LiteLLM request target does not match "
            f"{authoritative_llm_config.__name__}(): {', '.join(mismatches)}"
        )


def _resolved_model_name(agent: object) -> str:
    return str(
        getattr(agent, "model_name", None) or getattr(agent, "_model_name", "") or "",
    )


def enable_responses_api_for_terminus2_if_needed(
    agent: object,
    *,
    logger: logging.Logger | None = None,
) -> None:
    mn = _resolved_model_name(agent).lower()
    if not any(mn.startswith(prefix) for prefix in _RESPONSES_API_MODEL_PREFIXES):
        return
    llm = getattr(agent, "_llm", None)
    if not isinstance(llm, LiteLLM):
        return
    if getattr(llm, "_use_responses_api", False):
        return
    llm._use_responses_api = True
    if logger:
        logger.info(
            "terminus2: enabling LiteLLM Responses API for model=%s",
            _resolved_model_name(agent),
        )


__all__ = [
    "apply_authoritative_llm_to_terminus2",
    "enable_responses_api_for_terminus2_if_needed",
]
