"""
Robust A-MEM memory layer — drop-in replacement for memory_layer.py.

Key differences from the original:
  - No response_format / JSON schema dependency in LLM calls
  - Plain-text prompts with section-marker parsing (via llm_text_parsers)
  - Structured logging instead of print()
  - Retry wrapper for transient LLM failures
  - Connectivity check on controller init
  - Graceful degradation: evolution failure -> memory stored without evolution
"""

from typing import List, Dict, Optional, Literal, Any
from contextlib import contextmanager
import contextvars
import json
import re
import uuid
import os
import time
import logging
import functools
import hashlib
from datetime import datetime
from abc import ABC, abstractmethod

from memory_layer import SimpleEmbeddingRetriever, simple_tokenize
from llm_text_parsers import (
    ANALYZE_CONTENT_PROMPT,
    ANALYZE_CONTENT_PREF_PROMPT,
    EVOLUTION_DECISION_PROMPT,
    EVOLUTION_DECISION_PREF_PROMPT,
    STRENGTHEN_DETAILS_PROMPT,
    STRENGTHEN_DETAILS_PREF_PROMPT,
    UPDATE_NEIGHBORS_PROMPT,
    UPDATE_NEIGHBORS_PREF_PROMPT,
    FOCUSED_KEYWORDS_PROMPT,
    parse_analyze_content,
    parse_evolution_decision,
    parse_strengthen_details,
    parse_update_neighbors,
    validate_analysis_result,
)

logger = logging.getLogger("amem_robust")
raw_logger = logging.getLogger("amem_robust_raw")
_usage_stage_var: contextvars.ContextVar[str] = contextvars.ContextVar("amem_llm_usage_stage", default="unknown")
_usage_recorder_var: contextvars.ContextVar[Optional["LLMUsageRecorder"]] = contextvars.ContextVar(
    "amem_llm_usage_recorder", default=None
)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump())
    if hasattr(value, "dict"):
        return _jsonable(value.dict())
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return str(value)


def _deep_get(mapping: Any, path: List[str], default: Any = None) -> Any:
    current = mapping
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        else:
            current = getattr(current, key, None)
        if current is None:
            return default
    return current


def _number(value: Any) -> float:
    if value is None:
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _env_enabled(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _openrouter_cache_control() -> Dict[str, str]:
    cache_control = {"type": "ephemeral"}
    ttl = os.getenv("OPENROUTER_PROMPT_CACHE_TTL", "").strip()
    if ttl:
        cache_control["ttl"] = ttl
    return cache_control


def _openrouter_prompt_cache_enabled(model: str) -> bool:
    return model.startswith("openrouter/") and _env_enabled("OPENROUTER_PROMPT_CACHE")


def is_nonrecoverable_llm_error(exc: BaseException) -> bool:
    """Return True for configuration/auth errors that should stop a run."""
    text = f"{exc.__class__.__module__}.{exc.__class__.__name__}: {exc}"
    markers = (
        "AuthenticationError",
        "Unauthorized",
        "User not found",
        '"code":401',
        "code': 401",
        "LLM Provider NOT provided",
        "BadRequestError",
        "Provider List:",
        '"code":403',
        "code': 403",
        "Key limit exceeded",
        "total limit",
    )
    return any(marker in text for marker in markers)


class LLMUsageRecorder:
    """Append-only per-call token usage recorder for provider-reported usage."""

    def __init__(self, path: str | os.PathLike[str], metadata: Optional[Dict[str, Any]] = None):
        self.path = os.fspath(path)
        self.metadata = metadata or {}
        self.records: List[Dict[str, Any]] = []
        self.totals: Dict[str, Dict[str, float]] = {}
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)

    def _bucket(self, name: str) -> Dict[str, float]:
        if name not in self.totals:
            self.totals[name] = {
                "calls": 0.0,
                "agent_steps": 0.0,
                "prompt_tokens": 0.0,
                "completion_tokens": 0.0,
                "total_tokens": 0.0,
                "reasoning_tokens": 0.0,
                "cached_tokens": 0.0,
                "cache_write_tokens": 0.0,
                "response_cost": 0.0,
            }
        return self.totals[name]

    def record(self, entry: Dict[str, Any]) -> None:
        usage = entry.get("usage") or {}
        stage = str(entry.get("stage") or "unknown")
        stage_group = stage.split(".", 1)[0]
        prompt_tokens = _number(usage.get("prompt_tokens") or usage.get("input_tokens"))
        completion_tokens = _number(usage.get("completion_tokens") or usage.get("output_tokens"))
        total_tokens = _number(usage.get("total_tokens")) or prompt_tokens + completion_tokens
        reasoning_tokens = _number(
            _deep_get(usage, ["completion_tokens_details", "reasoning_tokens"])
            or _deep_get(usage, ["output_tokens_details", "reasoning_tokens"])
            or usage.get("reasoning_tokens")
        )
        cached_tokens = _number(
            _deep_get(usage, ["prompt_tokens_details", "cached_tokens"])
            or _deep_get(usage, ["input_tokens_details", "cached_tokens"])
            or usage.get("cached_tokens")
        )
        cache_write_tokens = _number(
            _deep_get(usage, ["prompt_tokens_details", "cache_write_tokens"])
            or _deep_get(usage, ["input_tokens_details", "cache_write_tokens"])
            or usage.get("cache_write_tokens")
        )
        response_cost = _number(entry.get("response_cost"))

        for bucket_name in ("all", stage_group, stage):
            bucket = self._bucket(bucket_name)
            bucket["calls"] += 1
            bucket["agent_steps"] += 1
            bucket["prompt_tokens"] += prompt_tokens
            bucket["completion_tokens"] += completion_tokens
            bucket["total_tokens"] += total_tokens
            bucket["reasoning_tokens"] += reasoning_tokens
            bucket["cached_tokens"] += cached_tokens
            bucket["cache_write_tokens"] += cache_write_tokens
            bucket["response_cost"] += response_cost

        json_entry = _jsonable(entry)
        self.records.append(json_entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(json_entry, ensure_ascii=False) + "\n")

    def summary(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "usage_log": self.path,
            "totals": self.totals,
        }


@contextmanager
def llm_usage_recorder(recorder: Optional[LLMUsageRecorder]):
    token = _usage_recorder_var.set(recorder)
    try:
        yield recorder
    finally:
        _usage_recorder_var.reset(token)


@contextmanager
def llm_usage_stage(stage: str):
    token = _usage_stage_var.set(stage)
    try:
        yield
    finally:
        _usage_stage_var.reset(token)


def _extract_response_cost(response: Any) -> Optional[float]:
    hidden_params = getattr(response, "_hidden_params", None)
    if hidden_params is None and isinstance(response, dict):
        hidden_params = response.get("_hidden_params")
    cost = _deep_get(hidden_params, ["response_cost"])
    if cost is None:
        cost = _deep_get(hidden_params, ["additional_headers", "llm_provider-x-litellm-response-cost"])
    try:
        return float(cost) if cost is not None else None
    except (TypeError, ValueError):
        return None


def record_llm_response_usage(
    *,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: Optional[int],
    response: Any,
    elapsed: float,
    response_chars: int,
) -> None:
    recorder = _usage_recorder_var.get()
    if recorder is None:
        return

    usage = _jsonable(getattr(response, "usage", None) or (response.get("usage") if isinstance(response, dict) else None) or {})
    entry = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "stage": _usage_stage_var.get(),
        "model": model,
        "prompt_chars": len(prompt),
        "prompt_md5": hashlib.md5(prompt.encode("utf-8")).hexdigest(),
        "temperature": temperature,
        "max_tokens": max_tokens,
        "elapsed_seconds": round(elapsed, 4),
        "response_chars": response_chars,
        "usage": usage,
        "response_cost": _extract_response_cost(response),
    }
    recorder.record(entry)


def require_text_completion_content(response, model: str) -> str:
    try:
        choice = response.choices[0]
        message = choice.message
        content = message.content
    except Exception as exc:
        raise RuntimeError(f"Malformed completion response from {model}: {exc}") from exc
    if content is None:
        finish_reason = getattr(choice, "finish_reason", None)
        tool_calls = getattr(message, "tool_calls", None)
        raise RuntimeError(
            f"LLM returned no text content for model={model}; finish_reason={finish_reason!r}; tool_calls={tool_calls!r}"
        )
    return content


def normalize_openai_compatible_base_url(api_base: Optional[str]) -> Optional[str]:
    """
    Normalize an OpenAI-compatible base URL.

    Accepts either an API root such as:
      https://host/v1
    or a full chat completions endpoint such as:
      https://host/v1/chat/completions

    Returns the API root expected by the OpenAI client / LiteLLM.
    """
    if api_base is None:
        return None

    normalized = api_base.strip().rstrip("/")
    if not normalized:
        return None

    suffix = "/chat/completions"
    if normalized.endswith(suffix):
        normalized = normalized[: -len(suffix)]

    return normalized


def is_salesforce_gateway_base_url(api_base: Optional[str]) -> bool:
    if not api_base:
        return False
    normalized = normalize_openai_compatible_base_url(api_base) or ""
    return "gateway.salesforceresearch.ai/openai/process/" in normalized


def normalize_litellm_model_name(backend: str, model: str) -> str:
    """Normalize provider-qualified model names for LiteLLM backends."""
    normalized = model.strip()
    if backend == "openrouter" and normalized and not normalized.startswith("openrouter/"):
        return f"openrouter/{normalized}"
    return normalized

# ---------------------------------------------------------------------------
# Retry decorator
# ---------------------------------------------------------------------------

def retry_llm_call(max_retries: int = 2, base_delay: float = 1.0):
    """Decorator: retry an LLM call with exponential backoff."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if is_nonrecoverable_llm_error(e):
                        logger.error("LLM call %s failed with non-recoverable error: %s", func.__name__, e)
                        raise
                    last_exc = e
                    if attempt < max_retries:
                        delay = base_delay * (2 ** attempt)
                        logger.warning(
                            "LLM call %s failed (attempt %d/%d): %s — retrying in %.1fs",
                            func.__name__, attempt + 1, max_retries + 1, e, delay,
                        )
                        time.sleep(delay)
            logger.error("LLM call %s failed after %d attempts: %s",
                         func.__name__, max_retries + 1, last_exc)
            raise last_exc
        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Robust LLM Controllers — no response_format parameter
# ---------------------------------------------------------------------------

class RobustBaseLLMController(ABC):
    """Base class for robust LLM controllers (no JSON schema dependency)."""

    SYSTEM_MESSAGE = "Follow the format specified in the prompt exactly. Do not add extra commentary."

    @abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        """Get a plain-text completion from the LLM."""
        pass

    def check_connectivity(self):
        """Send a test call to verify the backend is reachable."""
        try:
            response = self.get_completion("Reply with exactly one word: READY", temperature=0.0)
            if not response or not response.strip():
                raise ConnectionError("Empty response from LLM backend")
            logger.info("LLM connectivity check passed (response: %s)", response.strip()[:50])
        except Exception as e:
            raise ConnectionError(
                f"Cannot reach LLM backend: {e}. "
                "Check that the server is running and accessible."
            ) from e


class RobustOpenAIController(RobustBaseLLMController):
    def __init__(self, model: str = "gpt-4", api_key: Optional[str] = None, api_base: Optional[str] = None):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("OpenAI package not found. Install it with: pip install openai")
        self.model = model
        normalized_api_base = normalize_openai_compatible_base_url(
            api_base or os.getenv("OPENAI_BASE_URL") or os.getenv("PPAPI_BASE_URL")
        )
        uses_salesforce_gateway = is_salesforce_gateway_base_url(normalized_api_base)

        if api_key is None:
            api_key = (
                os.getenv('OPENAI_API_KEY')
                or os.getenv('PPAPI_API_KEY')
                or os.getenv('OPENAI_KEY')
                or os.getenv('X_API_KEY')
            )
        if api_key is None:
            raise ValueError(
                "OpenAI-compatible API key not found. Set OPENAI_API_KEY, PPAPI_API_KEY, or X_API_KEY."
            )

        client_kwargs = {}
        if uses_salesforce_gateway:
            client_kwargs["api_key"] = "dummy"
            client_kwargs["default_headers"] = {"X-Api-Key": api_key}
        else:
            client_kwargs["api_key"] = api_key

        if normalized_api_base:
            client_kwargs["base_url"] = normalized_api_base
        self.client = OpenAI(**client_kwargs)
        self.base_url = normalized_api_base or "https://api.openai.com/v1"
        logger.info(
            "RobustOpenAIController initialized model=%s base_url=%s auth_mode=%s",
            self.model, self.base_url, "x-api-key" if uses_salesforce_gateway else "bearer",
        )

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 1000) -> str:
        prompt_digest = hashlib.md5(prompt.encode("utf-8")).hexdigest()[:8]
        prompt_preview = " ".join(prompt.strip().split())[:120]
        start_time = time.time()
        logger.info(
            "LLM request start model=%s digest=%s prompt_chars=%d temperature=%.2f max_tokens=%d preview=%r",
            self.model, prompt_digest, len(prompt), temperature, max_tokens, prompt_preview,
        )
        if raw_logger.handlers:
            raw_logger.info(
                "REQUEST model=%s digest=%s temperature=%.2f max_tokens=%d\nPROMPT:\n%s",
                self.model, prompt_digest, temperature, max_tokens, prompt,
            )
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_completion_tokens=max_tokens,
            timeout=180.0,
        )
        elapsed = time.time() - start_time
        content = require_text_completion_content(response, self.model)
        record_llm_response_usage(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response=response,
            elapsed=elapsed,
            response_chars=len(content or ""),
        )
        logger.info(
            "LLM request done model=%s digest=%s elapsed=%.2fs response_chars=%d",
            self.model, prompt_digest, elapsed, len(content or ""),
        )
        if raw_logger.handlers:
            raw_logger.info(
                "RESPONSE model=%s digest=%s elapsed=%.2fs\n%s",
                self.model, prompt_digest, elapsed, content or "",
            )
        return content


class RobustOllamaController(RobustBaseLLMController):
    """Direct Ollama library controller (no LiteLLM proxy)."""

    def __init__(self, model: str = "llama2"):
        self.model = model

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        try:
            from ollama import chat
        except ImportError:
            raise ImportError("ollama package not found. Install it with: pip install ollama")
        response = chat(
            model=self.model,
            messages=[
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            options={"temperature": temperature},
        )
        return response["message"]["content"]


class RobustSGLangController(RobustBaseLLMController):
    def __init__(self, model: str = "llama2",
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = f"{sglang_host}:{sglang_port}"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "text": prompt,
            "sampling_params": {
                "temperature": temperature,
                "max_new_tokens": 1000,
            }
        }
        response = self._requests.post(
            f"{self.base_url}/generate",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=60,
        )
        if response.status_code == 200:
            return response.json().get("text", "")
        raise RuntimeError(f"SGLang server returned status {response.status_code}: {response.text}")


class RobustVLLMController(RobustBaseLLMController):
    """Controller for vLLM's OpenAI-compatible API server."""

    def __init__(self, model: str = "llama2",
                 api_base: Optional[str] = None,
                 vllm_host: str = "http://localhost",
                 vllm_port: int = 30000):
        import requests as _requests
        self._requests = _requests
        self.model = model
        self.base_url = normalize_openai_compatible_base_url(api_base) or f"{vllm_host}:{vllm_port}/v1"

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7) -> str:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt},
            ],
            "temperature": temperature,
            "max_tokens": 1000,
        }
        response = self._requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Content-Type": "application/json"},
            json=payload,
            timeout=120,
        )
        if response.status_code == 200:
            return response.json()["choices"][0]["message"]["content"]
        raise RuntimeError(f"vLLM server returned status {response.status_code}: {response.text}")


class RobustLiteLLMController(RobustBaseLLMController):
    """LiteLLM controller for universal LLM access (Ollama, SGLang, etc.)."""

    def __init__(self, model: str, api_base: Optional[str] = None,
                 api_key: Optional[str] = None):
        from litellm import completion as _completion
        self._completion = _completion
        self.model = model
        self.api_base = normalize_openai_compatible_base_url(api_base)
        self.api_key = (
            api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("PPAPI_API_KEY")
            or os.getenv("OPENAI_KEY")
            or "EMPTY"
        )

    @retry_llm_call(max_retries=2)
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: Optional[int] = None) -> str:
        completion_args = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.SYSTEM_MESSAGE},
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
        }
        if max_tokens is not None:
            completion_args["max_completion_tokens"] = max_tokens
        if self.api_base:
            completion_args["api_base"] = self.api_base
        if self.api_key:
            completion_args["api_key"] = self.api_key
        if _openrouter_prompt_cache_enabled(self.model):
            completion_args["cache_control"] = _openrouter_cache_control()

        start_time = time.time()
        response = self._completion(**completion_args)
        elapsed = time.time() - start_time
        content = require_text_completion_content(response, self.model)
        record_llm_response_usage(
            model=self.model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            response=response,
            elapsed=elapsed,
            response_chars=len(content or ""),
        )
        return content


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

class RobustLLMController:
    """Factory that selects the right robust LLM controller."""

    def __init__(self,
                 backend: Literal["openai", "openrouter", "ollama", "sglang", "vllm"] = "sglang",
                 model: str = "gpt-4",
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False):
        if backend == "openai":
            self.llm = RobustOpenAIController(model, api_key, api_base)
        elif backend == "openrouter":
            self.llm = RobustLiteLLMController(
                model=normalize_litellm_model_name("openrouter", model),
                api_base=api_base or "https://openrouter.ai/api/v1",
                api_key=api_key or os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY") or "EMPTY",
            )
        elif backend == "ollama":
            self.llm = RobustOllamaController(model)
        elif backend == "sglang":
            self.llm = RobustSGLangController(model, sglang_host, sglang_port)
        elif backend == "vllm":
            self.llm = RobustVLLMController(model, api_base, sglang_host, sglang_port)
        else:
            raise ValueError("Backend must be 'openai', 'openrouter', 'ollama', 'sglang', or 'vllm'")

        if check_connection:
            self.llm.check_connectivity()


# ---------------------------------------------------------------------------
# RobustMemoryNote
# ---------------------------------------------------------------------------

class RobustMemoryNote:
    """Memory note that uses plain-text LLM calls for metadata extraction."""

    def __init__(self,
                 content: str,
                 id: Optional[str] = None,
                 keywords: Optional[List[str]] = None,
                 links: Optional[Dict] = None,
                 importance_score: Optional[float] = None,
                 retrieval_count: Optional[int] = None,
                 timestamp: Optional[str] = None,
                 last_accessed: Optional[str] = None,
                 context: Optional[str] = None,
                 evolution_history: Optional[List] = None,
                 category: Optional[str] = None,
                 tags: Optional[List[str]] = None,
                 llm_controller: Optional[RobustLLMController] = None,
                 is_preference: bool = False,
                 pref_domain: str = "",
                 pref_holder: str = "none",
                 change_type: str = "none",
                 analyze_prompt: Optional[str] = None):

        self.content = content

        if llm_controller and any(p is None for p in [keywords, context, category, tags]):
            analysis = self.analyze_content(content, llm_controller, analyze_prompt)
            logger.debug("analysis result: %s", analysis)
            keywords = keywords or analysis["keywords"]
            context = context or analysis["context"]
            tags = tags or analysis["tags"]
            # Preference metadata from analysis (only override if not explicitly passed)
            if not is_preference:
                is_preference = analysis.get("is_preference", False)
            if not pref_domain:
                pref_domain = analysis.get("pref_domain", "")
            if pref_holder == "none":
                pref_holder = analysis.get("pref_holder", "none")
            if change_type == "none":
                change_type = analysis.get("change_type", "none")
            # Validate ask_to_forget: must be an explicit imperative command directed at the assistant.
            # Narratives like "I used to like X but now prefer Y" are NOT ask_to_forget.
            if change_type == "ask_to_forget":
                _forget_imperatives = [
                    "forget", "stop treating", "ignore", "remove",
                    "don't consider", "do not consider", "no longer consider",
                    "don't remember", "do not remember", "don't keep", "don't store",
                    "please don't", "stop remembering",
                ]
                _content_lower = content.lower()
                if not any(imp in _content_lower for imp in _forget_imperatives):
                    # Downgrade: if a new preference is stated, it's a replacement; otherwise a flip
                    if any(w in _content_lower for w in ["prefer", "like", "love", "enjoy", "switched", "moved on", "into"]):
                        change_type = "object_replacement"
                    else:
                        change_type = "same_object_flip"

        self.id = id or str(uuid.uuid4())
        self.keywords = keywords or []
        self.links = links or []
        self.importance_score = importance_score or 1.0
        self.retrieval_count = retrieval_count or 0
        current_time = datetime.now().strftime("%Y%m%d%H%M")
        self.timestamp = timestamp or current_time
        self.last_accessed = last_accessed or current_time

        self.context = context or "General"
        if isinstance(self.context, list):
            self.context = " ".join(self.context)

        self.evolution_history = evolution_history or []
        self.category = category or "Uncategorized"
        self.tags = tags or []

        # Preference metadata
        self.is_preference: bool = is_preference
        self.pref_domain: str = pref_domain or ""
        self.pref_holder: str = pref_holder or "none"
        self.change_type: str = change_type or "none"

    @staticmethod
    def analyze_content(content: str, llm_controller: RobustLLMController, analyze_prompt: str = None) -> Dict:
        """Analyze content using plain-text prompt + section-marker parsing."""
        prompt = (analyze_prompt or ANALYZE_CONTENT_PROMPT).format(content=content)
        try:
            response = llm_controller.llm.get_completion(prompt)
            analysis = parse_analyze_content(response, content)

            # If keywords still empty after parsing, try focused retry
            if not analysis["keywords"]:
                logger.info("Keywords empty after initial parse — retrying with focused prompt")
                retry_prompt = FOCUSED_KEYWORDS_PROMPT.format(content=content)
                retry_response = llm_controller.llm.get_completion(retry_prompt, temperature=0.3)
                from llm_text_parsers import _parse_list_items
                analysis["keywords"] = _parse_list_items(retry_response)

            # Final validation
            analysis = validate_analysis_result(analysis, content)
            return analysis

        except Exception as e:
            if is_nonrecoverable_llm_error(e):
                raise
            logger.error("Error analyzing content: %s", e)
            # Graceful degradation: heuristic keywords/context
            from llm_text_parsers import _heuristic_keywords, _heuristic_context
            return {
                "keywords": _heuristic_keywords(content),
                "context": _heuristic_context(content),
                "tags": _heuristic_keywords(content, 3),
            }


# ---------------------------------------------------------------------------
# RobustAgenticMemorySystem
# ---------------------------------------------------------------------------

class RobustAgenticMemorySystem:
    """Memory management system using plain-text LLM calls (no JSON schema)."""

    def __init__(self,
                 model_name: str = 'all-MiniLM-L6-v2',
                 llm_backend: str = "sglang",
                 llm_model: str = "gpt-4o-mini",
                 evo_threshold: int = 100,
                 api_key: Optional[str] = None,
                 api_base: Optional[str] = None,
                 sglang_host: str = "http://localhost",
                 sglang_port: int = 30000,
                 check_connection: bool = False,
                 preference_aware: Optional[bool] = None,
                 preference_aware_level: Optional[str] = None):

        self.memories: Dict[str, RobustMemoryNote] = {}
        self.retriever = SimpleEmbeddingRetriever(model_name)
        self.llm_controller = RobustLLMController(
            llm_backend, llm_model, api_key, api_base,
            sglang_host, sglang_port, check_connection,
        )
        self.evo_cnt = 0
        self.evo_threshold = evo_threshold

        # Resolve preference_aware_level. Accept the legacy `preference_aware`
        # bool kwarg and map it to the equivalent level for backward compat.
        if preference_aware_level is None:
            if preference_aware is True:
                preference_aware_level = "full"
            else:
                preference_aware_level = "none"
        if preference_aware_level not in ("none", "patch_only", "full"):
            raise ValueError(
                f"preference_aware_level must be one of none|patch_only|full, got {preference_aware_level!r}"
            )
        self.preference_aware_level = preference_aware_level
        self.preference_aware = preference_aware_level != "none"

        use_pref_analyze = preference_aware_level in ("patch_only", "full")
        use_pref_graph = preference_aware_level == "full"

        self._analyze_prompt = ANALYZE_CONTENT_PREF_PROMPT if use_pref_analyze else ANALYZE_CONTENT_PROMPT
        self._evolution_prompt = EVOLUTION_DECISION_PREF_PROMPT if use_pref_graph else EVOLUTION_DECISION_PROMPT
        self._strengthen_prompt = STRENGTHEN_DETAILS_PREF_PROMPT if use_pref_graph else STRENGTHEN_DETAILS_PROMPT
        self._update_neighbors_prompt = UPDATE_NEIGHBORS_PREF_PROMPT if use_pref_graph else UPDATE_NEIGHBORS_PROMPT

    # ---- public API (mirrors AgenticMemorySystem) ----

    def add_note(self, content: str, time: str = None, **kwargs) -> str:
        """Add a new memory note."""
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            analyze_prompt=self._analyze_prompt,
            **kwargs,
        )
        evo_label, note = self.process_memory(note)
        self.memories[note.id] = note
        self.retriever.add_documents([
            "content:" + note.content +
            " context:" + note.context +
            " keywords: " + ", ".join(note.keywords) +
            " tags: " + ", ".join(note.tags)
        ])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id

    def add_note_with_trace(self, content: str, time: str = None, **kwargs) -> tuple:
        """Add a new memory note and return an evolution trace for patch mode."""
        note = RobustMemoryNote(
            content=content,
            llm_controller=self.llm_controller,
            timestamp=time,
            analyze_prompt=self._analyze_prompt,
            **kwargs,
        )
        evo_label, note, trace = self.process_memory_with_trace(note)
        self.memories[note.id] = note
        self.retriever.add_documents([
            "content:" + note.content +
            " context:" + note.context +
            " keywords: " + ", ".join(note.keywords) +
            " tags: " + ", ".join(note.tags)
        ])
        if evo_label:
            self.evo_cnt += 1
            if self.evo_cnt % self.evo_threshold == 0:
                self.consolidate_memories()
        return note.id, trace

    def consolidate_memories(self):
        """Re-initialize the retriever with current memory state."""
        try:
            model_name = self.retriever.model.get_config_dict()['model_name']
        except (AttributeError, KeyError):
            model_name = 'all-MiniLM-L6-v2'

        self.retriever = SimpleEmbeddingRetriever(model_name)
        for memory in self.memories.values():
            metadata_text = f"{memory.context} {' '.join(memory.keywords)} {' '.join(memory.tags)}"
            self.retriever.add_documents([memory.content + " , " + metadata_text])

    def find_related_memories(self, query: str, k: int = 5) -> tuple:
        """Find related memories using embedding retrieval."""
        if not self.memories:
            return "", []

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            memory_str += (
                "memory index:" + str(i) +
                "\t talk start time:" + all_memories[i].timestamp +
                "\t memory content: " + all_memories[i].content +
                "\t memory context: " + all_memories[i].context +
                "\t memory keywords: " + str(all_memories[i].keywords) +
                "\t memory tags: " + str(all_memories[i].tags) + "\n"
            )
        return memory_str, indices

    def find_related_memories_raw(self, query: str, k: int = 5) -> str:
        """Find related memories with neighborhood expansion."""
        if not self.memories:
            return ""

        indices = self.retriever.search(query, k)
        all_memories = list(self.memories.values())
        memory_str = ""
        for i in indices:
            j = 0
            memory_str += (
                "talk start time:" + all_memories[i].timestamp +
                "memory content: " + all_memories[i].content +
                "memory context: " + all_memories[i].context +
                "memory keywords: " + str(all_memories[i].keywords) +
                "memory tags: " + str(all_memories[i].tags) + "\n"
            )
            neighborhood = all_memories[i].links
            for neighbor in neighborhood:
                memory_str += (
                    "talk start time:" + all_memories[neighbor].timestamp +
                    "memory content: " + all_memories[neighbor].content +
                    "memory context: " + all_memories[neighbor].context +
                    "memory keywords: " + str(all_memories[neighbor].keywords) +
                    "memory tags: " + str(all_memories[neighbor].tags) + "\n"
                )
                if j >= k:
                    break
                j += 1
        return memory_str

    # ---- evolution (3 sequential plain-text calls) ----

    def process_memory(self, note: RobustMemoryNote) -> tuple:
        """Process a memory note for evolution using plain-text LLM calls.

        Uses up to 3 sequential calls (conditional):
          1. Evolution decision
          2. Strengthen details (skip if no strengthen)
          3. Update neighbors (skip if no update)
        """
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)

        if len(indices) == 0:
            return False, note

        try:
            # ---- Call 1: Evolution decision ----
            decision_prompt = self._evolution_prompt.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            logger.info("process_memory note=%s stage=evolution_decision neighbors=%d", note.id, len(indices))
            decision_response = self.llm_controller.llm.get_completion(decision_prompt)
            decision = parse_evolution_decision(decision_response)
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            # ---- Call 2: Strengthen details (conditional) ----
            if should_strengthen:
                strengthen_prompt = self._strengthen_prompt.format(
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                logger.info("process_memory note=%s stage=strengthen_details", note.id)
                strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
                strengthen = parse_strengthen_details(strengthen_response)
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            # ---- Call 3: Update neighbors (conditional) ----
            if should_update:
                update_prompt = self._update_neighbors_prompt.format(
                    content=note.content,
                    context=note.context,
                    nearest_neighbors_memories=neighbor_memory,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                logger.info("process_memory note=%s stage=update_neighbors", note.id)
                update_response = self.llm_controller.llm.get_completion(update_prompt)
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note

        except Exception as e:
            if is_nonrecoverable_llm_error(e):
                raise
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            return False, note

    def process_memory_with_trace(self, note: RobustMemoryNote) -> tuple:
        """Process a memory note and return a detailed evolution trace."""
        neighbor_memory, indices = self.find_related_memories(note.content, k=5)
        trace = {
            "neighbor_memory": neighbor_memory,
            "neighbor_indices": indices,
            "decision_prompt": None,
            "decision_response": None,
            "decision_parsed": None,
            "strengthen_prompt": None,
            "strengthen_response": None,
            "strengthen_parsed": None,
            "update_prompt": None,
            "update_response": None,
            "neighbor_updates_parsed": None,
        }

        if len(indices) == 0:
            return False, note, trace

        try:
            decision_prompt = self._evolution_prompt.format(
                context=note.context,
                content=note.content,
                keywords=note.keywords,
                nearest_neighbors_memories=neighbor_memory,
            )
            trace["decision_prompt"] = decision_prompt
            decision_response = self.llm_controller.llm.get_completion(decision_prompt)
            trace["decision_response"] = decision_response
            decision = parse_evolution_decision(decision_response)
            trace["decision_parsed"] = decision
            logger.debug("Evolution decision: %s", decision)

            if decision["decision"] == "NO_EVOLUTION":
                return False, note, trace

            should_strengthen = decision["decision"] in ("STRENGTHEN", "STRENGTHEN_AND_UPDATE")
            should_update = decision["decision"] in ("UPDATE_NEIGHBOR", "STRENGTHEN_AND_UPDATE")

            if should_strengthen:
                strengthen_prompt = self._strengthen_prompt.format(
                    content=note.content,
                    keywords=note.keywords,
                    nearest_neighbors_memories=neighbor_memory,
                )
                trace["strengthen_prompt"] = strengthen_prompt
                strengthen_response = self.llm_controller.llm.get_completion(strengthen_prompt)
                trace["strengthen_response"] = strengthen_response
                strengthen = parse_strengthen_details(strengthen_response)
                trace["strengthen_parsed"] = strengthen
                logger.debug("Strengthen details: %s", strengthen)

                note.links.extend(strengthen["connections"])
                if strengthen["tags"]:
                    note.tags = strengthen["tags"]

            if should_update:
                update_prompt = self._update_neighbors_prompt.format(
                    content=note.content,
                    context=note.context,
                    nearest_neighbors_memories=neighbor_memory,
                    max_neighbor_idx=len(indices) - 1,
                    neighbor_count=len(indices),
                )
                trace["update_prompt"] = update_prompt
                update_response = self.llm_controller.llm.get_completion(update_prompt)
                trace["update_response"] = update_response
                neighbor_updates = parse_update_neighbors(update_response, len(indices))
                trace["neighbor_updates_parsed"] = neighbor_updates
                logger.debug("Neighbor updates: %s", neighbor_updates)

                noteslist = list(self.memories.values())
                notes_id = list(self.memories.keys())
                for i in range(min(len(indices), len(neighbor_updates))):
                    upd = neighbor_updates[i]
                    memorytmp_idx = indices[i]
                    if memorytmp_idx >= len(noteslist):
                        continue
                    notetmp = noteslist[memorytmp_idx]
                    if upd["tags"]:
                        notetmp.tags = upd["tags"]
                    if upd["context"]:
                        notetmp.context = upd["context"]
                    self.memories[notes_id[memorytmp_idx]] = notetmp

            return True, note, trace

        except Exception as e:
            if is_nonrecoverable_llm_error(e):
                raise
            logger.error("Evolution failed for note %s: %s — storing without evolution", note.id, e)
            trace["error"] = str(e)
            return False, note, trace
