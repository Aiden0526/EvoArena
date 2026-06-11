"""Pure Terminus2 baseline for Harbor trials.

:class:`Terminus2Baseline` subclasses Harbor's ``Terminus2`` with **no**
run-loop overrides. It applies the same authoritative LLM env as EvoMem.
"""

from __future__ import annotations

from typing import Any

from harbor.agents.terminus_2 import Terminus2

from .terminus2_llm_sync import (
    apply_authoritative_llm_to_terminus2,
    enable_responses_api_for_terminus2_if_needed,
)


class Terminus2Baseline(Terminus2):
    """Upstream ``Terminus2`` semantics with authoritative-env LLM sync only."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        kwargs.pop("oh_config_path", None)
        kwargs.pop("oh_llm_block", None)
        super().__init__(*args, **kwargs)
        apply_authoritative_llm_to_terminus2(self)
        enable_responses_api_for_terminus2_if_needed(self, logger=self.logger)
