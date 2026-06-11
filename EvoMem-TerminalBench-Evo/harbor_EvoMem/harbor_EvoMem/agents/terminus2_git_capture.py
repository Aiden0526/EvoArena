"""Optional git diff capture around Terminus2."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from harbor.agents.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .. import patch_capture
from .terminus2_llm_sync import (
    apply_authoritative_llm_to_terminus2,
    enable_responses_api_for_terminus2_if_needed,
)


class Terminus2GitCapture(Terminus2):
    """Terminus2 wrapped with scratch-git baseline/diff capture for downstream tooling."""

    def __init__(
        self,
        workdir: str = patch_capture.DEFAULT_WORKDIR,
        enable_capture: bool = True,
        *args: Any,
        **kwargs: Any,
    ):
        # Ignored compat kwargs (historically wired to Terminus2 toml paths).
        kwargs.pop("oh_config_path", None)
        kwargs.pop("oh_llm_block", None)

        self._workdir = workdir
        self._enable_capture = _as_bool(enable_capture)
        self._capture: Optional[patch_capture.CapturedDiff] = None
        super().__init__(*args, **kwargs)
        apply_authoritative_llm_to_terminus2(self)
        enable_responses_api_for_terminus2_if_needed(self, logger=self.logger)

    @property
    def harbor_evomem_dir(self) -> Path:
        out = self.logs_dir / "harbor_evomem"
        out.mkdir(parents=True, exist_ok=True)
        return out

    async def _setup_baseline(self, environment: BaseEnvironment) -> Optional[str]:
        try:
            return await patch_capture.init_baseline(
                environment, workdir=self._workdir, logger=self.logger
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("patch baseline setup failed: %s", exc)
            return None

    async def _capture_post_run(
        self, environment: BaseEnvironment, baseline_sha: Optional[str]
    ) -> patch_capture.CapturedDiff:
        try:
            captured = await patch_capture.capture_diff(
                environment, workdir=self._workdir, logger=self.logger
            )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("patch capture failed: %s", exc)
            captured = patch_capture.CapturedDiff(
                diff_text="",
                changed_files=[],
                baseline_sha=baseline_sha,
                head_sha=None,
            )
        self._dump_capture(captured)
        return captured

    def _dump_capture(self, captured: patch_capture.CapturedDiff) -> None:
        out = self.harbor_evomem_dir
        try:
            (out / "diff.patch").write_text(captured.diff_text or "", encoding="utf-8")
            (out / "changed_files.txt").write_text(
                "\n".join(captured.changed_files) + ("\n" if captured.changed_files else ""),
                encoding="utf-8",
            )
            meta = {
                "workdir": self._workdir,
                "baseline_sha": captured.baseline_sha,
                "head_sha": captured.head_sha,
                "n_changed_files": len(captured.changed_files),
                "diff_chars": len(captured.diff_text or ""),
            }
            (out / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("failed to dump capture artefacts: %s", exc)

    async def run(  # type: ignore[override]
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if not self._enable_capture:
            await super().run(
                instruction=instruction,
                environment=environment,
                context=context,
            )
            return

        baseline_sha = await self._setup_baseline(environment)
        try:
            await super().run(
                instruction=instruction,
                environment=environment,
                context=context,
            )
        finally:
            self._capture = await self._capture_post_run(environment, baseline_sha)

    @property
    def captured(self) -> Optional[patch_capture.CapturedDiff]:
        return self._capture


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off", ""}
    return bool(value)
