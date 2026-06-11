"""Harbor agent: Terminus2 + EvoMem."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from .. import family_trace_memory, memory_bridge
from ..chain_id import derive_chain_id
from .terminus2_git_capture import Terminus2GitCapture, _as_bool


MEMORY_BLOCK_OPEN = "<!-- BEGIN EVOMEM (injected by harbor_evomem) -->"
MEMORY_BLOCK_CLOSE = "<!-- END EVOMEM -->"


def _inject_memory(problem: str, memory_md: str) -> str:
    body = (memory_md or "").strip()
    if not body:
        return problem
    return f"{MEMORY_BLOCK_OPEN}\n{body}\n{MEMORY_BLOCK_CLOSE}\n\n{problem}"


def _trim_text_to_budget(text: str, char_budget: int) -> str:
    if len(text) <= char_budget:
        return text
    clipped = text[:char_budget].rsplit("\n", 1)[0].rstrip()
    if not clipped:
        clipped = text[:char_budget].rstrip()
    return clipped + "\n\n...[memory truncated]"


class Terminus2EvoMem(Terminus2GitCapture):
    """Terminus2 + chain-scoped EvoMem.

    The injected context points Terminus2 at one compact family-memory file:
    a base recipe plus relevant conditional patches compiled from prior
    task-memory records. Extra git diff capture and post-run environment
    reads are opt-in so EvoMem does not perturb root-task agent behavior.
    """

    def __init__(
        self,
        chain_id: Optional[str] = None,
        host_root: Optional[str] = None,
        char_budget: int = 3500,
        terminal_family_char_budget: Optional[int] = None,
        enable_git_capture: bool = False,
        post_run_environment_capture: bool = False,
        *args: Any,
        **kwargs: Any,
    ):
        self._explicit_chain_id = chain_id
        self._host_root = host_root
        self._char_budget = int(char_budget)
        self._terminal_family_char_budget = terminal_family_char_budget
        self._post_run_environment_capture = _as_bool(post_run_environment_capture)
        self._chain_id: Optional[str] = None
        self._task_name: Optional[str] = None
        self._injected_instruction: Optional[str] = None
        self._original_instruction: Optional[str] = None
        self._family_paths: Optional[family_trace_memory.FamilyPaths] = None
        self._family_probe: dict[str, Any] = {}
        self._family_task_delta = ""
        self._rendered_terminal_family_md = ""
        self._family_workspace_files: dict[str, str] = {}
        super().__init__(*args, enable_capture=enable_git_capture, **kwargs)

    def _terminal_family_budget(self) -> int:
        total = max(512, int(self._char_budget))
        if self._terminal_family_char_budget is not None:
            return max(128, min(int(self._terminal_family_char_budget), total - 256))
        third = max(256, min(total // 3, 2400))
        return min(third, total - 256)

    async def run(  # type: ignore[override]
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        self._task_name = environment.environment_name
        self._family_paths = None
        self._family_probe = {}
        self._family_task_delta = ""
        self._rendered_terminal_family_md = ""
        self._family_workspace_files = {}

        tb = self._terminal_family_budget()

        extra_env = self._extra_env or {}
        explicit = self._explicit_chain_id or extra_env.get("HARBOR_EVOMEM_CHAIN_ID")
        host_root = self._host_root or extra_env.get("HARBOR_EVOMEM_HOST_ROOT")
        self._chain_id = derive_chain_id(self._task_name, explicit)
        self._family_paths = (
            family_trace_memory.FamilyPaths(self._chain_id, host_root) if self._chain_id else None
        )

        rendered_family = ""
        if self._family_paths is not None:
            try:
                prev_nm, prev_text = family_trace_memory.previous_release_snapshot_task(
                    self._family_paths,
                    self._task_name or "task",
                )
                if prev_nm:
                    self._family_probe = await family_trace_memory.probe_environment_for_family(
                        environment,
                        logger=self.logger,
                    )
                    delta = family_trace_memory.task_diff(prev_text, instruction)
                    self._family_task_delta = delta

                    probe_md = family_trace_memory.probe_summary_md(self._family_probe)
                    family_context = family_trace_memory.retrieve_family_context(
                        self._family_paths,
                        task_name=self._task_name or "task",
                        instruction_text=instruction,
                        probe=self._family_probe,
                    )
                    if not family_trace_memory.family_context_has_reusable_memory(
                        family_context
                    ):
                        self.logger.info(
                            "family_trace_memory: no reusable base recipe or actionable patch for task=%s; skipping injection",
                            self._task_name,
                        )
                        self._family_workspace_files = {}
                        rendered_family = ""
                        self._rendered_terminal_family_md = ""
                    else:
                        try:
                            self._family_workspace_files = await family_trace_memory.hydrate_family_workspace_files(
                                environment,
                                family_context,
                                logger=self.logger,
                            )
                            rendered_family = family_trace_memory.render_family_pointer_markdown(
                                family_context,
                            ).strip()
                        except Exception as exc:  # noqa: BLE001
                            self.logger.warning("family_trace_memory hydration failed: %s", exc)
                            self._family_workspace_files = {}
                            rendered_family = family_trace_memory.render_family_context_markdown(
                                family_context,
                                probe_md=probe_md,
                                char_budget=tb,
                            ).strip()

                    self._rendered_terminal_family_md = rendered_family
                else:
                    self._family_probe = {}
                    self._family_task_delta = ""
                    self._family_workspace_files = {}
                    rendered_family = ""
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("family_trace_memory pre-run failed: %s", exc)
                rendered_family = ""

        parts: list[str] = []
        if rendered_family:
            parts.append(rendered_family.strip())

        combined = "\n\n".join(parts)
        if len(combined) > self._char_budget:
            combined = _trim_text_to_budget(combined, self._char_budget)

        self._original_instruction = instruction
        self._injected_instruction = _inject_memory(instruction, combined)
        self._dump_pre_run_artefacts(combined)

        try:
            await super().run(
                instruction=self._injected_instruction,
                environment=environment,
                context=context,
            )
        finally:
            try:
                await self._ingest_terminal_family_memory(environment)
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("family_trace_memory ingest failed: %s", exc)

    def _dump_pre_run_artefacts(self, rendered: str) -> None:
        out = self.harbor_evomem_dir
        try:
            meta = {
                "task_name": self._task_name,
                "chain_id": self._chain_id,
                "memory_chars": len(rendered or ""),
                "char_budget": self._char_budget,
                "terminal_family_chars": len(self._rendered_terminal_family_md or ""),
                "family_workspace_files": sorted(self._family_workspace_files.keys()),
                "injection": "compact_family_memory_v1"
                if (rendered or self._rendered_terminal_family_md)
                else "none",
            }
            (out / "pre_run_meta.json").write_text(
                json.dumps(meta, indent=2), encoding="utf-8"
            )
            if self._family_probe:
                raw_probe = json.dumps(self._family_probe, indent=2, ensure_ascii=False)
                (out / "terminal_family_probe.json").write_text(
                    raw_probe[:120_000],
                    encoding="utf-8",
                )
            if self._family_task_delta:
                (out / "terminal_family_task_diff.md").write_text(
                    self._family_task_delta, encoding="utf-8"
                )
            if self._rendered_terminal_family_md:
                (out / "rendered_terminal_family.md").write_text(
                    self._rendered_terminal_family_md,
                    encoding="utf-8",
                )
            if self._family_workspace_files:
                local_dir = out / "terminal_family_workspace"
                local_dir.mkdir(parents=True, exist_ok=True)
                for path, content in self._family_workspace_files.items():
                    rel = path.split("/app/.harbor_evomem/", 1)[-1] if "/app/.harbor_evomem/" in path else path.rsplit("/", 1)[-1]
                    target = local_dir / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
            if rendered:
                (out / "rendered_memory.md").write_text(rendered, encoding="utf-8")
            if self._original_instruction is not None:
                (out / "original_instruction.md").write_text(
                    self._original_instruction, encoding="utf-8"
                )
            if self._injected_instruction is not None:
                (out / "injected_instruction.md").write_text(
                    self._injected_instruction, encoding="utf-8"
                )
        except Exception as exc:  # noqa: BLE001
            self.logger.warning("failed to dump pre-run artefacts: %s", exc)

    async def _ingest_terminal_family_memory(self, environment: BaseEnvironment) -> None:
        if self._family_paths is None or not self._task_name:
            return

        probe: dict[str, Any] = (
            self._family_probe if isinstance(self._family_probe, dict) else {}
        )
        if not probe and self._post_run_environment_capture:
            try:
                probe = await family_trace_memory.probe_environment_for_family(
                    environment,
                    logger=self.logger,
                )
            except Exception as exc:  # noqa: BLE001
                self.logger.warning("family_trace_memory probe fallback failed: %s", exc)
                probe = {}

        history = self._load_history()
        captured = self.captured
        diff_txt = getattr(captured, "diff_text", None) if captured else None
        heads = getattr(captured, "changed_files", []) if captured else []
        snapshot_candidates = family_trace_memory._task_memory_candidate_paths(  # type: ignore[attr-defined]
            self._original_instruction or "",
            list(heads or []),
            history=history,
        )
        if self._post_run_environment_capture:
            file_snapshots = await family_trace_memory.capture_file_snapshots(
                environment,
                snapshot_candidates,
                logger=self.logger,
            )
        else:
            file_snapshots = {}
        terminal_trace = family_trace_memory.build_terminal_trace(history)

        if self._post_run_environment_capture:
            reward = await family_trace_memory.try_read_verifier_reward(environment)
        else:
            reward = None

        prev_nm, prev_text = family_trace_memory.previous_release_snapshot_task(
            self._family_paths,
            self._task_name,
        )

        family_trace_memory.persist_instruction_snapshot(
            self._family_paths,
            self._task_name,
            self._original_instruction or "",
        )

        probe_md = family_trace_memory.probe_summary_md(probe)
        family_trace_memory.update_family_state(
            self._family_paths,
            self._task_name,
            probe_md,
        )

        family_trace_memory.store_family_memory(
            self._family_paths,
            task_name=self._task_name,
            probe=probe,
            prev_task_name=prev_nm,
            prev_instruction_text=prev_text,
            instruction_text=self._original_instruction or "",
            history=history,
            diff_text=diff_txt or "",
            changed_files=list(heads or []),
            validation_passed=reward,
            file_snapshots=file_snapshots,
            terminal_trace=terminal_trace,
            logger=self.logger,
        )

    def _load_history(self) -> list[dict[str, Any]]:
        candidate = Path(self.logs_dir) / "trajectory.json"
        return memory_bridge.load_terminus2_history(candidate)
