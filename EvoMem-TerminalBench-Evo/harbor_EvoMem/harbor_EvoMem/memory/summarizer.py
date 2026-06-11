"""LLM-backed EvoMem summariser.

It converts a Terminus2 trajectory plus bounded diff metadata into a structured
feature-patch record payload for later sibling tasks in the same chain.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class TrajectoryEvent:
    kind: str = ""
    summary: str = ""
    files: list[str] = field(default_factory=list)
    reasoning: str = ""


@dataclass
class TrajectorySlice:
    events: list[TrajectoryEvent] = field(default_factory=list)


@dataclass
class ValidationEpisode:
    files_changed: list[str] = field(default_factory=list)
    reasoning_excerpts: list[str] = field(default_factory=list)
    status: str = "unknown"
    tests_before: list[dict[str, Any]] = field(default_factory=list)
    tests: list[dict[str, Any]] = field(default_factory=list)


class PatchMemorySummarizer:
    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ):
        """Parameters are retained for EvoClaw API parity — values are ignored."""

        _ = (model, api_key, base_url)

        from ..memory_bridge import authoritative_llm_config, litellm_model_for_config

        cfg = authoritative_llm_config()
        self.model = cfg.model
        self.litellm_model = litellm_model_for_config(cfg)
        self.api_key = cfg.api_key
        self.base_url = cfg.base_url

    def suggest_feature_segmentation(
        self,
        milestone_id: str,
        changed_files: list[str],
        diff_text: str,
        trajectory_slice: Optional[TrajectorySlice],
        episodes: Optional[list[ValidationEpisode]] = None,
    ) -> list[dict[str, Any]]:
        if episodes:
            return self._segment_by_episodes(
                milestone_id=milestone_id,
                changed_files=changed_files,
                diff_text=diff_text,
                trajectory_slice=trajectory_slice,
                episodes=episodes,
            )
        return self._segment_by_path(
            milestone_id=milestone_id,
            changed_files=changed_files,
            diff_text=diff_text,
            trajectory_slice=trajectory_slice,
        )

    def _segment_by_episodes(
        self,
        milestone_id: str,
        changed_files: list[str],
        diff_text: str,
        trajectory_slice: Optional[TrajectorySlice],
        episodes: list[ValidationEpisode],
    ) -> list[dict[str, Any]]:
        changed_set = set(changed_files)
        output: list[dict[str, Any]] = []
        covered: set[str] = set()

        seen_filesets: dict[frozenset, int] = {}
        for ep in episodes:
            fs = frozenset(ep.files_changed)
            seen_filesets[fs] = episodes.index(ep)
        deduped = [episodes[i] for i in sorted(set(seen_filesets.values()))]

        for idx, episode in enumerate(deduped, start=1):
            ep_files = [f for f in episode.files_changed if f in changed_set]
            if not ep_files:
                continue
            covered.update(ep_files)
            title = self._title_from_paths(ep_files)
            feature_id = f"{milestone_id.lower()}-feature-{idx:02d}"
            reasoning = (
                self._distill_reasoning(episode.reasoning_excerpts)
                if episode.reasoning_excerpts
                else ""
            )
            output.append(
                {
                    "feature_id": feature_id,
                    "feature_title": title,
                    "files_changed": sorted(ep_files),
                    "diff_excerpt": self._excerpt_diff(diff_text, ep_files),
                    "trajectory_excerpt": self._excerpt_trajectory(trajectory_slice, ep_files),
                    "reasoning_excerpt": reasoning[:800],
                    "episode_status": episode.status,
                    "tests_before": [
                        {"command": t.command, "status": t.status} for t in episode.tests_before
                    ],
                    "tests_after": [
                        {"command": t.command, "status": t.status} for t in episode.tests
                    ],
                }
            )

        uncovered = [f for f in changed_files if f not in covered]
        if uncovered:
            idx = len(output) + 1
            title = self._title_from_paths(uncovered)
            output.append(
                {
                    "feature_id": f"{milestone_id.lower()}-feature-{idx:02d}",
                    "feature_title": title,
                    "files_changed": sorted(uncovered),
                    "diff_excerpt": self._excerpt_diff(diff_text, uncovered),
                    "trajectory_excerpt": self._excerpt_trajectory(trajectory_slice, uncovered),
                    "reasoning_excerpt": "",
                    "episode_status": "unknown",
                    "tests_before": [],
                    "tests_after": [],
                }
            )

        if not output:
            return self._segment_by_path(
                milestone_id=milestone_id,
                changed_files=changed_files,
                diff_text=diff_text,
                trajectory_slice=trajectory_slice,
            )

        return output

    def _segment_by_path(
        self,
        milestone_id: str,
        changed_files: list[str],
        diff_text: str,
        trajectory_slice: Optional[TrajectorySlice],
    ) -> list[dict[str, Any]]:
        grouped: dict[str, list[str]] = {}
        for path in changed_files:
            parts = [part for part in path.split("/") if part]
            group_key = "/".join(parts[:2]) if len(parts) >= 2 else path
            grouped.setdefault(group_key, []).append(path)

        output = []
        for index, (group_key, files) in enumerate(sorted(grouped.items()), start=1):
            title = group_key.split("/")[-1].replace("_", "-")
            output.append(
                {
                    "feature_id": f"{milestone_id.lower()}-feature-{index:02d}",
                    "feature_title": title,
                    "files_changed": sorted(files),
                    "diff_excerpt": self._excerpt_diff(diff_text, files),
                    "trajectory_excerpt": self._excerpt_trajectory(trajectory_slice, files),
                    "reasoning_excerpt": "",
                    "episode_status": "unknown",
                    "tests_before": [],
                    "tests_after": [],
                }
            )
        return output

    _ACTION_RE = re.compile(
        r"\b(add|fix|update|implement|change|modify|remove|replace|refactor|ensure|support|handle)\b",
        re.IGNORECASE,
    )

    def _distill_reasoning(self, excerpts: list[str], max_len: int = 300) -> str:
        if not excerpts:
            return ""
        candidates = [e.strip() for e in excerpts[:3] if e.strip()]
        if not candidates:
            return ""
        action_excerpts = [e for e in candidates if self._ACTION_RE.search(e)]
        raw = action_excerpts[0] if action_excerpts else candidates[0]
        stripped = re.sub(
            r"^(I'?ll|I will|I need to|I should|I'?m going to|I?'m|I?)?\s*",
            "",
            raw,
            flags=re.IGNORECASE,
        )
        sentence = re.split(r"(?<=[.!?])\s+", stripped)[0]
        sentence = re.sub(r"\s+(and|then|so|to)\s*$", "", sentence, flags=re.IGNORECASE).strip()
        if len(sentence) > max_len:
            sentence = sentence[:max_len].rsplit(" ", 1)[0] + "..."
        return sentence

    def _title_from_paths(self, paths: list[str]) -> str:
        if not paths:
            return "unknown"
        parts = [p for p in paths[0].split("/") if p]
        if len(parts) >= 2:
            return parts[1].replace("_", "-")
        return parts[-1].replace("_", "-") if parts else "unknown"

    def summarize_feature_patch(
        self,
        milestone_id: str,
        srs_text: Optional[str],
        feature_group: dict[str, Any],
        trajectory_slice: Optional[TrajectorySlice],
        related_prior_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        prompt = self._build_summary_prompt(
            milestone_id=milestone_id,
            srs_text=srs_text,
            feature_group=feature_group,
            trajectory_slice=trajectory_slice,
            related_prior_records=related_prior_records,
        )
        try:
            response = self._call_llm_json(prompt, schema_name="feature_patch_summary")
            if isinstance(response, dict):
                return response
        except Exception:
            pass
        return self._fallback_summary(milestone_id, srs_text, feature_group, trajectory_slice)

    def _build_summary_prompt(
        self,
        milestone_id: str,
        srs_text: Optional[str],
        feature_group: dict[str, Any],
        trajectory_slice: Optional[TrajectorySlice],
        related_prior_records: list[dict[str, Any]],
    ) -> str:
        srs_excerpt = (srs_text or "")[:3000]
        trajectory_excerpt = feature_group.get("trajectory_excerpt") or self._excerpt_trajectory(
            trajectory_slice, feature_group.get("files_changed", [])
        )
        reasoning_excerpt = feature_group.get("reasoning_excerpt", "")
        related_excerpt = json.dumps(related_prior_records[:3], indent=2)

        tests_before = feature_group.get("tests_before", [])
        tests_after = feature_group.get("tests_after", [])
        validation_block = ""
        if tests_before or tests_after:
            validation_block = f"\nTests before this change: {json.dumps(tests_before)}"
            validation_block += f"\nTests after this change: {json.dumps(tests_after)}"

        return f"""
You are summarizing one feature-level code change for an evolving software benchmark memory.
Use the agent reasoning and trajectory as the primary source of intent. Use the diff and SRS as supporting evidence.

The "why_changed" field must explain the specific technical reason for the change — what was broken/missing
and how the change addresses it. Do NOT say "to satisfy milestone requirements" — instead describe
the concrete behavior gap this change closes (e.g. "RFE lacked set_params_routing because
_parameter_constraints did not include the metadata routing mixin").

Return JSON only with keys:
feature_title, what_changed, why_changed, before_behavior, after_behavior,
constraints_to_preserve, known_risks, feature_tags, symbols_changed.

Milestone: {milestone_id}

SRS excerpt:
{srs_excerpt}

Files changed:
{json.dumps(feature_group.get("files_changed", []), indent=2)}

Diff excerpt:
{feature_group.get("diff_excerpt", "")[:5000]}

Agent reasoning (why the agent made this change):
{reasoning_excerpt[:3000]}

Trajectory excerpt:
{trajectory_excerpt[:3000]}
{validation_block}

Related prior records:
{related_excerpt[:2000]}
""".strip()

    # LiteLLM-style ``provider/model`` prefixes should be preserved for
    # OpenRouter/LiteLLM-compatible endpoints, but stripped for OpenAI's
    # native endpoint.
    _OPENAI_COMPAT_PREFIXES = ("openai/",)

    @classmethod
    def _model_for_endpoint(cls, model: str, base_url: Optional[str]) -> str:
        base = str(base_url or "").lower()
        if "openrouter.ai" in base:
            return model
        for prefix in cls._OPENAI_COMPAT_PREFIXES:
            if model.startswith(prefix) and "api.openai.com" in base:
                return model[len(prefix):]
        return model

    @staticmethod
    def _resolve_chat_completions_url(base_url: Optional[str]) -> str:
        """Resolve the OpenAI-compatible chat-completions endpoint from ``LLM_BASE_URL``."""

        if not base_url or not str(base_url).strip():
            from ..memory_bridge import MissingLLMConfigError

            raise MissingLLMConfigError(
                "EvoMem summariser requires LLM_BASE_URL (set in scripts/terminus2_llm.env)."
            )
        url = str(base_url).rstrip("/")
        # Accept both "<host>" and "<host>/v1" as valid base_url values.
        if url.endswith("/chat/completions"):
            return url
        if not url.endswith("/v1"):
            url = url + "/v1"
        return url + "/chat/completions"

    def _call_llm_json(self, prompt: str, schema_name: str) -> dict[str, Any]:
        if not self.api_key:
            raise RuntimeError("LLM API key is unavailable")

        try:
            import litellm
        except ImportError as exc:  # pragma: no cover - Harbor runtime has LiteLLM
            raise RuntimeError(
                "EvoMem summariser requires LiteLLM so it can use the same "
                "request path as the Terminus2 agent."
            ) from exc

        response = litellm.completion(
            model=self.litellm_model,
            messages=[
                {"role": "system", "content": "Return strict JSON only."},
                {"role": "user", "content": prompt},
            ],
            api_key=self.api_key,
            api_base=self.base_url,
            drop_params=True,
            timeout=60,
            temperature=0.0,
            response_format={"type": "json_object"},
        )

        message = self._extract_message_content(response)
        if message is None:
            raise RuntimeError(
                f"LLM response for {schema_name} did not contain message content: "
                f"{str(response)[:500]}"
            )
        if isinstance(message, dict):
            return message
        if not isinstance(message, str):
            raise RuntimeError(
                f"LLM response for {schema_name} had non-text message content: "
                f"{type(message).__name__}"
            )
        return json.loads(message)

    @staticmethod
    def _extract_message_content(response: Any) -> Any:
        def get(obj: Any, key: str) -> Any:
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        choices = get(response, "choices")
        if not choices:
            return None
        first = choices[0]
        message = get(first, "message")
        if message is None:
            return None
        return get(message, "content")

    def _fallback_summary(
        self,
        milestone_id: str,
        srs_text: Optional[str],
        feature_group: dict[str, Any],
        trajectory_slice: Optional[TrajectorySlice],
    ) -> dict[str, Any]:
        files_changed = feature_group.get("files_changed", [])
        feature_title = (
            feature_group.get("feature_title") or feature_group.get("feature_id") or milestone_id
        )
        srs_summary = self._srs_summary(srs_text)

        reasoning_excerpt = feature_group.get("reasoning_excerpt", "")
        tests_before = feature_group.get("tests_before", [])
        tests_after = feature_group.get("tests_after", [])

        why_parts: list[str] = []
        if reasoning_excerpt:
            distilled = self._distill_reasoning([reasoning_excerpt])
            if distilled:
                why_parts.append(distilled)
        if tests_before and not reasoning_excerpt:
            failing_before = [t for t in tests_before if t.get("status") == "failed"]
            if failing_before:
                cmd = failing_before[0].get("command", "")
                why_parts.append(f"Fixing failing test: {cmd}")
        if not why_parts and srs_summary:
            why_parts.append(srs_summary)
        if not why_parts:
            why_parts.append(f"Code change to implement {feature_title} for {milestone_id}")

        if tests_before:
            before_status = tests_before[-1].get("status", "unknown") if tests_before else "unknown"
            before_behavior = f"Tests were {before_status} before this change."
        else:
            before_behavior = f"Baseline state before implementing {feature_title}."

        if tests_after:
            after_status = tests_after[-1].get("status", "unknown") if tests_after else "unknown"
            after_cmd = tests_after[-1].get("command", "") if tests_after else ""
            after_behavior = f"After change: {after_cmd} {after_status}."
        else:
            after_behavior = f"After change: {feature_title} updated in {', '.join(files_changed[:3])}."

        known_risks: list[str] = []
        traj_commands = trajectory_slice.commands[-3:] if trajectory_slice else []
        if traj_commands:
            known_risks.append(f"Recent test command: {traj_commands[-1]}")

        symbols_changed = [path.split("/")[-1] for path in files_changed[:5]]
        return {
            "feature_title": feature_title,
            "what_changed": (
                f"Updated {', '.join(files_changed[:5])}"
                if files_changed
                else f"Updated code for {milestone_id}"
            ),
            "why_changed": " ".join(why_parts),
            "before_behavior": before_behavior,
            "after_behavior": after_behavior,
            "constraints_to_preserve": [
                "Do not regress previously passing behavior in overlapping files."
            ],
            "known_risks": known_risks,
            "feature_tags": self._feature_tags(feature_title, files_changed, srs_summary),
            "symbols_changed": symbols_changed,
        }

    def _srs_summary(self, srs_text: Optional[str]) -> str:
        if not srs_text:
            return ""
        for line in srs_text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("# ").strip()
            if stripped:
                return stripped[:200]
        return ""

    def _feature_tags(
        self, feature_title: str, files_changed: list[str], srs_summary: str
    ) -> list[str]:
        tags: list[str] = []
        for value in [feature_title, srs_summary, *files_changed]:
            normalized = value.replace("/", " ").replace("_", " ").replace("-", " ")
            for token in normalized.split():
                token = token.lower().strip()
                if len(token) >= 3 and token not in tags:
                    tags.append(token)
        return tags[:12]

    def _excerpt_diff(self, diff_text: str, files: list[str]) -> str:
        if not diff_text:
            return ""
        if not files:
            return diff_text[:5000]
        lines = diff_text.splitlines()
        collected: list[str] = []
        keep = False
        for line in lines:
            if line.startswith("diff --git "):
                keep = any(path in line for path in files)
            if keep:
                collected.append(line)
        return "\n".join(collected)[:5000]

    def _excerpt_trajectory(
        self, trajectory_slice: Optional[TrajectorySlice], files: list[str]
    ) -> str:
        if not trajectory_slice:
            return ""
        rendered: list[str] = []
        file_set = set(files)
        for event in trajectory_slice.events:
            if file_set and not file_set.intersection(event.files):
                continue
            line = f"[{event.kind}] {event.summary}"
            if event.reasoning:
                line += f" | reasoning: {event.reasoning[:200]}"
            rendered.append(line)
        if not rendered:
            rendered = []
            for event in trajectory_slice.events[-10:]:
                line = f"[{event.kind}] {event.summary}"
                if event.reasoning:
                    line += f" | reasoning: {event.reasoning[:200]}"
                rendered.append(line)
        return "\n".join(rendered)
