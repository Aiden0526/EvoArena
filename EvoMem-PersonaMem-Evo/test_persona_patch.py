"""
Evaluate patch-augmented A-MEM on the Persona-release benchmark.

This script mirrors `test_persona_robust.py` so Persona-release results stay in
the same CSV-first workflow, but it swaps the underlying memory system to the
patch-augmented variant used by `test_advanced_patch.py`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import pickle
import subprocess
import sys
import time
from collections import OrderedDict
import numpy as np
from sentence_transformers import SentenceTransformer
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from llm_text_parsers import parse_keywords_response
from memory_layer_patch import PatchAugmentedMemorySystem, PatchConfig
from memory_layer_robust import (
    LLMUsageRecorder,
    RobustLLMController,
    is_nonrecoverable_llm_error,
    llm_usage_recorder,
    llm_usage_stage,
)
from patch_prompts import PATCH_DETAIL_REVISION_PROMPT, PATCH_GATING_PROMPT
from test_persona_robust import (
    check_mcq_correctness,
    create_mcq_options,
    extract_final_answer,
    load_benchmark_rows,
    load_existing_output_rows,
    load_chat_history_messages,
    make_progress,
    normalize_openai_env,
    parse_incorrect_answers,
    parse_persona_ids,
    parse_user_query,
    resolve_chat_history_path,
    resolve_persona_root,
    sanitize_filename_part,
    stable_int_seed,
    serialize_chat_message,
    setup_logger,
    write_metrics_file,
    write_usage_summary_file,
)


logger = logging.getLogger("persona_patch")
csv.field_size_limit(sys.maxsize)


def build_patch_sample_id(chat_history_path: Path, include_system_messages: bool, persona_id: str = "") -> str:
    payload = (
        f"{chat_history_path.resolve()}|"
        f"include_system={include_system_messages}|"
        f"persona_id={persona_id}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def sanitize_gt_text(value: Any) -> str:
    import re

    return re.sub(r"\s+", " ", str(value or "")).strip()


class GoldPatchRetriever:
    def __init__(self, patch_file: Path, model_name: str = "all-MiniLM-L6-v2") -> None:
        self.patch_file = Path(patch_file)
        self.model = SentenceTransformer(model_name)
        self.by_sample: Dict[str, List[Dict[str, Any]]] = {}
        self.by_row_key: Dict[Tuple[str, str, str, str, str, str], Dict[str, Any]] = {}
        self.by_change_key: Dict[Tuple[str, str, str, str, str], Dict[str, Any]] = {}
        self.embedding_cache: Dict[str, np.ndarray] = {}
        with self.patch_file.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                keys = {str(record.get("sample_id", ""))}
                for value in (record.get("sample_ids") or {}).values():
                    if value:
                        keys.add(str(value))
                for key in keys:
                    if key:
                        self.by_sample.setdefault(key, []).append(record)
                row_key = self._row_key_from_values(
                    record.get("persona_id"),
                    record.get("change_family"),
                    record.get("change_step"),
                    record.get("prev_pref"),
                    record.get("preference"),
                    record.get("related_conversation_snippet_text"),
                )
                change_key = self._change_key_from_values(
                    record.get("persona_id"),
                    record.get("change_family"),
                    record.get("change_step"),
                    record.get("prev_pref"),
                    record.get("preference"),
                )
                self.by_row_key[row_key] = record
                self.by_change_key[change_key] = record

    @staticmethod
    def _normalize_step(value: Any) -> str:
        if value is None or value == "":
            return ""
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return sanitize_gt_text(value)

    @classmethod
    def _row_key_from_values(
        cls,
        persona_id: Any,
        change_family: Any,
        change_step: Any,
        prev_pref: Any,
        preference: Any,
        related_snippet: Any,
    ) -> Tuple[str, str, str, str, str, str]:
        return (
            sanitize_gt_text(persona_id),
            sanitize_gt_text(change_family),
            cls._normalize_step(change_step),
            sanitize_gt_text(prev_pref),
            sanitize_gt_text(preference),
            sanitize_gt_text(related_snippet),
        )

    @classmethod
    def _change_key_from_values(
        cls,
        persona_id: Any,
        change_family: Any,
        change_step: Any,
        prev_pref: Any,
        preference: Any,
    ) -> Tuple[str, str, str, str, str]:
        return (
            sanitize_gt_text(persona_id),
            sanitize_gt_text(change_family),
            cls._normalize_step(change_step),
            sanitize_gt_text(prev_pref),
            sanitize_gt_text(preference),
        )

    def lookup_row_patch(self, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        row_key = self._row_key_from_values(
            row.get("persona_id"),
            row.get("change_family"),
            row.get("change_step"),
            row.get("prev_pref"),
            row.get("preference"),
            row.get("related_conversation_snippet"),
        )
        patch = self.by_row_key.get(row_key)
        if patch is not None:
            record = dict(patch)
            record["retrieval_similarity"] = 1.0
            record["retrieval_mode"] = "oracle_exact_row"
            return record
        change_key = self._change_key_from_values(
            row.get("persona_id"),
            row.get("change_family"),
            row.get("change_step"),
            row.get("prev_pref"),
            row.get("preference"),
        )
        patch = self.by_change_key.get(change_key)
        if patch is not None:
            record = dict(patch)
            record["retrieval_similarity"] = 1.0
            record["retrieval_mode"] = "oracle_change_key"
            return record
        return None

    def _ensure_embeddings(self, sample_id: str) -> tuple[List[Dict[str, Any]], np.ndarray]:
        records = self.by_sample.get(sample_id, [])
        if sample_id not in self.embedding_cache:
            docs = [sanitize_gt_text(record.get("default_index_document", "")) for record in records]
            self.embedding_cache[sample_id] = (
                self.model.encode(docs, normalize_embeddings=True, show_progress_bar=False)
                if docs else np.zeros((0, 384), dtype=float)
            )
        return records, self.embedding_cache[sample_id]

    def retrieve(self, sample_id: str, query: str, top_k: int, min_similarity: float = 0.0) -> List[Dict[str, Any]]:
        records, doc_embs = self._ensure_embeddings(sample_id)
        if not records:
            return []
        query_text = sanitize_gt_text(query)
        if not query_text:
            return []
        query_emb = self.model.encode([query_text], normalize_embeddings=True, show_progress_bar=False)[0]
        sims = doc_embs @ query_emb
        ranked = np.argsort(sims)[::-1]
        results: List[Dict[str, Any]] = []
        for idx in ranked:
            sim = float(sims[int(idx)])
            if sim < min_similarity:
                continue
            record = dict(records[int(idx)])
            record["retrieval_similarity"] = sim
            record["retrieval_mode"] = "similarity"
            results.append(record)
            if len(results) >= top_k:
                break
        return results


def format_gt_patch_for_context(patch: Dict[str, Any], max_snippet_chars: int = 700) -> str:
    overall = patch.get("gold_patch_overall", {})
    temporal = patch.get("temporal_span", {})
    temporal_note = overall.get("temporal_order_note") or temporal.get("temporal_order_note") or ""
    snippet_text = sanitize_gt_text(patch.get("related_conversation_snippet_text", ""))
    if len(snippet_text) > max_snippet_chars:
        snippet_text = snippet_text[:max_snippet_chars].rstrip() + " ..."
    return (
        f"GroundTruthPatch {patch.get('patch_id', '')}\n"
        f"Benchmark split: {patch.get('benchmark_split', '')}\n"
        f"Change family: {patch.get('change_family', '')}\n"
        f"Session {temporal.get('session_id', 0)} TurnSpan {temporal.get('snippet_start_turn_index')}"
        f"-{temporal.get('snippet_end_turn_index')} TriggerTurn {temporal.get('trigger_turn_index')}"
        f" (TurnNumber {temporal.get('trigger_turn_number')})\n"
        f"Temporal interpretation: {temporal_note}\n"
        f"Earlier preference: {patch.get('prev_pref', '')}\n"
        f"Later preference: {patch.get('preference', '')}\n"
        f"Trigger: {patch.get('trigger_text', '')}\n"
        f"Evidence snippet: {snippet_text}"
    )


def build_gt_patch_context(current_context: str, gt_patches: List[Dict[str, Any]]) -> str:
    if not gt_patches:
        return current_context
    patch_blocks = "\n\n".join(format_gt_patch_for_context(patch) for patch in gt_patches)
    return (
        f"{current_context}\n\n"
        f"[Retrieved Ground-Truth Preference Change Patches]\n{patch_blocks}\n\n"
        "Interpretation rule: treat larger turn indices as more recent, but do not confuse conversation recency with the preference's own condition or time scope. "
        "If a preference says things like when dining alone, on weekends, in the morning, during spring, on weekdays, or when at home, those phrases are part of the preference itself and must be matched against the question scenario. "
        "Do not assume the later preference is automatically the answer; decide whether the earlier preference, the later preference, or the transition itself best fits the question's condition, timeframe, and wording."
    )


def shard_rows_by_chat_history(
    rows: List[Dict[str, str]],
    size: str,
    persona_root: Path,
    num_workers: int,
    worker_id: int,
) -> List[Dict[str, str]]:
    if num_workers <= 1:
        return rows

    buckets: Dict[str, int] = {}
    sharded: List[Dict[str, str]] = []
    for row in rows:
        chat_history_path = resolve_chat_history_path(row, size=size, persona_root=persona_root)
        key = str(chat_history_path)
        assigned_worker = buckets.setdefault(key, len(buckets) % num_workers)
        if assigned_worker == worker_id:
            sharded.append(row)
    return sharded


def build_output_path(
    output_arg: Optional[str],
    persona_root: Path,
    benchmark_file: Path,
    model: str,
    size: str,
    patch_usage: str,
    worker_id: int,
    num_workers: int,
) -> Path:
    if output_arg:
        output_path = Path(output_arg)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
        return output_path

    results_dir = persona_root / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%m%d%Y_%H%M%S")
    worker_suffix = f".worker_{worker_id}" if num_workers > 1 else ""
    filename = (
        f"amem_patch_{patch_usage}_{size}_bench-{sanitize_filename_part(benchmark_file.stem)}"
        f"_model-{sanitize_filename_part(model)}{worker_suffix}_{timestamp}.csv"
    )
    return results_dir / filename


class PersonaPatchAgent:
    def __init__(
        self,
        model: str,
        backend: str,
        retrieve_k: int,
        patch_top_k: int,
        patch_usage: str,
        answer_temperature: float,
        cache_root: Path,
        current_memory_cache_root: Optional[Path] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        api_base: Optional[str] = None,
        exclude_revoke_patches: bool = False,
        exclude_add_patches: bool = False,
        min_patch_similarity: float = 0.0,
        force_reingest_patches: bool = False,
        preference_aware_level: str = "none",
        require_pref_change: bool = False,
        llm_patch_filter: bool = False,
        gt_patch_retriever: Optional[GoldPatchRetriever] = None,
        gt_patch_top_k: Optional[int] = None,
        gt_patch_min_similarity: Optional[float] = None,
        gp_patch_retrieval: str = "similarity",
    ):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.patch_top_k = patch_top_k
        self.patch_usage = patch_usage
        self.answer_temperature = answer_temperature
        self.cache_root = cache_root
        self.current_memory_cache_root = current_memory_cache_root
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.api_base = api_base
        self.exclude_revoke_patches = exclude_revoke_patches
        self.exclude_add_patches = exclude_add_patches
        self.min_patch_similarity = min_patch_similarity
        self.force_reingest_patches = force_reingest_patches
        self.preference_aware_level = preference_aware_level
        self.require_pref_change = require_pref_change
        self.llm_patch_filter = llm_patch_filter
        self.gt_patch_retriever = gt_patch_retriever
        self.gt_patch_top_k = gt_patch_top_k or patch_top_k
        self.gt_patch_min_similarity = min_patch_similarity if gt_patch_min_similarity is None else gt_patch_min_similarity
        self.gp_patch_retrieval = gp_patch_retrieval
        self.helper_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.memory_system: Optional[PatchAugmentedMemorySystem] = None
        self.sample_id: Optional[str] = None
        self.current_row: Optional[Dict[str, Any]] = None

    def _load_current_memory_override(self, sample_id: str) -> None:
        if self.current_memory_cache_root is None:
            return
        assert self.memory_system is not None
        memory_cache = self.current_memory_cache_root / f"memory_cache_{sample_id}.pkl"
        retriever_cache = self.current_memory_cache_root / f"retriever_cache_{sample_id}.pkl"
        retriever_embeddings = self.current_memory_cache_root / f"retriever_cache_embeddings_{sample_id}.npy"
        missing = [
            str(path)
            for path in (memory_cache, retriever_cache, retriever_embeddings)
            if not path.exists()
        ]
        if missing:
            raise FileNotFoundError(
                "Cannot load current-memory override; missing robust cache artifacts: "
                + ", ".join(missing)
            )
        with memory_cache.open("rb") as f:
            self.memory_system.base_system.memories = pickle.load(f)
        self.memory_system.base_system.retriever = self.memory_system.base_system.retriever.load(
            str(retriever_cache),
            str(retriever_embeddings),
        )

    def set_sample(self, sample_id: str) -> None:
        if self.sample_id == sample_id and self.memory_system is not None:
            return
        self.sample_id = sample_id
        self.memory_system = PatchAugmentedMemorySystem(
            sample_id=sample_id,
            model_name="all-MiniLM-L6-v2",
            llm_backend=self.backend,
            llm_model=self.model,
            sglang_host=self.sglang_host,
            sglang_port=self.sglang_port,
            api_base=self.api_base,
            store_root=str(self.cache_root),
            config=PatchConfig(
                patch_top_k=self.patch_top_k,
                retrieve_k_current=self.retrieve_k,
                patch_usage=self.patch_usage,
                exclude_revoke_patches=self.exclude_revoke_patches,
                exclude_add_patches=self.exclude_add_patches,
                min_patch_similarity=self.min_patch_similarity,
                force_reingest_patches=self.force_reingest_patches,
                preference_aware_level=self.preference_aware_level,
                require_preference_change=self.require_pref_change,
                llm_patch_filter=self.llm_patch_filter,
            ),
        )
        self._load_current_memory_override(sample_id)

    def set_current_row(self, row: Dict[str, Any]) -> None:
        self.current_row = row

    def has_cached_state(self) -> bool:
        return bool(self.memory_system and self.memory_system.has_complete_global_graph_cache())

    def add_memory(self, content: str, message_index: int, role: str) -> None:
        assert self.memory_system is not None
        self.memory_system.ingest_turn_with_patch_history(
            content,
            time=f"message_{message_index:04d}",
            session_id=0,
            session_date_time=f"message_{message_index:04d}",
            session_summary="",
            turn_position=message_index,
            turn_number=message_index + 1,
            dia_id=f"message_{message_index:04d}",
            speaker=role,
        )

    def mark_complete(self) -> None:
        assert self.memory_system is not None
        self.memory_system.mark_sample_complete()

    def generate_query_keywords(self, question: str, option_mapping: Optional[Dict[str, str]] = None) -> str:
        if option_mapping:
            option_text = "\n".join(f"{key}. {value}" for key, value in option_mapping.items())
            prompt = f"""You are generating retrieval keywords for a memory system.

Question:
{question}

Answer options:
{option_text}

Return a compact comma-separated keyword list focused on preference signals, constraints, tastes, habits, changes over time, and any distinctive attributes needed to answer the question."""
        else:
            prompt = f"""You are generating retrieval keywords for a memory system.

Question:
{question}

Return a compact comma-separated keyword list focused on preference signals, constraints, tastes, habits, changes over time, and any distinctive attributes needed to answer the question."""
        response = self.helper_llm.llm.get_completion(prompt, temperature=0.0)
        return parse_keywords_response(response)

    def _retrieve_context_bundle(
        self,
        question: str,
        option_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        assert self.memory_system is not None
        keywords = self.generate_query_keywords(question, option_mapping)
        raw_context = self.memory_system.retrieve_current_context(keywords, self.retrieve_k)
        current_context = raw_context
        patches = self.memory_system.retrieve_relevant_patches(keywords, self.patch_top_k)
        gt_patches: List[Dict[str, Any]] = []
        gt_patch_context = ""
        if self.gt_patch_retriever is not None and self.sample_id is not None:
            if self.gp_patch_retrieval == "oracle" and self.current_row is not None:
                oracle_patch = self.gt_patch_retriever.lookup_row_patch(self.current_row)
                gt_patches = [oracle_patch] if oracle_patch is not None else []
            else:
                gt_patches = self.gt_patch_retriever.retrieve(
                    self.sample_id,
                    keywords,
                    self.gt_patch_top_k,
                    min_similarity=self.gt_patch_min_similarity,
                )
            gt_patch_context = build_gt_patch_context("", gt_patches).strip()
            if gt_patches:
                current_context = build_gt_patch_context(current_context, gt_patches)
        return {
            "keywords": keywords,
            "raw_context": raw_context,
            "narrowed_context": "",
            "current_context": current_context,
            "patches": patches,
            "gt_patches": gt_patches,
            "gt_patch_context": gt_patch_context,
        }

    def _answer_with_patch_policy(
        self,
        question: str,
        answer_instruction: str,
        base_prompt_builder,
        option_mapping: Optional[Dict[str, str]] = None,
    ) -> Dict[str, object]:
        assert self.memory_system is not None
        bundle = self._retrieve_context_bundle(question, option_mapping)
        current_context = str(bundle["current_context"])
        patches = list(bundle["patches"])

        if self.patch_usage == "gated":
            patch_summaries = self.memory_system._build_patch_summary_context(patches)
            gating_prompt = PATCH_GATING_PROMPT.format(
                question=question,
                answer_instruction=answer_instruction,
                current_context=current_context,
                patch_summaries=patch_summaries,
            )
            gating_response = self.memory_system.base_system.llm_controller.llm.get_completion(
                gating_prompt,
                temperature=self.answer_temperature,
                max_tokens=4096,
            )
            gating_result = self.memory_system._parse_patch_gating_response(gating_response)
            selected_patches = self.memory_system._filter_selected_patches(
                gating_result["selected_patch_ids"],
                patches,
            )
            if not gating_result["need_patch_detail"] or not selected_patches:
                return {
                    "response": gating_result["draft_answer"],
                    "keywords": bundle["keywords"],
                    "raw_context": bundle["raw_context"],
                    "narrowed_context": bundle["narrowed_context"],
                    "context": current_context,
                    "prompt": gating_prompt,
                    "answer_metadata": {
                        "patch_usage": "gated",
                        "need_patch_detail": bool(gating_result["need_patch_detail"]),
                        "used_patch_detail": False,
                        "selected_patch_ids": [patch.get("patch_id") for patch in selected_patches],
                        "selected_gt_patch_ids": [patch.get("patch_id") for patch in bundle.get("gt_patches", [])],
                        "gating_reason": gating_result["reason"],
                    },
                }

            detail_context = self.memory_system.build_augmented_context(current_context, selected_patches)
            detail_prompt = PATCH_DETAIL_REVISION_PROMPT.format(
                question=question,
                answer_instruction=answer_instruction,
                current_context=current_context,
                patch_details="\n\n".join(
                    self.memory_system.format_patch_for_context(patch) for patch in selected_patches
                ),
                draft_answer=gating_result["draft_answer"],
            )
            detail_response = self.memory_system.base_system.llm_controller.llm.get_completion(
                detail_prompt,
                temperature=self.answer_temperature,
                max_tokens=4096,
            )
            detail_result = self.memory_system._parse_patch_revision_response(detail_response)
            return {
                "response": detail_result["final_answer"],
                "keywords": bundle["keywords"],
                "raw_context": bundle["raw_context"],
                "narrowed_context": bundle["narrowed_context"],
                "context": detail_context,
                "prompt": detail_prompt,
                "answer_metadata": {
                    "patch_usage": "gated",
                    "need_patch_detail": True,
                    "used_patch_detail": True,
                    "selected_patch_ids": [patch.get("patch_id") for patch in selected_patches],
                    "selected_gt_patch_ids": [patch.get("patch_id") for patch in bundle.get("gt_patches", [])],
                    "gating_reason": gating_result["reason"],
                    "revision_reason": detail_result["reason"],
                },
            }

        context = self.memory_system.build_augmented_context(current_context, patches)
        prompt = base_prompt_builder(context)
        response = self.memory_system.base_system.llm_controller.llm.get_completion(
            prompt,
            temperature=self.answer_temperature,
            max_tokens=4096,
        )
        return {
            "response": response,
            "keywords": bundle["keywords"],
            "raw_context": bundle["raw_context"],
            "narrowed_context": bundle["narrowed_context"],
            "context": context,
            "prompt": prompt,
            "answer_metadata": {
                "patch_usage": "always",
                "need_patch_detail": bool(patches),
                "used_patch_detail": bool(patches),
                "selected_patch_ids": [patch.get("patch_id") for patch in patches],
                "selected_gt_patch_ids": [patch.get("patch_id") for patch in bundle.get("gt_patches", [])],
                "gp_patch_retrieval": self.gp_patch_retrieval,
                "selected_gt_patch_modes": [patch.get("retrieval_mode") for patch in bundle.get("gt_patches", [])],
                "gating_reason": "always_on_patch_context",
            },
        }

    def answer_mcq(self, question: str, option_mapping: Dict[str, str]) -> Dict[str, object]:
        options_text = "\n".join(f"{key}. {value}" for key, value in option_mapping.items())
        answer_instruction = (
            "Choose the single best option using the retrieved conversation evidence. "
            "End with exactly one line in this format: Final Answer: <LETTER>.\n\n"
            f"Options:\n{options_text}"
        )

        def build_prompt(context: str) -> str:
            return f"""You are answering a PersonaMem benchmark question using retrieved conversation memories.

Use only the retrieved memories below as evidence. Focus on the user's demonstrated preferences, priorities, and changes over time. Avoid generic common-sense guessing.

Reasoning rules:
1. If a ground-truth patch or historical patch shows an earlier preference and a later preference, do not assume the later preference automatically wins.
2. First identify the operative part of the question: is it asking about a general preference, a condition-specific preference, or a time-specific preference?
3. Treat condition and time phrases inside the preference itself as critical evidence. Phrases like when dining alone, on weekends, in the morning, during spring, on weekdays, or when at home define when that preference applies.
4. Keep two notions of time separate: conversation recency tells you which evidence is newer, while temporal phrases inside the preference tell you when the preference applies. A newer preference can still be about a narrower scenario such as weekends, mornings, or cold weather.
5. Prefer the option that best matches the retrieved evidence, especially the preference information in patches. Check whether the earlier preference or the later preference better matches the question's scenario, condition, or timeframe, then choose the option that fits that state.

Retrieved memories:
{context}

Question:
{question}

Options:
{options_text}

Before choosing, briefly determine:
- what condition or time scope the question is asking about
- whether that condition/time scope matches the earlier preference, the later preference, or neither
- which option best expresses that matched state

Then end with exactly one line in this format:
Final Answer: <LETTER>"""

        return self._answer_with_patch_policy(
            question,
            answer_instruction,
            build_prompt,
            option_mapping=option_mapping,
        )

    def answer_openended(self, question: str) -> Dict[str, object]:
        answer_instruction = (
            "Answer the question directly in natural language. Keep the answer grounded in the evidence and "
            "avoid unsupported details."
        )

        def build_prompt(context: str) -> str:
            return f"""You are answering a PersonaMem benchmark question using retrieved conversation memories.

Use only the retrieved memories below as evidence. Focus on the user's demonstrated preferences, priorities, and changes over time. Avoid generic common-sense guessing.

Reasoning rules:
1. If a ground-truth patch or historical patch shows an earlier preference and a later preference, do not assume the later preference automatically wins.
2. First determine what the question is asking about: the user's current preference, an earlier preference, a condition-specific preference, or a time-specific preference.
3. Use temporal recency as evidence, but only after checking whether the question's condition or timeframe points to the earlier state, the later state, or a conditional split between them.
4. Prefer the option that best matches the retrieved evidence, even if another option sounds more generally reasonable.

Retrieved memories:
{context}

Question:
{question}

Answer the question directly in natural language. Keep the answer grounded in the retrieved memories and avoid mentioning unsupported details."""

        return self._answer_with_patch_policy(question, answer_instruction, build_prompt)


class AgentManager:
    def __init__(
        self,
        model: str,
        backend: str,
        retrieve_k: int,
        patch_top_k: int,
        patch_usage: str,
        answer_temperature: float,
        cache_dir: Path,
        current_memory_cache_root: Optional[Path],
        include_system_messages: bool,
        max_live_agents: int,
        sglang_host: str,
        sglang_port: int,
        api_base: Optional[str],
        eval_logger: logging.Logger,
        exclude_revoke_patches: bool = False,
        exclude_add_patches: bool = False,
        min_patch_similarity: float = 0.0,
        force_reingest_patches: bool = False,
        preference_aware_level: str = "none",
        require_pref_change: bool = False,
        llm_patch_filter: bool = False,
        gt_patch_file: Optional[Path] = None,
        gt_patch_top_k: Optional[int] = None,
        gt_patch_min_similarity: Optional[float] = None,
        gp_patch_retrieval: str = "similarity",
    ):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.patch_top_k = patch_top_k
        self.patch_usage = patch_usage
        self.answer_temperature = answer_temperature
        self.cache_dir = cache_dir
        self.current_memory_cache_root = current_memory_cache_root
        self.include_system_messages = include_system_messages
        self.max_live_agents = max_live_agents
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.api_base = api_base
        self.eval_logger = eval_logger
        self.exclude_revoke_patches = exclude_revoke_patches
        self.exclude_add_patches = exclude_add_patches
        self.min_patch_similarity = min_patch_similarity
        self.force_reingest_patches = force_reingest_patches
        self.preference_aware_level = preference_aware_level
        self.require_pref_change = require_pref_change
        self.llm_patch_filter = llm_patch_filter
        self.gt_patch_file = gt_patch_file
        self.gt_patch_top_k = gt_patch_top_k or patch_top_k
        self.gt_patch_min_similarity = min_patch_similarity if gt_patch_min_similarity is None else gt_patch_min_similarity
        self.gt_patch_retriever = GoldPatchRetriever(gt_patch_file) if gt_patch_file else None
        self.gp_patch_retrieval = gp_patch_retrieval
        self.agent_cache: "OrderedDict[str, PersonaPatchAgent]" = OrderedDict()

    def _new_agent(self) -> PersonaPatchAgent:
        return PersonaPatchAgent(
            model=self.model,
            backend=self.backend,
            retrieve_k=self.retrieve_k,
            patch_top_k=self.patch_top_k,
            patch_usage=self.patch_usage,
            answer_temperature=self.answer_temperature,
            cache_root=self.cache_dir,
            current_memory_cache_root=self.current_memory_cache_root,
            sglang_host=self.sglang_host,
            sglang_port=self.sglang_port,
            api_base=self.api_base,
            exclude_revoke_patches=self.exclude_revoke_patches,
            exclude_add_patches=self.exclude_add_patches,
            min_patch_similarity=self.min_patch_similarity,
            force_reingest_patches=self.force_reingest_patches,
            preference_aware_level=self.preference_aware_level,
            require_pref_change=self.require_pref_change,
            llm_patch_filter=self.llm_patch_filter,
            gt_patch_retriever=self.gt_patch_retriever,
            gt_patch_top_k=self.gt_patch_top_k,
            gt_patch_min_similarity=self.gt_patch_min_similarity,
            gp_patch_retrieval=self.gp_patch_retrieval,
        )

    def get_agent(self, chat_history_path: Path, persona_id: str) -> PersonaPatchAgent:
        sample_id = build_patch_sample_id(chat_history_path, self.include_system_messages, persona_id)
        if sample_id in self.agent_cache:
            agent = self.agent_cache.pop(sample_id)
            self.agent_cache[sample_id] = agent
            return agent

        agent = self._new_agent()
        agent.set_sample(sample_id)

        if self.current_memory_cache_root is not None:
            assert agent.memory_system is not None
            patch_records = agent.memory_system.store.load_patch_index_records(sample_id)
            if not patch_records:
                raise RuntimeError(
                    "current_memory_cache_root is for QA-only mixed-cache reruns, but the patch cache "
                    f"has no patch records for sample {sample_id}. Build or locate the matching patch cache first, "
                    "or remove --current_memory_cache_root."
                )
            self.eval_logger.info(
                "Loading robust current memory plus %d cached patch records for %s",
                len(patch_records),
                chat_history_path.name,
            )
        elif agent.has_cached_state():
            self.eval_logger.info("Loading cached patch A-MEM state for %s", chat_history_path.name)
        else:
            messages = load_chat_history_messages(chat_history_path)
            resume_message_index = agent.memory_system.get_resume_turn_index() if agent.memory_system is not None else 0
            if resume_message_index > 0:
                self.eval_logger.info(
                    "Resuming patch A-MEM state for %s from message %s/%s",
                    chat_history_path.name,
                    resume_message_index,
                    len(messages),
                )
            else:
                self.eval_logger.info("Building patch A-MEM state for %s", chat_history_path.name)
            loaded_messages = resume_message_index
            for idx, message in enumerate(
                make_progress(messages[resume_message_index:], desc=f"Build patch memory {chat_history_path.stem}", leave=False),
                start=resume_message_index,
            ):
                role = str(message.get("role", "")).lower()
                if not self.include_system_messages and role == "system":
                    continue
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                try:
                    agent.add_memory(serialize_chat_message(message, idx), idx, role or "unknown")
                except Exception as e:
                    if is_nonrecoverable_llm_error(e):
                        if agent.memory_system is not None:
                            agent.memory_system.mark_sample_failed(
                                turn_position=idx,
                                turn_number=idx + 1,
                                exc=e,
                            )
                        self.eval_logger.error(
                            "Fatal LLM error while building patch A-MEM for %s at message %s/%s; clean resume index is %s",
                            chat_history_path.name,
                            idx,
                            len(messages),
                            agent.memory_system.get_resume_turn_index() if agent.memory_system is not None else idx,
                        )
                    raise
                loaded_messages += 1
            agent.mark_complete()
            self.eval_logger.info(
                "Cached %d messages for %s", loaded_messages, chat_history_path.name
            )

        self.agent_cache[sample_id] = agent
        while len(self.agent_cache) > self.max_live_agents:
            self.agent_cache.popitem(last=False)
        return agent


def evaluate_persona_benchmark(
    benchmark_file: Path,
    model: str,
    backend: str,
    size: str,
    output_path: Path,
    persona_root: Path,
    retrieve_k: int,
    patch_top_k: int,
    patch_usage: str,
    answer_temperature: float,
    sglang_host: str,
    sglang_port: int,
    api_base: Optional[str],
    include_system_messages: bool,
    max_live_agents: int,
    num_workers: int,
    worker_id: int,
    persona_ids: Optional[List[str]],
    max_items: Optional[int],
    save_debug_columns: bool,
    eval_mode: str,
    cache_root: Optional[Path] = None,
    current_memory_cache_root: Optional[Path] = None,
    exclude_revoke_patches: bool = False,
    exclude_add_patches: bool = False,
    min_patch_similarity: float = 0.0,
    force_reingest_patches: bool = False,
    preference_aware_level: str = "none",
    require_pref_change: bool = False,
    llm_patch_filter: bool = False,
    gt_patch_file: Optional[Path] = None,
    gt_patch_top_k: Optional[int] = None,
    gt_patch_min_similarity: Optional[float] = None,
    gp_patch_retrieval: str = "similarity",
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    worker_suffix = f"_worker{worker_id}" if num_workers > 1 else ""
    log_path = Path(__file__).resolve().parent / "logs" / (
        f"persona_patch_{sanitize_filename_part(model)}_{backend}_{size}_{patch_usage}{worker_suffix}_{timestamp}.log"
    )
    eval_logger = setup_logger(log_path)

    eval_logger.info("Persona root: %s", persona_root)
    eval_logger.info("Loading benchmark from %s", benchmark_file)
    eval_logger.info("Size: %s", size)
    eval_logger.info("Backend: %s", backend)
    eval_logger.info("Patch usage: %s", patch_usage)
    eval_logger.info("Patch top_k: %s", patch_top_k)
    if current_memory_cache_root is not None:
        eval_logger.info("Current memory override cache root: %s", current_memory_cache_root)
    eval_logger.info("Using patch-augmented A-MEM memory layer")

    fieldnames, rows = load_benchmark_rows(benchmark_file, persona_ids=persona_ids, max_items=max_items)
    eval_logger.info("Loaded %d benchmark rows before worker sharding", len(rows))

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")
    if num_workers > 1:
        rows = shard_rows_by_chat_history(rows, size=size, persona_root=persona_root, num_workers=num_workers, worker_id=worker_id)
        eval_logger.info("Worker %d/%d processing %d rows after chat-history sharding", worker_id, num_workers, len(rows))

    if cache_root is not None:
        cache_dir = Path(cache_root).resolve()
    else:
        pref_suffix_map = {
            "none": "",
            "patch_only": "_prefaware_patchonly",
            "full": "_prefaware_full",
        }
        pref_suffix = pref_suffix_map.get(preference_aware_level, "")
        cache_dir = Path(__file__).resolve().parent / (
            f"cached_memories_persona_patch_{backend}_{sanitize_filename_part(model)}_{size}_{patch_usage}{pref_suffix}"
        )
    cache_dir.mkdir(parents=True, exist_ok=True)

    agent_manager = AgentManager(
        model=model,
        backend=backend,
        retrieve_k=retrieve_k,
        patch_top_k=patch_top_k,
        patch_usage=patch_usage,
        answer_temperature=answer_temperature,
        cache_dir=cache_dir,
        current_memory_cache_root=current_memory_cache_root,
        include_system_messages=include_system_messages,
        max_live_agents=max_live_agents,
        sglang_host=sglang_host,
        sglang_port=sglang_port,
        api_base=api_base,
        eval_logger=eval_logger,
        exclude_revoke_patches=exclude_revoke_patches,
        exclude_add_patches=exclude_add_patches,
        min_patch_similarity=min_patch_similarity,
        force_reingest_patches=force_reingest_patches,
        preference_aware_level=preference_aware_level,
        require_pref_change=require_pref_change,
        llm_patch_filter=llm_patch_filter,
        gt_patch_file=gt_patch_file,
        gt_patch_top_k=gt_patch_top_k,
        gt_patch_min_similarity=gt_patch_min_similarity,
        gp_patch_retrieval=gp_patch_retrieval,
    )

    output_fieldnames = list(fieldnames)
    result_columns: List[str] = []
    if eval_mode in ("mcq", "both"):
        result_columns.extend(
            [
                f"model_response_mcq_{size}",
                f"predicted_answer_mcq_{size}",
                f"is_correct_mcq_{size}",
                f"correct_mcq_option_{size}",
                f"raw_input_prompt_mcq_{size}",
            ]
        )
        if save_debug_columns:
            result_columns.extend(
                [
                    f"retrieval_keywords_{size}",
                    f"retrieved_context_{size}",
                    f"narrowed_context_{size}",
                    f"amem_prompt_{size}",
                    f"answer_metadata_mcq_{size}",
                ]
            )
    if eval_mode in ("generative", "both"):
        result_columns.extend(
            [
                f"model_response_openended_{size}",
                f"is_correct_openended_{size}",
                f"raw_input_prompt_openended_{size}",
            ]
        )
        if save_debug_columns:
            result_columns.extend(
                [
                    f"retrieval_keywords_openended_{size}",
                    f"retrieved_context_openended_{size}",
                    f"narrowed_context_openended_{size}",
                    f"amem_prompt_openended_{size}",
                    f"answer_metadata_openended_{size}",
                ]
            )
    for column in result_columns:
        if column not in output_fieldnames:
            output_fieldnames.append(column)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    existing_rows, existing_fieldnames = load_existing_output_rows(output_path)
    if existing_fieldnames:
        output_fieldnames = existing_fieldnames
    resume_row_index = len(existing_rows)
    if resume_row_index:
        eval_logger.info("Resuming from %d existing result rows in %s", resume_row_index, output_path)
    written_rows: List[Dict[str, str]] = list(existing_rows)
    correct = sum(1 for row in written_rows if row.get(f"is_correct_mcq_{size}") == "True")
    mcq_processed = sum(1 for row in written_rows if row.get(f"is_correct_mcq_{size}") in ("True", "False"))
    usage_log_path = output_path.with_name(f"{output_path.stem}_llm_usage.jsonl")
    if not resume_row_index and usage_log_path.exists():
        usage_log_path.unlink()
    usage_recorder = LLMUsageRecorder(
        usage_log_path,
        metadata={
            "suite": "persona",
            "method": "patch",
            "model": model,
            "backend": backend,
            "size": size,
            "benchmark_file": str(benchmark_file),
            "output_path": str(output_path),
            "worker_id": worker_id,
            "num_workers": num_workers,
            "patch_usage": patch_usage,
        },
    )

    file_mode = "a" if resume_row_index else "w"
    with output_path.open(file_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        if not resume_row_index:
            writer.writeheader()

        for idx, row in enumerate(make_progress(rows[resume_row_index:], desc="Persona patch eval"), start=resume_row_index):
            output_row = row.copy()
            try:
                chat_history_path = resolve_chat_history_path(row, size=size, persona_root=persona_root)
                with llm_usage_recorder(usage_recorder), llm_usage_stage("ingestion"):
                    agent = agent_manager.get_agent(chat_history_path, str(row.get("persona_id", "")))
                agent.set_current_row(row)
                user_query = parse_user_query(row.get("user_query", ""))
                question = user_query.get("content", "")

                if eval_mode in ("mcq", "both"):
                    _mcq_instruction, option_mapping, correct_letter = create_mcq_options(
                        row.get("correct_answer", ""),
                        parse_incorrect_answers(row.get("incorrect_answers", "")),
                        seed=stable_int_seed(row.get("persona_id", ""), row.get("user_query", "")),
                    )
                    with llm_usage_recorder(usage_recorder), llm_usage_stage("qa.mcq"):
                        answer_result = agent.answer_mcq(question, option_mapping)
                    prediction = extract_final_answer(str(answer_result["response"]))
                    is_correct = check_mcq_correctness(
                        prediction,
                        row.get("correct_answer", ""),
                        option_mapping,
                    )
                    output_row[f"model_response_mcq_{size}"] = str(answer_result["response"])
                    output_row[f"predicted_answer_mcq_{size}"] = prediction
                    output_row[f"is_correct_mcq_{size}"] = str(is_correct)
                    output_row[f"correct_mcq_option_{size}"] = correct_letter
                    output_row[f"raw_input_prompt_mcq_{size}"] = str(answer_result["prompt"])

                    if save_debug_columns:
                        output_row[f"retrieval_keywords_{size}"] = str(answer_result["keywords"])
                        output_row[f"retrieved_context_{size}"] = str(answer_result["context"])
                        output_row[f"narrowed_context_{size}"] = str(answer_result["narrowed_context"])
                        output_row[f"amem_prompt_{size}"] = str(answer_result["prompt"])
                        output_row[f"answer_metadata_mcq_{size}"] = json.dumps(
                            answer_result["answer_metadata"], ensure_ascii=False
                        )

                    if is_correct:
                        correct += 1
                    mcq_processed += 1

                if eval_mode in ("generative", "both"):
                    with llm_usage_recorder(usage_recorder), llm_usage_stage("qa.openended"):
                        openended_result = agent.answer_openended(question)
                    output_row[f"model_response_openended_{size}"] = str(openended_result["response"])
                    output_row[f"is_correct_openended_{size}"] = ""
                    output_row[f"raw_input_prompt_openended_{size}"] = str(openended_result["prompt"])

                    if save_debug_columns:
                        output_row[f"retrieval_keywords_openended_{size}"] = str(openended_result["keywords"])
                        output_row[f"retrieved_context_openended_{size}"] = str(openended_result["context"])
                        output_row[f"narrowed_context_openended_{size}"] = str(openended_result["narrowed_context"])
                        output_row[f"amem_prompt_openended_{size}"] = str(openended_result["prompt"])
                        output_row[f"answer_metadata_openended_{size}"] = json.dumps(
                            openended_result["answer_metadata"], ensure_ascii=False
                        )
            except Exception as e:
                if is_nonrecoverable_llm_error(e):
                    raise
                if eval_mode in ("mcq", "both"):
                    output_row[f"model_response_mcq_{size}"] = f"ERROR: {e}"
                    output_row[f"predicted_answer_mcq_{size}"] = ""
                    output_row[f"is_correct_mcq_{size}"] = ""
                    output_row[f"correct_mcq_option_{size}"] = ""
                    output_row[f"raw_input_prompt_mcq_{size}"] = ""
                if eval_mode in ("generative", "both"):
                    output_row[f"model_response_openended_{size}"] = f"ERROR: {e}"
                    output_row[f"is_correct_openended_{size}"] = ""
                    output_row[f"raw_input_prompt_openended_{size}"] = ""
                eval_logger.exception("Failed on row %d (persona_id=%s)", idx, row.get("persona_id", ""))

            writer.writerow(output_row)
            f.flush()
            written_rows.append(output_row)

    metrics_path = write_metrics_file(output_path, written_rows, size)
    usage_summary_path = write_usage_summary_file(output_path, usage_recorder)
    if mcq_processed:
        eval_logger.info("Overall accuracy: %.3f (%d/%d)", correct / mcq_processed, correct, mcq_processed)
    eval_logger.info("Results saved to %s", output_path)
    eval_logger.info("Metrics saved to %s", metrics_path)
    eval_logger.info("LLM usage saved to %s", usage_log_path)
    eval_logger.info("LLM usage summary saved to %s", usage_summary_path)
    return output_path


def _worker_output_path(output_path: Path, worker_id: int) -> Path:
    suffix = output_path.suffix or ".csv"
    stem = output_path.stem if output_path.suffix else output_path.name
    return output_path.with_name(f"{stem}.worker_{worker_id}{suffix}")


def merge_worker_outputs(worker_outputs: List[Path], merged_output: Path, size: str) -> Path:
    rows: List[Dict[str, str]] = []
    fieldnames: List[str] = []

    for worker_output in worker_outputs:
        with worker_output.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            if not fieldnames:
                fieldnames = reader.fieldnames or []
            for row in reader:
                rows.append(row)

    merged_output.parent.mkdir(parents=True, exist_ok=True)
    with merged_output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    write_metrics_file(merged_output, rows, size)
    merged_usage_log = merged_output.with_name(f"{merged_output.stem}_llm_usage.jsonl")
    if merged_usage_log.exists():
        merged_usage_log.unlink()
    merged_usage_recorder = LLMUsageRecorder(
        merged_usage_log,
        metadata={
            "merged_output": str(merged_output),
            "worker_outputs": [str(path) for path in worker_outputs],
        },
    )
    for worker_output in worker_outputs:
        worker_usage_log = worker_output.with_name(f"{worker_output.stem}_llm_usage.jsonl")
        if not worker_usage_log.exists():
            continue
        with worker_usage_log.open("r", encoding="utf-8") as f_usage:
            for line in f_usage:
                if not line.strip():
                    continue
                merged_usage_recorder.record(json.loads(line))
    write_usage_summary_file(merged_output, merged_usage_recorder)
    return merged_output


def run_batch_workers(args: argparse.Namespace, benchmark_file: Path, output_path: Path) -> Path:
    if args.batch < 1:
        raise ValueError("batch must be at least 1")

    child_base = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--benchmark_file", str(benchmark_file),
        "--model", args.model,
        "--backend", args.backend,
        "--size", args.size,
        "--retrieve_k", str(args.retrieve_k),
        "--patch_top_k", str(args.patch_top_k),
        "--patch_usage", args.patch_usage,
        "--answer_temperature", str(args.answer_temperature),
        "--num_workers", str(args.batch),
        "--sglang_host", args.sglang_host,
        "--sglang_port", str(args.sglang_port),
        "--max_live_agents", str(args.max_live_agents),
        "--eval_mode", args.eval_mode,
    ]
    if args.persona_root:
        child_base.extend(["--persona_root", str(args.persona_root)])
    if args.persona_ids:
        child_base.extend(["--persona_ids", args.persona_ids])
    if args.max_items is not None:
        child_base.extend(["--max_items", str(args.max_items)])
    if not args.include_system_messages:
        child_base.append("--skip_system_messages")
    if args.save_debug_columns:
        child_base.append("--save_debug_columns")
    if args.cache_root:
        child_base.extend(["--cache_root", args.cache_root])
    if args.current_memory_cache_root:
        child_base.extend(["--current_memory_cache_root", args.current_memory_cache_root])
    if args.exclude_revoke_patches:
        child_base.append("--exclude_revoke_patches")
    if args.exclude_add_patches:
        child_base.append("--exclude_add_patches")
    if args.min_patch_similarity > 0.0:
        child_base.extend(["--min_patch_similarity", str(args.min_patch_similarity)])
    if args.force_reingest_patches:
        child_base.append("--force_reingest_patches")
    if args.preference_aware_level and args.preference_aware_level != "none":
        child_base.extend(["--preference_aware_level", args.preference_aware_level])
    elif args.preference_aware:
        child_base.append("--preference_aware")
    if args.require_pref_change:
        child_base.append("--require_pref_change")
    if args.llm_patch_filter:
        child_base.append("--llm_patch_filter")
    if args.gt_patch:
        child_base.append("--gt_patch")
    if args.gt_patch_file:
        child_base.extend(["--gt_patch_file", args.gt_patch_file])
    if args.gt_patch_top_k is not None:
        child_base.extend(["--gt_patch_top_k", str(args.gt_patch_top_k)])
    if args.gt_patch_min_similarity is not None:
        child_base.extend(["--gt_patch_min_similarity", str(args.gt_patch_min_similarity)])
    if args.gp_patch_retrieval != "similarity":
        child_base.extend(["--gp_patch_retrieval", args.gp_patch_retrieval])

    worker_outputs: List[Path] = []
    processes = []
    for worker_id in range(args.batch):
        worker_output = _worker_output_path(output_path, worker_id)
        cmd = child_base + ["--worker_id", str(worker_id), "--output", str(worker_output)]
        worker_stderr = worker_output.with_suffix(worker_output.suffix + ".stderr.log")
        worker_stderr.parent.mkdir(parents=True, exist_ok=True)
        stderr_handle = worker_stderr.open("w", encoding="utf-8")
        process = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent),
            stdout=None,
            stderr=stderr_handle,
        )
        stderr_handle.close()
        processes.append((worker_id, worker_output, worker_stderr, process))
        worker_outputs.append(worker_output)

    failures = []
    remaining = list(processes)
    while remaining:
        progressed = False
        for worker_id, worker_output, worker_stderr, process in list(remaining):
            return_code = process.poll()
            if return_code is None:
                continue
            progressed = True
            remaining.remove((worker_id, worker_output, worker_stderr, process))
            if return_code != 0:
                failures.append((worker_id, return_code))
                logger.error(
                    "worker_id=%s failed exit=%s stderr_log=%s; terminating remaining workers",
                    worker_id,
                    return_code,
                    worker_stderr,
                )
                for other_worker_id, _other_output, other_stderr, other_process in remaining:
                    if other_process.poll() is None:
                        logger.error("terminating worker_id=%s stderr_log=%s", other_worker_id, other_stderr)
                        other_process.terminate()
                for other_worker_id, _other_output, other_stderr, other_process in remaining:
                    try:
                        other_process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        logger.error("killing worker_id=%s stderr_log=%s", other_worker_id, other_stderr)
                        other_process.kill()
                        other_process.wait()
                remaining.clear()
                break
            logger.info("worker_id=%s completed output=%s stderr_log=%s", worker_id, worker_output, worker_stderr)
        if remaining and not progressed:
            time.sleep(1.0)

    if failures:
        failed = ", ".join(f"worker {worker_id} (exit {code})" for worker_id, code in failures)
        raise RuntimeError(f"batch run failed: {failed}")

    return merge_worker_outputs(worker_outputs, output_path, args.size)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run patch-augmented A-MEM on Persona-release benchmark")
    parser.add_argument(
        "--benchmark_file",
        type=str,
        default="data/Persona-release/benchmark_v34/text/benchmark_49p_ood_v34.csv",
        help="Path to the Persona benchmark CSV",
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model to use")
    parser.add_argument(
        "--backend",
        type=str,
        default="openai",
        choices=["openai", "openrouter", "ollama", "sglang", "vllm"],
        help="LLM backend",
    )
    parser.add_argument("--size", type=str, default="32k", help="Chat history size column to use")
    parser.add_argument("--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--eval_mode",
        type=str,
        choices=["mcq", "generative", "both"],
        default="mcq",
        help="Evaluation mode: mcq, generative, or both",
    )
    parser.add_argument("--persona_root", type=str, default=None, help="Path to Persona-release root")
    parser.add_argument("--persona_ids", type=str, default=None, help="Comma-separated persona ids filter")
    parser.add_argument("--max_items", type=int, default=None, help="Maximum rows to process")
    parser.add_argument("--retrieve_k", type=int, default=12, help="Number of current memories to retrieve")
    parser.add_argument("--patch_top_k", type=int, default=3, help="Number of historical patches to retrieve")
    parser.add_argument(
        "--patch_usage",
        type=str,
        default="always",
        choices=["always", "gated"],
        help="How to use historical patches at answer time",
    )
    parser.add_argument(
        "--answer_temperature",
        type=float,
        default=0.0,
        help="Temperature for final answer calls",
    )
    parser.add_argument("--sglang_host", type=str, default="http://localhost", help="SGLang host")
    parser.add_argument("--sglang_port", type=int, default=30000, help="SGLang port")
    parser.add_argument(
        "--skip_system_messages",
        action="store_true",
        help="Do not index chat-history system messages into A-MEM",
    )
    parser.add_argument(
        "--save_debug_columns",
        action="store_true",
        help="Write retrieval keywords, contexts, prompts, and patch metadata into the output CSV",
    )
    parser.add_argument(
        "--max_live_agents",
        type=int,
        default=2,
        help="Maximum number of cached in-memory agent instances",
    )
    parser.add_argument("--num_workers", type=int, default=1, help="Total workers for manual sharding")
    parser.add_argument("--worker_id", type=int, default=0, help="Worker id for manual sharding")
    parser.add_argument("--batch", type=int, default=None, help="Launch this many worker processes and merge outputs")
    parser.add_argument("--cache_root", type=str, default=None, help="Override patch cache directory (skip auto-naming)")
    parser.add_argument(
        "--current_memory_cache_root",
        type=str,
        default=None,
        help=(
            "Optional robust/baseline cache root for QA-only mixed-cache reruns. "
            "When set, current global memory evidence is loaded from this root while "
            "historical patch records still come from --cache_root."
        ),
    )
    parser.add_argument("--exclude_revoke_patches", action="store_true", default=False,
                        help="Skip REVOKE_PREFERENCE patches from context injection")
    parser.add_argument("--exclude_add_patches", action="store_true", default=False,
                        help="Skip ADD_PREFERENCE patches (already captured in global memory)")
    parser.add_argument("--min_patch_similarity", type=float, default=0.0,
                        help="Minimum cosine similarity threshold to inject a patch (0=no filter, 0.5=recommended)")
    parser.add_argument("--force_reingest_patches", action="store_true", default=False,
                        help="Clear existing global graph cache AND patch files, then re-ingest everything from scratch")
    parser.add_argument("--preference_aware", action="store_true", default=False,
                        help="[legacy] Use preference-aware prompts for memory building and patch classification. "
                             "Equivalent to --preference_aware_level full.")
    parser.add_argument("--preference_aware_level", type=str, default="none",
                        choices=["none", "patch_only", "full"],
                        help="Granular preference-aware mode: "
                             "'none' = original prompts; "
                             "'patch_only' = PREF ANALYZE + PREF patch prompts with base graph prompts; "
                             "'full' = PREF prompts everywhere (legacy --preference_aware behavior).")
    parser.add_argument("--require_pref_change", action="store_true", default=False,
                        help="Require IS_PREFERENCE_CHANGE=YES to commit a patch. "
                             "Default (False) commits a patch for any evolution that modifies memory, "
                             "capturing subtle natural-language changes that the LLM may not classify as explicit preference changes.")
    parser.add_argument("--llm_patch_filter", action="store_true", default=False,
                        help="Run an LLM relevance filter over SBERT-retrieved patch candidates before injection.")
    parser.add_argument("--gt_patch", action="store_true", default=False,
                        help="Retrieve additional ground-truth benchmark patches for the current sample and append them to context.")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Optional OpenAI-compatible API base URL. Useful for custom OpenRouter-compatible endpoints.")
    parser.add_argument("--gt_patch_file", type=str, default=None,
                        help="Path to a JSONL gold patch store. Defaults to analysis/gold_persona_patches_9p_nonood.jsonl when --gt_patch is set.")
    parser.add_argument("--gt_patch_top_k", type=int, default=None,
                        help="Top-k ground-truth patches to retrieve per sample. Defaults to --patch_top_k.")
    parser.add_argument("--gt_patch_min_similarity", type=float, default=None,
                        help="Minimum cosine similarity for retrieved ground-truth patches. Defaults to --min_patch_similarity.")
    parser.add_argument("--gp_patch_retrieval", choices=["similarity", "oracle"], default="similarity",
                        help="How to select ground-truth patches: similarity retrieval within the sample, or oracle row-aligned injection for changed rows.")
    args = parser.parse_args()

    # Resolve legacy flag → level, with level taking precedence if both given.
    if args.preference_aware_level == "none" and args.preference_aware:
        args.preference_aware_level = "full"

    normalize_openai_env()

    benchmark_file = Path(args.benchmark_file).resolve()
    persona_root = resolve_persona_root(benchmark_file, args.persona_root)
    persona_ids = parse_persona_ids(args.persona_ids)
    output_path = build_output_path(
        args.output,
        persona_root,
        benchmark_file,
        args.model,
        args.size,
        args.patch_usage,
        args.worker_id,
        args.num_workers,
    )
    include_system_messages = not args.skip_system_messages
    args.include_system_messages = include_system_messages

    if args.batch is not None:
        if args.batch == 1:
            args.num_workers = 1
            args.worker_id = 0
        else:
            run_batch_workers(args, benchmark_file, output_path)
            return

    evaluate_persona_benchmark(
        benchmark_file=benchmark_file,
        model=args.model,
        backend=args.backend,
        size=args.size,
        output_path=output_path,
        persona_root=persona_root,
        retrieve_k=args.retrieve_k,
        patch_top_k=args.patch_top_k,
        patch_usage=args.patch_usage,
        answer_temperature=args.answer_temperature,
        sglang_host=args.sglang_host,
        sglang_port=args.sglang_port,
        api_base=args.api_base,
        include_system_messages=include_system_messages,
        max_live_agents=args.max_live_agents,
        num_workers=args.num_workers,
        worker_id=args.worker_id,
        persona_ids=persona_ids,
        max_items=args.max_items,
        save_debug_columns=args.save_debug_columns,
        eval_mode=args.eval_mode,
        cache_root=Path(args.cache_root) if args.cache_root else None,
        current_memory_cache_root=Path(args.current_memory_cache_root) if args.current_memory_cache_root else None,
        exclude_revoke_patches=args.exclude_revoke_patches,
        exclude_add_patches=args.exclude_add_patches,
        min_patch_similarity=args.min_patch_similarity,
        force_reingest_patches=args.force_reingest_patches,
        preference_aware_level=args.preference_aware_level,
        require_pref_change=args.require_pref_change,
        llm_patch_filter=args.llm_patch_filter,
        gt_patch_file=(Path(args.gt_patch_file) if args.gt_patch_file else (Path(__file__).resolve().parent / "analysis/gold_persona_patches_9p_nonood.jsonl" if args.gt_patch else None)),
        gt_patch_top_k=args.gt_patch_top_k,
        gt_patch_min_similarity=args.gt_patch_min_similarity,
        gp_patch_retrieval=args.gp_patch_retrieval,
    )


if __name__ == "__main__":
    main()
