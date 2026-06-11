"""Patch-augmented wrapper around the robust A-Mem memory layer."""

from __future__ import annotations

import copy
import json
import logging
import os
import pickle
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

try:
    import numpy as np
except ImportError:  # pragma: no cover - optional runtime dependency
    np = None

try:
    from rank_bm25 import BM25Okapi
except ImportError:  # pragma: no cover - optional runtime dependency
    BM25Okapi = None

from memory_layer import SimpleEmbeddingRetriever, simple_tokenize
from memory_layer_robust import RobustAgenticMemorySystem, RobustLLMController, is_nonrecoverable_llm_error
from patch_prompts import (
    PATCH_CONTEXT_INSTRUCTION,
    PATCH_CONTEXT_PREF_INSTRUCTION,
    PATCH_DETAIL_REVISION_PROMPT,
    PATCH_GATING_PROMPT,
    PATCH_RELEVANCE_FILTER_PROMPT,
    PATCH_SUMMARIZATION_PROMPT,
    PATCH_SUMMARIZATION_PREF_PROMPT,
)
from patch_store import PatchStore
from locomo_eval_utils import (
    build_locomo_answer_instruction,
    build_locomo_answer_prompt,
    locomo_answer_temperature,
)

logger = logging.getLogger("amem_patch_layer")


@dataclass
class PatchConfig:
    patch_top_k: int = 2
    retrieve_k_current: int = 10
    patch_usage: str = "always"
    exclude_revoke_patches: bool = False  # skip REVOKE_PREFERENCE patches from context injection
    exclude_add_patches: bool = False     # skip ADD_PREFERENCE patches (already in global memory)
    min_patch_similarity: float = 0.0    # minimum cosine similarity to inject a patch (0=no filter)
    force_reingest_patches: bool = False  # clear cached sample state and re-ingest from scratch
    # Preference-aware mode:
    #   none        - original robust + patch prompts everywhere
    #   patch_only  - PREF ANALYZE (for is_preference) + PREF PATCH prompts; base graph prompts
    #   full        - PREF prompts for both graph + patch (current legacy "prefaware" behavior)
    preference_aware_level: str = "none"
    preference_aware: bool = False        # legacy bool — maps to level=full when True
    require_preference_change: bool = False  # if False (default), commit patch for any evolution even when
                                             # LLM says IS_PREFERENCE_CHANGE=NO; if True, restore old gate
    llm_patch_filter: bool = False           # if True, run LLM relevance filter after SBERT retrieval to
                                             # remove cross-domain patches before injection
    llm_patch_filter_candidates: int = 15    # how many SBERT candidates to pass to the LLM filter
    patch_node_rerank: bool = False          # if True, select only the most relevant nodes/links from retrieved patches
    patch_node_top_k: int = 2                # number of node/link evidence items to keep after reranking
    patch_node_query_mode: str = "expanded"  # how to build the node/link rerank query
    patch_hybrid_retrieval: bool = False     # if True, combine embedding and BM25 scores at patch level
    patch_hybrid_alpha: float = 0.7          # embedding weight for hybrid retrieval
    patch_hybrid_node_rerank: bool = False   # if True, also use hybrid scoring for node/link reranking

    def __post_init__(self) -> None:
        if self.preference_aware and self.preference_aware_level == "none":
            # Legacy callers still pass the bool; promote to "full" to match prior behavior.
            self.preference_aware_level = "full"
        if self.preference_aware_level not in ("none", "patch_only", "full"):
            raise ValueError(
                f"preference_aware_level must be one of none|patch_only|full, got {self.preference_aware_level!r}"
            )
        if self.patch_node_query_mode not in ("question_only", "keywords_only", "expanded", "answer_style"):
            raise ValueError(
                "patch_node_query_mode must be one of question_only|keywords_only|expanded|answer_style"
            )
        # Keep the bool in sync so existing code paths reading `.preference_aware` still work.
        self.preference_aware = self.preference_aware_level != "none"


class PatchAugmentedMemorySystem:
    """Wrapper that adds patch history on top of the robust global memory graph."""

    def __init__(
        self,
        sample_id: str,
        model_name: str = "all-MiniLM-L6-v2",
        llm_backend: str = "sglang",
        llm_model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        store_root: Optional[str] = None,
        config: Optional[PatchConfig] = None,
    ):
        self.sample_id = sample_id
        self.config = config or PatchConfig()
        self.model_name = model_name
        self.llm_backend = llm_backend
        self.llm_model = llm_model
        self.api_key = api_key
        self.api_base = api_base
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.store = PatchStore(
            store_root or os.path.join(os.path.dirname(__file__), f"cached_memories_patch_{llm_backend}_{llm_model}")
        )
        self.patch_llm = RobustLLMController(
            backend=llm_backend,
            model=llm_model,
            api_key=api_key,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.loaded_from_complete_cache = False
        self._initialize_runtime()

    def _initialize_runtime(self) -> None:
        self.base_system = RobustAgenticMemorySystem(
            model_name=self.model_name,
            llm_backend=self.llm_backend,
            llm_model=self.llm_model,
            api_key=self.api_key,
            api_base=self.api_base,
            sglang_host=self.sglang_host,
            sglang_port=self.sglang_port,
            preference_aware_level=self.config.preference_aware_level,
        )
        self.patch_retriever = SimpleEmbeddingRetriever(self.model_name)
        self._patch_counter = 0
        self.loaded_from_complete_cache = False
        if self.config.force_reingest_patches:
            self.store.clear_global_graph_cache(self.sample_id)
            self.store.clear_patch_data(self.sample_id)
        self._load_global_graph_cache_if_exists()
        self._load_or_build_patch_retriever()
        self._sync_patch_counter()

    def set_sample(self, sample_id: str) -> None:
        if self.sample_id == sample_id and (self.base_system.memories or self.store.load_build_status(sample_id)):
            return
        self.sample_id = sample_id
        self._initialize_runtime()

    def _sync_patch_counter(self) -> None:
        patches = self.store.load_all_patches(self.sample_id)
        if not patches:
            self._patch_counter = 0
            return
        max_id = 0
        for patch in patches:
            try:
                max_id = max(max_id, int(str(patch.get('patch_id', '0')).split('_')[-1]))
            except ValueError:
                continue
        self._patch_counter = max_id

    def has_complete_global_graph_cache(self) -> bool:
        if self.config.force_reingest_patches:
            return False
        status = self.store.load_build_status(self.sample_id) or {}
        memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file = self.store.global_graph_paths(self.sample_id)
        return bool(
            status.get('global_graph_complete')
            and os.path.exists(memory_cache_file)
            and os.path.exists(retriever_cache_file)
            and os.path.exists(retriever_cache_embeddings_file)
        )

    def _load_global_graph_cache_if_exists(self) -> None:
        if self.config.force_reingest_patches:
            return
        memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file = self.store.global_graph_paths(self.sample_id)
        status = self.store.load_build_status(self.sample_id) or {}
        if not (
            os.path.exists(memory_cache_file)
            and os.path.exists(retriever_cache_file)
            and os.path.exists(retriever_cache_embeddings_file)
        ):
            return
        with open(memory_cache_file, 'rb') as f:
            self.base_system.memories = pickle.load(f)
        self.base_system.retriever = self.base_system.retriever.load(
            retriever_cache_file, retriever_cache_embeddings_file
        )
        self.loaded_from_complete_cache = bool(status.get('global_graph_complete', False))

    def get_build_status(self) -> Dict[str, Any]:
        return self.store.load_build_status(self.sample_id) or {}

    def get_resume_turn_index(self) -> int:
        status = self.get_build_status()
        if status.get("fatal_error") and status.get("resume_turn_index") is not None:
            try:
                return max(0, int(status.get("resume_turn_index")))
            except (TypeError, ValueError):
                pass
        last_turn_number = status.get("last_turn_number")
        if last_turn_number is not None:
            try:
                return max(0, int(last_turn_number))
            except (TypeError, ValueError):
                pass
        last_turn_position = status.get("last_turn_position")
        if last_turn_position is not None:
            try:
                return max(0, int(last_turn_position) + 1)
            except (TypeError, ValueError):
                pass
        return 0

    def _save_global_graph_cache(
        self,
        session_id: Optional[int] = None,
        turn_position: Optional[int] = None,
        turn_number: Optional[int] = None,
        complete: bool = False,
    ) -> None:
        memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file = self.store.global_graph_paths(self.sample_id)
        with open(memory_cache_file, 'wb') as f:
            pickle.dump(self.base_system.memories, f)
        self.base_system.retriever.save(retriever_cache_file, retriever_cache_embeddings_file)
        self.store.save_build_status(
            self.sample_id,
            {
                'sample_id': self.sample_id,
                'global_graph_complete': complete,
                'last_session_id': session_id,
                'last_turn_position': turn_position,
                'last_turn_number': turn_number,
                'memory_count': len(self.base_system.memories),
                'updated_at': datetime.utcnow().isoformat() + 'Z',
            },
        )

    def mark_sample_complete(self) -> None:
        status = self.store.load_build_status(self.sample_id) or {}
        self._save_global_graph_cache(
            session_id=status.get('last_session_id'),
            turn_position=status.get('last_turn_position'),
            turn_number=status.get('last_turn_number'),
            complete=True,
        )
        self.loaded_from_complete_cache = True

    def mark_sample_failed(
        self,
        *,
        turn_position: Optional[int],
        turn_number: Optional[int],
        exc: BaseException,
    ) -> None:
        # Do not save the in-memory graph here; it may have been partially mutated
        # by the failing LLM call. Keep the previous cache as the clean resume point.
        status = self.store.load_build_status(self.sample_id) or {}
        status.update(
            {
                'sample_id': self.sample_id,
                'global_graph_complete': False,
                'fatal_error': True,
                'fatal_error_stage': 'memory_build',
                'failed_turn_position': turn_position,
                'failed_turn_number': turn_number,
                'resume_turn_index': self.get_resume_turn_index(),
                'fatal_error_type': f"{exc.__class__.__module__}.{exc.__class__.__name__}",
                'fatal_error_message': str(exc),
                'updated_at': datetime.utcnow().isoformat() + 'Z',
            }
        )
        self.store.save_build_status(self.sample_id, status)

    def _load_or_build_patch_retriever(self) -> None:
        index_records = self.store.load_patch_index_records(self.sample_id)
        if not index_records:
            return
        index_records, index_changed = self._refresh_patch_index_records(index_records)
        cache_file, embeddings_file = self.store.patch_retriever_paths(self.sample_id)
        if (not index_changed) and os.path.exists(cache_file) and os.path.exists(embeddings_file):
            self.patch_retriever = self.patch_retriever.load(cache_file, embeddings_file)
            return
        self.patch_retriever = SimpleEmbeddingRetriever(self.model_name)
        self.patch_retriever.add_documents([r["index_document"] for r in index_records])
        self._save_patch_retriever()

    def _refresh_patch_index_records(self, index_records: List[Dict[str, Any]]) -> tuple[List[Dict[str, Any]], bool]:
        refreshed_records: List[Dict[str, Any]] = []
        changed = False
        for record in index_records:
            patch_id = str(record.get("patch_id", ""))
            patch_record = self.store.load_patch(self.sample_id, patch_id) or {}
            if not patch_record:
                refreshed_records.append(record)
                continue
            rebuilt_doc = self.build_patch_index_document(patch_record)
            refreshed = dict(record)
            refreshed["index_document"] = rebuilt_doc
            trigger_turn = patch_record.get("trigger_turn") or {}
            refreshed["session_id"] = trigger_turn.get("session_id", refreshed.get("session_id", 0))
            refreshed["turn_position"] = trigger_turn.get("turn_position", refreshed.get("turn_position", 0))
            refreshed["temporal_order"] = f"{int(refreshed.get('session_id', 0) or 0):06d}_{int(refreshed.get('turn_position', 0) or 0):08d}"
            patch_record_changed = patch_record.get("index_document") != rebuilt_doc
            record_changed = record.get("index_document") != rebuilt_doc
            if patch_record_changed:
                patch_record["index_document"] = rebuilt_doc
                self.store.save_patch(self.sample_id, patch_record)
            if patch_record_changed or record_changed:
                changed = True
            refreshed_records.append(refreshed)
        if changed:
            self.store.save_patch_index_records(self.sample_id, refreshed_records)
        return refreshed_records, changed

    def _save_patch_retriever(self) -> None:
        cache_file, embeddings_file = self.store.patch_retriever_paths(self.sample_id)
        self.patch_retriever.save(cache_file, embeddings_file)

    def ingest_turn_with_patch_history(
        self,
        content: str,
        time: Optional[str] = None,
        session_id: Optional[int] = None,
        session_date_time: Optional[str] = None,
        session_summary: Optional[str] = None,
        turn_position: Optional[int] = None,
        turn_number: Optional[int] = None,
        dia_id: Optional[str] = None,
        speaker: Optional[str] = None,
    ) -> str:
        before_memories = copy.deepcopy(self.base_system.memories)
        note_id, evolve_trace = self.base_system.add_note_with_trace(content, time=time)
        after_memories = self.base_system.memories

        # Extract preference metadata from the newly created note
        new_note = after_memories.get(note_id)
        pref_domain = getattr(new_note, "pref_domain", "") if new_note else ""
        analyze_change_type = getattr(new_note, "change_type", "none") if new_note else "none"

        # In preference-aware mode, skip patch detection entirely for non-preference turns.
        # Task completions (translations, summaries, etc.) may cause neighbor updates but
        # should never produce patches — gate here before the expensive diff step.
        if self.config.preference_aware and not getattr(new_note, "is_preference", False):
            self._save_global_graph_cache(session_id, turn_position, turn_number, complete=False)
            return note_id

        diff_result = self.detect_patchable_change(before_memories, after_memories)
        if diff_result["patch_type"] == "additive_only":
            self._save_global_graph_cache(session_id, turn_position, turn_number, complete=False)
            return note_id

        detail_blocks = self.build_patch_detail_blocks(before_memories, after_memories, diff_result)
        patch_overall = self.summarize_patch_with_llm(
            trigger_turn={
                "session_id": session_id,
                "session_date_time": session_date_time,
                "session_summary": session_summary,
                "turn_position": turn_position,
                "turn_number": turn_number,
                "dia_id": dia_id,
                "speaker": speaker,
                "text": content,
            },
            evolve_trace=evolve_trace,
            detail_blocks=detail_blocks,
        )
        patch_record = self.build_patch_record(
            trigger_turn={
                "session_id": session_id,
                "session_date_time": session_date_time,
                "session_summary": session_summary,
                "turn_position": turn_position,
                "turn_number": turn_number,
                "dia_id": dia_id,
                "speaker": speaker,
                "text": content,
            },
            patch_type=diff_result["patch_type"],
            patch_overall=patch_overall,
            diff_result=diff_result,
            detail_blocks=detail_blocks,
            evolve_trace=evolve_trace,
            pref_domain=pref_domain,
            analyze_change_type=analyze_change_type,
        )
        self.commit_patch_if_needed(patch_record)
        self._save_global_graph_cache(session_id, turn_position, turn_number, complete=False)
        return note_id

    def detect_patchable_change(self, before_memories: Dict[str, Any], after_memories: Dict[str, Any]) -> Dict[str, Any]:
        created_note_ids = []
        updated_note_ids = []
        updated_fields: Dict[str, List[str]] = {}
        patch_type = "additive_only"

        for note_id, after_note in after_memories.items():
            if note_id not in before_memories:
                created_note_ids.append(note_id)
                continue
            before_note = before_memories[note_id]
            fields = []
            for field in ("content", "context", "keywords", "tags"):
                if getattr(before_note, field) != getattr(after_note, field):
                    fields.append(field)
            if list(getattr(before_note, "links", [])) != list(getattr(after_note, "links", [])):
                fields.append("links")
            if fields:
                updated_note_ids.append(note_id)
                updated_fields[note_id] = fields

        for note_id in updated_note_ids:
            fields = updated_fields[note_id]
            if any(field in ("content", "context", "keywords", "tags") for field in fields):
                patch_type = "overwrite_update"
                break
            if "links" in fields:
                before_links = list(getattr(before_memories[note_id], "links", []))
                after_links = list(getattr(after_memories[note_id], "links", []))
                if before_links and before_links != after_links:
                    patch_type = "link_rewrite_update"
                    break

        return {
            "patch_type": patch_type,
            "created_note_ids": created_note_ids,
            "updated_note_ids": updated_note_ids,
            "updated_fields": updated_fields,
        }

    def build_patch_detail_blocks(self, before_memories: Dict[str, Any], after_memories: Dict[str, Any], diff_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        blocks = []
        for note_id in diff_result["updated_note_ids"]:
            before_note = before_memories[note_id]
            after_note = after_memories[note_id]
            before_payload = self._note_payload(before_note)
            after_payload = self._note_payload(after_note)
            blocks.append({
                "note_id": note_id,
                "changed_fields": diff_result["updated_fields"][note_id],
                "before": before_payload,
                "after": after_payload,
                "link_change_summary": self._link_change_summary(
                    list(getattr(before_note, "links", [])),
                    list(getattr(after_note, "links", [])),
                ),
            })
        return blocks

    def summarize_patch_with_llm(self, trigger_turn: Dict[str, Any], evolve_trace: Dict[str, Any], detail_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        session_metadata = {
            "session_id": trigger_turn.get("session_id"),
            "session_date_time": trigger_turn.get("session_date_time"),
            "session_summary": trigger_turn.get("session_summary"),
            "turn_position": trigger_turn.get("turn_position"),
            "turn_number": trigger_turn.get("turn_number"),
            "dia_id": trigger_turn.get("dia_id"),
            "speaker": trigger_turn.get("speaker"),
        }
        _summarization_prompt = PATCH_SUMMARIZATION_PREF_PROMPT if self.config.preference_aware else PATCH_SUMMARIZATION_PROMPT
        prompt = _summarization_prompt.format(
            trigger_turn=json.dumps(trigger_turn, ensure_ascii=False, indent=2),
            session_metadata=json.dumps(session_metadata, ensure_ascii=False, indent=2),
            evolve_trace=json.dumps(self._compact_trace(evolve_trace), ensure_ascii=False, indent=2),
            detail_blocks=json.dumps(self._compact_detail_blocks_for_summary(detail_blocks), ensure_ascii=False, indent=2),
        )
        response = self.patch_llm.llm.get_completion(prompt)
        return self._parse_patch_summary_response(response)

    def build_patch_record(self, trigger_turn: Dict[str, Any], patch_type: str, patch_overall: Dict[str, Any], diff_result: Dict[str, Any], detail_blocks: List[Dict[str, Any]], evolve_trace: Dict[str, Any], pref_domain: str = "", analyze_change_type: str = "none") -> Dict[str, Any]:
        self._patch_counter += 1
        patch_id = f"patch_{self._patch_counter:06d}"
        patch_record = {
            "patch_id": patch_id,
            "sample_id": self.sample_id,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "patch_type": patch_type,
            "trigger_turn": trigger_turn,
            "patch_overall": patch_overall,
            "affected_notes": diff_result["updated_note_ids"],
            "difference_to_previous_state": self._difference_summaries(detail_blocks),
            "changed_nodes": detail_blocks,
            "changed_links": [
                {
                    "note_id": block["note_id"],
                    "before_links": block["before"]["links"],
                    "after_links": block["after"]["links"],
                    "link_change_summary": block["link_change_summary"],
                }
                for block in detail_blocks if "links" in block["changed_fields"]
            ],
            "evolve_trace": self._compact_trace(evolve_trace),
        }
        patch_record["pref_domain"] = pref_domain or ""
        patch_record["revoke"] = bool(patch_overall.get("revoke", False))
        # Use patch summarization's change_type if it detected something; fall back to ANALYZE stage detection
        summarization_ct = patch_overall.get("change_type", "none") or "none"
        patch_record["change_type"] = summarization_ct if summarization_ct != "none" else (analyze_change_type or "none")
        patch_record["index_document"] = self.build_patch_index_document(patch_record)
        return patch_record

    def build_patch_index_document(self, patch_record: Dict[str, Any]) -> str:
        """Build the SBERT-indexed document for this patch.

        Ordering principle: put the semantically richest, question-matchable content
        FIRST so it dominates the embedding.  The raw trigger turn text is moved to a
        short suffix — it contains specific first-person vocabulary that rarely matches
        generic query phrasing and dilutes the structured signal when placed first.
        """
        overall = patch_record["patch_overall"]
        pref_domain = patch_record.get("pref_domain", "")
        change_type = patch_record.get("change_type", "") or overall.get("change_type", "")

        # Extract before/after contexts from changed nodes (these are the key signals)
        before_contexts, after_contexts, link_summaries = [], [], []
        for block in patch_record["changed_nodes"]:
            bc = (block.get("before") or {}).get("context", "")
            ac = (block.get("after") or {}).get("context", "")
            if bc:
                before_contexts.append(bc)
            if ac:
                after_contexts.append(ac)
            ls = block.get("link_change_summary", "")
            if ls:
                link_summaries.append(ls)

        # ── Lead with structured preference-change description ────────────────
        parts = []
        trigger_turn = patch_record.get("trigger_turn") or {}
        session_id = trigger_turn.get("session_id", 0)
        turn_position = trigger_turn.get("turn_position", 0)
        turn_number = trigger_turn.get("turn_number", turn_position)
        parts.append(
            f"Temporal order: session {session_id}, turn index {turn_position}, turn number {turn_number}. Later turns supersede earlier turns in the same domain."
        )
        if pref_domain:
            parts.append(f"Preference domain: {pref_domain}.")
        if before_contexts:
            parts.append(f"Earlier state before this turn: {' '.join(before_contexts)}.")
        if after_contexts:
            parts.append(f"Later state after this turn: {' '.join(after_contexts)}.")
        if change_type and change_type != "none":
            parts.append(f"Change type: {change_type}.")
        overall_summary = overall.get("overall_summary", "")
        if overall_summary:
            parts.append(f"Summary: {overall_summary}.")
        if link_summaries:
            parts.append(f"Related changes: {' '.join(link_summaries)}.")

        # ── Append a short snippet of the raw trigger for keyword fallback ────
        trigger_text = trigger_turn.get("text", "")
        if trigger_text:
            parts.append(f"Trigger snippet: {trigger_text[:120]}")

        return " ".join(parts).strip()

    def commit_patch_if_needed(self, patch_record: Dict[str, Any]) -> None:
        overall = patch_record["patch_overall"]
        speaker = (patch_record.get("trigger_turn") or {}).get("speaker", "").lower()
        if speaker == "assistant":
            logger.debug(
                "skip_patch_assistant sample=%s session=%s turn=%s",
                self.sample_id,
                patch_record["trigger_turn"].get("session_id"),
                patch_record["trigger_turn"].get("turn_position"),
            )
            return
        if self.config.require_preference_change and not overall.get("should_commit_patch", False):
            logger.debug(
                "skip_patch sample=%s session=%s turn=%s type=%s decision=%s is_pref_change=%s",
                self.sample_id,
                patch_record["trigger_turn"].get("session_id"),
                patch_record["trigger_turn"].get("turn_position"),
                patch_record["patch_type"],
                overall.get("decision"),
                overall.get("is_preference_change", False),
            )
            return
        self.store.save_patch(self.sample_id, patch_record)
        session_id = patch_record["trigger_turn"].get("session_id") or 0
        turn_position = patch_record["trigger_turn"].get("turn_position") or 0
        self.store.append_patch_index_record(
            self.sample_id,
            {
                "patch_id": patch_record["patch_id"],
                "sample_id": self.sample_id,
                "session_id": session_id,
                "turn_position": turn_position,
                "temporal_order": f"{session_id:06d}_{turn_position:08d}",
                "patch_type": overall.get("patch_type", "STRENGTHEN_AND_UPDATE"),
                "revoke": patch_record.get("revoke", False),
                "change_type": patch_record.get("change_type", "none"),
                "pref_domain": patch_record.get("pref_domain", ""),
                "index_document": patch_record["index_document"],
            },
        )
        self.patch_retriever.add_documents([patch_record["index_document"]])
        self._save_patch_retriever()
        logger.info(
            "commit_patch sample=%s patch=%s type=%s session=%s turn=%s affected=%s pattern=%s signals=%s",
            self.sample_id,
            patch_record["patch_id"],
            patch_record["patch_type"],
            patch_record["trigger_turn"].get("session_id"),
            patch_record["trigger_turn"].get("turn_position"),
            len(patch_record.get("changed_nodes", [])),
            patch_record["patch_overall"].get("change_pattern", ""),
            ",".join(patch_record["patch_overall"].get("selection_signals", [])),
        )

    def retrieve_current_context(self, query: str, k: Optional[int] = None) -> str:
        return self.base_system.find_related_memories_raw(query, k=k or self.config.retrieve_k_current)

    def summarize_patch_inventory(self) -> Dict[str, Any]:
        patches = self.store.load_all_patches(self.sample_id)
        patch_type_counts: Dict[str, int] = {}
        session_counts: Dict[str, int] = {}
        affected_note_counts: List[int] = []
        for patch in patches:
            patch_type = patch.get("patch_type", "unknown")
            patch_type_counts[patch_type] = patch_type_counts.get(patch_type, 0) + 1
            session_id = str(patch.get("trigger_turn", {}).get("session_id", "unknown"))
            session_counts[session_id] = session_counts.get(session_id, 0) + 1
            affected_note_counts.append(len(patch.get("changed_nodes", [])))
        status = self.store.load_build_status(self.sample_id) or {}
        global_graph_dir = self.store.global_graph_dir(self.sample_id)
        return {
            "sample_id": self.sample_id,
            "patch_count": len(patches),
            "patch_type_counts": patch_type_counts,
            "session_patch_counts": session_counts,
            "avg_changed_nodes": (sum(affected_note_counts) / len(affected_note_counts)) if affected_note_counts else 0.0,
            "max_changed_nodes": max(affected_note_counts) if affected_note_counts else 0,
            "global_graph_dir": str(global_graph_dir),
            "global_graph_files": sorted(p.name for p in global_graph_dir.glob('*')),
            "global_graph_complete": bool(status.get('global_graph_complete', False)),
            "memory_count": status.get('memory_count', len(self.base_system.memories)),
            "last_session_id": status.get('last_session_id'),
            "last_turn_position": status.get('last_turn_position'),
            "last_turn_number": status.get('last_turn_number'),
        }

    @staticmethod
    def _normalize_score_array(scores: Any) -> Any:
        if np is None:
            return scores
        arr = np.asarray(scores, dtype=float)
        if arr.size == 0:
            return arr
        max_v = float(arr.max())
        min_v = float(arr.min())
        if max_v - min_v < 1e-8:
            return np.ones_like(arr) if max_v > 0 else np.zeros_like(arr)
        return (arr - min_v) / (max_v - min_v)

    def _embedding_scores(self, query: str, documents: List[str]) -> Any:
        if np is None or not documents:
            return np.array([]) if np is not None else []
        embeddings = self.patch_retriever.model.encode(documents)
        q_emb = self.patch_retriever.model.encode([query])[0]
        from sklearn.metrics.pairwise import cosine_similarity as _cos_sim
        return _cos_sim([q_emb], embeddings)[0]

    def _bm25_scores(self, query: str, documents: List[str]) -> Any:
        if np is None or BM25Okapi is None or not documents:
            return np.array([]) if np is not None else []
        tokenized_docs = [simple_tokenize(doc.lower()) for doc in documents]
        bm25 = BM25Okapi(tokenized_docs)
        return np.asarray(bm25.get_scores(simple_tokenize(query.lower())), dtype=float)

    def _combined_scores(self, query: str, documents: List[str], use_hybrid: bool, alpha: float) -> Any:
        if np is None or not documents:
            return np.array([]) if np is not None else []
        embed = self._embedding_scores(query, documents)
        if not use_hybrid:
            return np.asarray(embed, dtype=float)
        bm25 = self._bm25_scores(query, documents)
        if bm25.size == 0:
            return np.asarray(embed, dtype=float)
        embed_norm = self._normalize_score_array(embed)
        bm25_norm = self._normalize_score_array(bm25)
        return alpha * embed_norm + (1.0 - alpha) * bm25_norm

    @staticmethod
    def _category_hint(category: Optional[int]) -> str:
        if category == 1:
            return "multi-hop relation chain, connect entities and events"
        if category == 2:
            return "temporal reasoning, date, session order, before after"
        if category == 3:
            return "open-domain detail, broad semantic match"
        if category == 4:
            return "single-hop factual recall, direct entity and attribute match"
        if category == 5:
            return "adversarial mention check, contradiction, whether mentioned"
        return "general conversational memory retrieval"

    def _build_node_query(self, question: str, query_keywords: str, category: Optional[int], patches: List[Dict[str, Any]]) -> str:
        mode = self.config.patch_node_query_mode
        if mode == "question_only":
            return question
        if mode == "keywords_only":
            return query_keywords

        parts = [f"Question: {question}"]
        if query_keywords:
            parts.append(f"Retrieval keywords: {query_keywords}")
        parts.append(f"Category hint: {self._category_hint(category)}")

        if mode in ("expanded", "answer_style") and patches:
            summaries = []
            for patch in patches[:3]:
                overall = patch.get("patch_overall", {})
                trigger = (patch.get("trigger_turn") or {}).get("text", "")
                bits = []
                if overall.get("overall_summary"):
                    bits.append(overall["overall_summary"])
                if overall.get("change_pattern"):
                    bits.append(overall["change_pattern"])
                if trigger:
                    bits.append(trigger[:120])
                if bits:
                    summaries.append(" | ".join(bits))
            if summaries:
                parts.append("Candidate patch summaries: " + " || ".join(summaries))

        if mode == "answer_style":
            parts.append("Target answer style: short factual phrase grounded in the conversation evidence.")

        return "\n".join(parts)

    def _patch_item_documents(self, patch_record: Dict[str, Any]) -> List[Dict[str, Any]]:
        docs: List[Dict[str, Any]] = []
        trigger = patch_record.get("trigger_turn", {})
        prefix = (
            f"Patch {patch_record.get('patch_id', '')}. "
            f"Session {trigger.get('session_id', 0)} turn {trigger.get('turn_number', trigger.get('turn_position', 0))}. "
        )
        for idx, block in enumerate(patch_record.get("changed_nodes", [])):
            before_ctx = (block.get("before") or {}).get("context", "")
            after_ctx = (block.get("after") or {}).get("context", "")
            link_summary = block.get("link_change_summary", "")
            changed_fields = ", ".join(block.get("changed_fields", []))
            doc = prefix + f"Node {block.get('note_id','')} changed fields: {changed_fields}. Before: {before_ctx}. After: {after_ctx}."
            if link_summary:
                doc += f" Links: {link_summary}."
            docs.append({
                "item_type": "node",
                "item_id": block.get("note_id", f"node_{idx}"),
                "patch_id": patch_record.get("patch_id"),
                "document": doc,
                "before_context": before_ctx,
                "after_context": after_ctx,
                "link_change_summary": link_summary,
            })
        for idx, block in enumerate(patch_record.get("changed_links", [])):
            summary = block.get("link_change_summary", "")
            if not summary:
                summary = f"Links changed from {block.get('before_links', [])} to {block.get('after_links', [])}."
            doc = prefix + f"Link change for note {block.get('note_id','')}: {summary}"
            docs.append({
                "item_type": "link",
                "item_id": block.get("note_id", f"link_{idx}"),
                "patch_id": patch_record.get("patch_id"),
                "document": doc,
                "before_links": block.get("before_links", []),
                "after_links": block.get("after_links", []),
                "link_change_summary": summary,
            })
        return docs

    def _select_patch_items(self, question: str, query_keywords: str, category: Optional[int], patches: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        if not self.config.patch_node_rerank or not patches:
            return {}
        node_query = self._build_node_query(question, query_keywords, category, patches)
        docs = []
        for patch in patches:
            docs.extend(self._patch_item_documents(patch))
        if not docs or np is None:
            return {}
        documents = [d['document'] for d in docs]
        scores = self._combined_scores(
            node_query,
            documents,
            use_hybrid=self.config.patch_hybrid_node_rerank,
            alpha=self.config.patch_hybrid_alpha,
        )
        ranked = sorted(
            [{**doc, 'score': float(scores[idx])} for idx, doc in enumerate(docs)],
            key=lambda item: item['score'],
            reverse=True,
        )[: max(1, self.config.patch_node_top_k)]
        selected: Dict[str, List[Dict[str, Any]]] = {}
        for item in ranked:
            selected.setdefault(item['patch_id'], []).append(item)
        return selected

    def _llm_filter_patches(
        self, query: str, candidate_records: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Use the LLM to keep only patches that are topically relevant to the query.

        Each candidate_record is an index record (not a full patch) that has at minimum
        `patch_id`, `index_document`, `change_type`, `pref_domain`.
        Returns a filtered subset in the original order.
        """
        if not candidate_records:
            return []

        candidate_lines = []
        for i, rec in enumerate(candidate_records, 1):
            doc = rec.get("index_document", "")
            # Trim to keep the prompt small — first 200 chars capture domain + before/after
            candidate_lines.append(f"{i}. {doc[:200]}")

        prompt = PATCH_RELEVANCE_FILTER_PROMPT.format(
            query=query,
            candidates="\n".join(candidate_lines),
        )
        try:
            raw = self.base_system.llm.get_completion(prompt, temperature=0.0)
        except Exception as exc:
            if is_nonrecoverable_llm_error(exc):
                raise
            logger.warning("llm_patch_filter_error %s — keeping all candidates", exc)
            return candidate_records

        raw = raw.strip()
        if not raw or raw.upper() == "NONE":
            logger.debug("llm_patch_filter: no relevant patches for query=%s", query[:60])
            return []

        import re
        keep_indices = set()
        for tok in re.split(r"[,\s]+", raw):
            tok = tok.strip().rstrip(".")
            if tok.isdigit():
                idx = int(tok)
                if 1 <= idx <= len(candidate_records):
                    keep_indices.add(idx - 1)  # convert to 0-based

        filtered = [rec for i, rec in enumerate(candidate_records) if i in keep_indices]
        logger.debug(
            "llm_patch_filter: %d/%d candidates kept for query=%s",
            len(filtered), len(candidate_records), query[:60],
        )
        return filtered

    def retrieve_relevant_patches(self, query: str, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        index_records = self.store.load_patch_index_records(self.sample_id)
        if not index_records:
            return []

        _top_k = top_k or self.config.patch_top_k
        min_sim = self.config.min_patch_similarity
        _sbert_k = self.config.llm_patch_filter_candidates if self.config.llm_patch_filter else _top_k
        _sbert_min_sim = 0.15 if self.config.llm_patch_filter else min_sim

        documents = [record.get('index_document', '') for record in index_records]
        if np is not None and documents:
            scores = self._combined_scores(
                query,
                documents,
                use_hybrid=self.config.patch_hybrid_retrieval,
                alpha=self.config.patch_hybrid_alpha,
            )
            top_indices = np.argsort(scores)[-_sbert_k:][::-1]
            if _sbert_min_sim > 0.0 and not self.config.patch_hybrid_retrieval:
                candidate_indices = [int(i) for i in top_indices if scores[i] >= _sbert_min_sim]
            else:
                candidate_indices = [int(i) for i in top_indices]
            if (
                not self.config.llm_patch_filter
                and not self.config.patch_hybrid_retrieval
                and candidate_indices
            ):
                candidate_scores = [float(scores[i]) for i in candidate_indices]
                max_sim = max(candidate_scores)
                sim_spread = max_sim - min(candidate_scores)
                if max_sim < 0.55 and sim_spread < 0.05:
                    candidate_indices = []
            indices = candidate_indices
        else:
            indices = self.patch_retriever.search(query, _sbert_k)
            scores = None

        candidate_index_records = []
        for idx in indices:
            if idx >= len(index_records):
                continue
            record = dict(index_records[idx])
            patch_type = record.get("patch_type", "")
            is_revoke = record.get("revoke", False) or patch_type == "REVOKE_PREFERENCE"
            if self.config.exclude_revoke_patches and is_revoke:
                continue
            if self.config.exclude_add_patches and patch_type == "ADD_PREFERENCE":
                continue
            record["_emb_idx"] = idx
            if scores is not None:
                record["_retrieval_score"] = float(scores[idx])
            candidate_index_records.append(record)

        if self.config.llm_patch_filter and candidate_index_records:
            candidate_index_records = self._llm_filter_patches(query, candidate_index_records)
            if min_sim > 0.0 and np is not None and self.patch_retriever.embeddings is not None and not self.config.patch_hybrid_retrieval:
                q_emb_cached = self.patch_retriever.model.encode([query])[0]
                from sklearn.metrics.pairwise import cosine_similarity as _cos_sim2
                sims2 = _cos_sim2([q_emb_cached], self.patch_retriever.embeddings)[0]
                candidate_index_records = [
                    r for r in candidate_index_records
                    if sims2[r["_emb_idx"]] >= min_sim
                ]

        patches = []
        for record in candidate_index_records[:_top_k]:
            patch_id = record["patch_id"]
            patch_record = self.store.load_patch(self.sample_id, patch_id)
            if patch_record:
                patch_record["_temporal_order"] = record.get(
                    "temporal_order",
                    f"{record.get('session_id', 0):06d}_{record.get('turn_position', 0):08d}",
                )
                patch_record["_retrieval_score"] = record.get("_retrieval_score")
                patches.append(patch_record)

        patches.sort(key=lambda p: p.get("_temporal_order", ""))
        return patches

    def format_patch_for_context(self, patch_record: Dict[str, Any], selected_items: Optional[List[Dict[str, Any]]] = None) -> str:
        overall = patch_record["patch_overall"]
        changed_lines = []
        if selected_items:
            for item in selected_items:
                changed_lines.append(
                    f"- Selected {item.get('item_type')} {item.get('item_id')} score={item.get('score', 0.0):.4f}: {item.get('document', '')}"
                )
        else:
            for block in patch_record["changed_nodes"]:
                changed_lines.append(f"- Note {block['note_id']} before context: {block['before']['context']}")
                changed_lines.append(f"  after context: {block['after']['context']}")
                if "tags" in block["changed_fields"]:
                    changed_lines.append(
                        f"  tags: {', '.join(block['before']['tags'])} -> {', '.join(block['after']['tags'])}"
                    )
                if block.get("link_change_summary"):
                    changed_lines.append(f"  links: {block['link_change_summary']}")
        patch_type = overall.get("patch_type", patch_record.get("patch_type", "STRENGTHEN_AND_UPDATE"))
        change_type = patch_record.get("change_type", "") or overall.get("change_type", "")
        change_type_str = f" | change_type: {change_type}" if change_type and change_type != "none" else ""
        revoke_str = " | REVOKE: YES" if patch_record.get("revoke", False) else ""
        selected_header = "Selected evidence:" if selected_items else "Key changes:"
        return (
            f"Session {patch_record['trigger_turn'].get('session_id')} "
            f"TurnIndex {patch_record['trigger_turn'].get('turn_position')} "
            f"TurnNumber {patch_record['trigger_turn'].get('turn_number')} "
            f"({patch_record['trigger_turn'].get('dia_id')}) "
            f"[preference change — type: {patch_type}{change_type_str}{revoke_str}]:\n"
            f"Temporal order rule: this patch happened at session {patch_record['trigger_turn'].get('session_id')}, turn {patch_record['trigger_turn'].get('turn_number')}. If another patch for the same domain has a larger session/turn number, the later patch wins.\n"
            f"Trigger: {patch_record['trigger_turn'].get('text', '')}\n"
            f"Overall summary: {overall.get('overall_summary', '')}\n"
            f"Reasoning: {overall.get('update_reasoning', '')}\n"
            f"Pattern: {overall.get('change_pattern', '')}\n"
            f"Signals: {', '.join(overall.get('selection_signals', []))}\n"
            f"{selected_header}\n" + "\n".join(changed_lines)
        )

    def build_augmented_context(self, current_context: str, patches: List[Dict[str, Any]], selected_patch_items: Optional[Dict[str, List[Dict[str, Any]]]] = None) -> str:
        if not patches:
            return current_context
        selected_patch_items = selected_patch_items or {}
        patch_blocks = "\n\n".join(
            f"[Historical Patch {idx + 1}]\n{self.format_patch_for_context(patch, selected_patch_items.get(patch.get('patch_id')))}"
            for idx, patch in enumerate(patches)
        )
        _context_instruction = PATCH_CONTEXT_PREF_INSTRUCTION if self.config.preference_aware else PATCH_CONTEXT_INSTRUCTION
        return (
            f"[Current Global Memory Evidence]\n{current_context}\n\n"
            f"[Relevant Historical Patches]\n{patch_blocks}\n\n"
            f"{_context_instruction}"
        )

    def _format_patch_summary_for_gating(self, patch_record: Dict[str, Any]) -> str:
        overall = patch_record.get("patch_overall", {})
        trigger_turn = patch_record.get("trigger_turn", {})
        signals = ", ".join(overall.get("selection_signals", [])) or "NONE"
        return (
            f"PatchID: {patch_record.get('patch_id', '')}\n"
            f"Session: {trigger_turn.get('session_id')}\n"
            f"Turn: {trigger_turn.get('turn_number')}\n"
            f"Summary: {overall.get('overall_summary', '')}\n"
            f"Reasoning: {overall.get('update_reasoning', '')}\n"
            f"Pattern: {overall.get('change_pattern', '')}\n"
            f"Signals: {signals}\n"
            f"Affected notes: {len(patch_record.get('affected_notes', []))}"
        )

    def _build_patch_summary_context(self, patches: List[Dict[str, Any]]) -> str:
        if not patches:
            return "[Historical Patch Summaries]\nNONE"
        summary_blocks = "\n\n".join(
            self._format_patch_summary_for_gating(patch) for patch in patches
        )
        return f"[Historical Patch Summaries]\n{summary_blocks}"

    @staticmethod
    def _parse_patch_gating_response(response: str) -> Dict[str, Any]:
        result = {
            "draft_answer": response.strip(),
            "need_patch_detail": False,
            "selected_patch_ids": [],
            "reason": "",
        }
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "DRAFT_ANSWER":
                result["draft_answer"] = value
            elif key == "NEED_PATCH_DETAIL":
                result["need_patch_detail"] = value.upper().startswith("Y")
            elif key == "SELECTED_PATCH_IDS":
                if value.upper() == "NONE":
                    result["selected_patch_ids"] = []
                else:
                    result["selected_patch_ids"] = [item.strip() for item in value.split(",") if item.strip()][:2]
            elif key == "REASON":
                result["reason"] = value
        return result

    @staticmethod
    def _parse_patch_revision_response(response: str) -> Dict[str, str]:
        result = {"final_answer": response.strip(), "reason": ""}
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "FINAL_ANSWER":
                result["final_answer"] = value
            elif key == "REASON":
                result["reason"] = value
        return result

    @staticmethod
    def _build_answer_instruction(question: str, category: int, answer: str) -> str:
        return build_locomo_answer_instruction(question, category, answer)

    @staticmethod
    def _build_answer_prompt(question: str, category: int, answer: str, context: str) -> str:
        return build_locomo_answer_prompt(question, category, answer, context)

    def _filter_selected_patches(self, selected_patch_ids: List[str], retrieved_patches: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        patch_map = {patch.get("patch_id"): patch for patch in retrieved_patches}
        selected = []
        for patch_id in selected_patch_ids:
            patch = patch_map.get(patch_id)
            if patch is not None:
                selected.append(patch)
        return selected

    def answer_with_patch_history_gated(self, question: str, category: int, answer: str, temperature_c5: float = 0.5) -> tuple:
        query = self.generate_query_llm(question)
        current_context = self.retrieve_current_context(query, self.config.retrieve_k_current)
        patches = self.retrieve_relevant_patches(query, self.config.patch_top_k)
        selected_patch_items = self._select_patch_items(question, query, category, patches)
        patch_summaries = self._build_patch_summary_context(patches)
        _context_instruction = PATCH_CONTEXT_PREF_INSTRUCTION if self.config.preference_aware else PATCH_CONTEXT_INSTRUCTION
        gating_context = (
            f"[Current Global Memory Evidence]\n{current_context}\n\n"
            f"{patch_summaries}\n\n"
            f"{_context_instruction}"
        )
        answer_instruction = self._build_answer_instruction(question, category, answer)
        gating_prompt = PATCH_GATING_PROMPT.format(
            question=question,
            answer_instruction=answer_instruction,
            current_context=current_context,
            patch_summaries=patch_summaries,
        )
        answer_temperature = locomo_answer_temperature(category, temperature_c5)
        gating_response = self.base_system.llm_controller.llm.get_completion(gating_prompt, temperature=answer_temperature)
        gating_result = self._parse_patch_gating_response(gating_response)

        if not gating_result["need_patch_detail"]:
            metadata = {
                "patch_usage": "gated",
                "need_patch_detail": False,
                "selected_patch_ids": [],
                "gating_reason": gating_result["reason"],
                "used_patch_detail": False,
                "selected_patch_items": selected_patch_items,
                "gating_prompt": gating_prompt,
                "gating_context": gating_context,
            }
            return gating_result["draft_answer"], gating_prompt, gating_context, metadata

        selected_patches = self._filter_selected_patches(gating_result["selected_patch_ids"], patches)
        if not selected_patches:
            metadata = {
                "patch_usage": "gated",
                "need_patch_detail": True,
                "selected_patch_ids": gating_result["selected_patch_ids"],
                "gating_reason": gating_result["reason"],
                "used_patch_detail": False,
                "selected_patch_items": selected_patch_items,
                "gating_prompt": gating_prompt,
                "gating_context": gating_context,
            }
            return gating_result["draft_answer"], gating_prompt, gating_context, metadata

        detail_selected_items = {patch.get('patch_id'): selected_patch_items.get(patch.get('patch_id'), []) for patch in selected_patches}
        detail_context = self.build_augmented_context(current_context, selected_patches, detail_selected_items)
        detail_prompt = PATCH_DETAIL_REVISION_PROMPT.format(
            question=question,
            answer_instruction=answer_instruction,
            current_context=current_context,
            patch_details="\n\n".join(self.format_patch_for_context(patch, detail_selected_items.get(patch.get('patch_id'))) for patch in selected_patches),
            draft_answer=gating_result["draft_answer"],
        )
        detail_response = self.base_system.llm_controller.llm.get_completion(detail_prompt, temperature=answer_temperature)
        detail_result = self._parse_patch_revision_response(detail_response)
        metadata = {
            "patch_usage": "gated",
            "need_patch_detail": True,
            "selected_patch_ids": [patch.get("patch_id") for patch in selected_patches],
            "gating_reason": gating_result["reason"],
            "revision_reason": detail_result["reason"],
            "used_patch_detail": True,
            "selected_patch_items": detail_selected_items,
            "gating_prompt": gating_prompt,
            "gating_context": gating_context,
        }
        return detail_result["final_answer"], detail_prompt, detail_context, metadata

    def answer_with_patch_history(self, question: str, category: int, answer: str, temperature_c5: float = 0.5) -> tuple:
        query = self.generate_query_llm(question)
        current_context = self.retrieve_current_context(query, self.config.retrieve_k_current)
        patches = self.retrieve_relevant_patches(query, self.config.patch_top_k)
        selected_patch_items = self._select_patch_items(question, query, category, patches)
        context = self.build_augmented_context(current_context, patches, selected_patch_items)
        prompt = self._build_answer_prompt(question, category, answer, context)
        response = self.base_system.llm_controller.llm.get_completion(
            prompt,
            temperature=locomo_answer_temperature(category, temperature_c5),
        )
        metadata = {
            "patch_usage": "always",
            "need_patch_detail": bool(patches),
            "selected_patch_ids": [patch.get("patch_id") for patch in patches],
            "selected_patch_items": selected_patch_items,
            "gating_reason": "always_on_patch_context",
            "used_patch_detail": bool(patches),
        }
        return response, prompt, context, metadata

    def generate_query_llm(self, question: str) -> str:
        prompt = f"""Given the following question, generate several keywords separated by commas.

Question: {question}

Keywords:"""
        return self.base_system.llm_controller.llm.get_completion(prompt, temperature=0.7).strip()

    @staticmethod
    def _note_payload(note_obj: Any) -> Dict[str, Any]:
        return {
            "content": note_obj.content,
            "context": note_obj.context,
            "keywords": list(note_obj.keywords),
            "tags": list(note_obj.tags),
            "links": list(note_obj.links),
            "retrieval_document": (
                "content:" + note_obj.content +
                " context:" + note_obj.context +
                " keywords: " + ", ".join(note_obj.keywords) +
                " tags: " + ", ".join(note_obj.tags)
            ),
        }

    @staticmethod
    def _link_change_summary(before_links: List[int], after_links: List[int]) -> str:
        if before_links == after_links:
            return ""
        return f"Links changed from {before_links} to {after_links}."

    @staticmethod
    def _difference_summaries(detail_blocks: List[Dict[str, Any]]) -> List[str]:
        summaries = []
        for block in detail_blocks:
            if "context" in block["changed_fields"]:
                summaries.append(
                    f"Updated {block['note_id']} context from '{block['before']['context']}' to '{block['after']['context']}'."
                )
            if "tags" in block["changed_fields"]:
                summaries.append(
                    f"Updated {block['note_id']} tags from {block['before']['tags']} to {block['after']['tags']}."
                )
            if "links" in block["changed_fields"]:
                summaries.append(
                    f"Rewrote {block['note_id']} links from {block['before']['links']} to {block['after']['links']}."
                )
        return summaries

    @staticmethod
    def _to_jsonable(value: Any) -> Any:
        if np is not None:
            if isinstance(value, np.ndarray):
                return value.tolist()
            if isinstance(value, np.generic):
                return value.item()
        if isinstance(value, dict):
            return {str(k): PatchAugmentedMemorySystem._to_jsonable(v) for k, v in value.items()}
        if isinstance(value, list):
            return [PatchAugmentedMemorySystem._to_jsonable(v) for v in value]
        if isinstance(value, tuple):
            return [PatchAugmentedMemorySystem._to_jsonable(v) for v in value]
        return value

    @staticmethod
    def _compact_detail_blocks_for_summary(detail_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        compact = []
        for block in detail_blocks:
            compact.append({
                "note_id": block["note_id"],
                "changed_fields": block["changed_fields"],
                "before_context": block["before"]["context"],
                "after_context": block["after"]["context"],
                "before_tags": block["before"]["tags"],
                "after_tags": block["after"]["tags"],
                "link_change_summary": block.get("link_change_summary", ""),
            })
        return compact

    @staticmethod
    def _compact_trace(trace: Dict[str, Any]) -> Dict[str, Any]:
        compact = {
            "neighbor_indices": trace.get("neighbor_indices", []),
            "decision_parsed": trace.get("decision_parsed"),
            "strengthen_parsed": trace.get("strengthen_parsed"),
            "neighbor_updates_parsed": trace.get("neighbor_updates_parsed"),
            "decision_response": trace.get("decision_response"),
            "strengthen_response": trace.get("strengthen_response"),
            "update_response": trace.get("update_response"),
        }
        return PatchAugmentedMemorySystem._to_jsonable(compact)

    @staticmethod
    def _parse_patch_summary_response(response: str) -> Dict[str, Any]:
        _valid_change_types = {
            "same_object_flip", "object_replacement", "conditional_preference",
            "attribute_swap", "temporal_validity", "ask_to_forget",
            "new_preference", "strengthen", "none",
        }
        result = {
            "is_preference_change": False,
            "should_commit_patch": False,
            "patch_type": "STRENGTHEN_AND_UPDATE",
            "revoke": False,
            "change_type": "none",
            "decision": "UNKNOWN",
            "overall_summary": "",
            "update_reasoning": "",
            "change_pattern": "",
            "selection_signals": [],
            "task_pattern_summary": "",
        }
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            if key == "IS_PREFERENCE_CHANGE":
                result["is_preference_change"] = value.upper().startswith("Y")
            elif key == "SHOULD_COMMIT_PATCH":
                result["should_commit_patch"] = value.upper().startswith("Y")
            elif key == "PATCH_TYPE":
                result["patch_type"] = value.upper()
            elif key == "REVOKE":
                result["revoke"] = value.upper().startswith("Y")
            elif key == "CHANGE_TYPE":
                ct = value.lower().strip("|").strip().split()[0] if value.strip() else "none"
                result["change_type"] = ct if ct in _valid_change_types else "none"
            elif key == "DECISION":
                result["decision"] = value
            elif key == "OVERALL_SUMMARY":
                result["overall_summary"] = value
            elif key == "UPDATE_REASONING":
                result["update_reasoning"] = value
            elif key == "CHANGE_PATTERN":
                result["change_pattern"] = value
            elif key == "SELECTION_SIGNALS":
                result["selection_signals"] = [item.strip() for item in value.split(",") if item.strip()][:5]
            elif key == "TASK_PATTERN_SUMMARY":
                result["task_pattern_summary"] = value
        # Enforce: non-preference changes are never committed
        if not result["is_preference_change"]:
            result["should_commit_patch"] = False
        return result
