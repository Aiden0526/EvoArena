"""
Evaluate A-MEM robust mode on the PersonaMem benchmark.

This script adapts the retrieval-based robust memory flow from
`test_advanced_robust.py` to Persona-release's CSV benchmark and chat-history
JSON format. It produces a results CSV compatible with Persona-release's
evaluation artifacts and a companion metrics text file with OOD breakdowns.

Example:
    python test_persona_robust.py \
        --benchmark_file data/Persona-release/benchmark_v34/text/benchmark_49p_ood_v34.csv \
        --model gpt-4o-mini \
        --backend openai \
        --size 32k
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import logging
import os
import pickle
import random
import re
import subprocess
import sys
import time
from collections import OrderedDict, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm

from llm_text_parsers import parse_keywords_response
from memory_layer_robust import (
    LLMUsageRecorder,
    RobustAgenticMemorySystem,
    RobustLLMController,
    is_nonrecoverable_llm_error,
    llm_usage_recorder,
    llm_usage_stage,
)


logger = logging.getLogger("persona_robust")
csv.field_size_limit(sys.maxsize)


def make_progress(iterable, desc: str, total: Optional[int] = None, leave: bool = True):
    return tqdm(
        iterable,
        desc=desc,
        total=total,
        dynamic_ncols=True,
        file=sys.stdout,
        mininterval=0.5,
        leave=leave,
    )


def normalize_openai_env() -> None:
    if os.getenv("OPENAI_API_KEY") or not os.getenv("OPENAI_KEY"):
        return
    os.environ["OPENAI_API_KEY"] = os.environ["OPENAI_KEY"]


def setup_logger(log_file: Optional[Path] = None) -> logging.Logger:
    eval_logger = logging.getLogger("persona_robust_eval")
    eval_logger.setLevel(logging.INFO)
    eval_logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    eval_logger.addHandler(console_handler)

    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        eval_logger.addHandler(file_handler)

    return eval_logger


def stable_int_seed(*parts: Any) -> int:
    payload = "||".join(str(part) for part in parts)
    return int(hashlib.md5(payload.encode("utf-8")).hexdigest()[:8], 16)


def sanitize_filename_part(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", str(value).strip())
    return re.sub(r"-+", "-", text).strip("-")


def parse_persona_ids(persona_ids_arg: Optional[str]) -> List[str]:
    if not persona_ids_arg:
        return []
    return [part.strip() for part in persona_ids_arg.split(",") if part.strip()]


def resolve_persona_root(benchmark_file: Path, explicit_root: Optional[str]) -> Path:
    if explicit_root:
        return Path(explicit_root).resolve()

    for candidate in [benchmark_file.parent, *benchmark_file.parents]:
        if (candidate / "EVALUATION_GUIDE.md").exists():
            return candidate
        # PersonaMem benchmark layout: <root>/benchmark_v34/text/*.csv and <root>/data/...
        if (candidate / "benchmark_v34").exists() and (candidate / "data").exists():
            return candidate

    # Fallback: if the benchmark is nested under benchmark_v34/text, lift to the dataset root.
    if benchmark_file.parent.name == "text" and benchmark_file.parent.parent.name.startswith("benchmark_"):
        candidate = benchmark_file.parent.parent.parent.resolve()
        if (candidate / "data").exists():
            return candidate

    return benchmark_file.parent.resolve()


def load_benchmark_rows(
    benchmark_file: Path,
    persona_ids: Optional[Iterable[str]] = None,
    max_items: Optional[int] = None,
) -> Tuple[List[str], List[Dict[str, str]]]:
    rows: List[Dict[str, str]] = []
    persona_id_set = {str(pid) for pid in (persona_ids or [])}

    with benchmark_file.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            if persona_id_set and str(row.get("persona_id", "")) not in persona_id_set:
                continue
            rows.append(row)
            if max_items is not None and len(rows) >= max_items:
                break

    return fieldnames, rows


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


def parse_user_query(user_query_raw: str) -> Dict[str, str]:
    try:
        user_query = json.loads(user_query_raw)
    except (json.JSONDecodeError, TypeError):
        try:
            user_query = ast.literal_eval(user_query_raw)
        except (ValueError, SyntaxError):
            user_query = {"role": "user", "content": str(user_query_raw).strip('"').strip("'")}

    if not isinstance(user_query, dict):
        user_query = {"role": "user", "content": str(user_query)}

    user_query.setdefault("role", "user")
    user_query["content"] = str(user_query.get("content", "")).strip()
    if user_query["content"]:
        user_query["content"] += (
            " Please recall my related preferences from our conversation history "
            "to give personalized responses."
        )
    return user_query


def parse_incorrect_answers(raw_value: str) -> List[str]:
    if not raw_value:
        return []
    try:
        values = json.loads(raw_value)
        return [str(value) for value in values]
    except json.JSONDecodeError:
        return []


def create_mcq_options(
    correct_answer: str,
    incorrect_answers: List[str],
    seed: int,
) -> Tuple[str, Dict[str, str], str]:
    rng = random.Random(seed)
    options = [correct_answer] + incorrect_answers
    rng.shuffle(options)

    option_mapping: Dict[str, str] = {}
    option_lines = []
    correct_letter = ""
    for idx, option in enumerate(options):
        letter = chr(65 + idx)
        option_mapping[letter] = option
        option_lines.append(f"{letter}. {option}")
        if option == correct_answer:
            correct_letter = letter

    instruction = (
        "Please choose the best answer from the following options:\n\n"
        + "\n".join(option_lines)
        + "\n\nThink briefly about which answer best fits the user's preferences and "
          "conversation history. Then give your final answer as 'Final Answer: [Letter]'."
    )
    return instruction, option_mapping, correct_letter


def extract_final_answer(response: str) -> str:
    if not response:
        return ""

    patterns = [
        r"\$\\boxed\{([A-Z])\}\$",
        r"\\boxed\{([A-Z])\}",
        r"Final Answer:\s*([A-Z])",
        r"final answer:\s*([A-Z])",
        r"Answer:\s*([A-Z])",
        r"answer:\s*([A-Z])",
        r"final answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"final answer is\s*([A-Z])",
        r"the answer is\s*\$?\\boxed\{([A-Z])\}\$?",
        r"the answer is\s*([A-Z])",
        r"\b([A-Z])\.\s*$",
        r"^\s*([A-D])\s*$",
    ]
    for pattern in patterns:
        match = re.search(pattern, response, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).upper()
    return ""


def check_mcq_correctness(predicted_answer: str, correct_answer: str, option_mapping: Dict[str, str]) -> bool:
    if not predicted_answer:
        return False
    return option_mapping.get(predicted_answer.upper(), "") == correct_answer


def load_chat_history_messages(chat_history_path: Path) -> List[Dict[str, Any]]:
    with chat_history_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if isinstance(data.get("chat_history"), list):
            return data["chat_history"]
        if isinstance(data.get("conversations"), list):
            return data["conversations"]
        for value in data.values():
            if isinstance(value, dict) and isinstance(value.get("conversations"), list):
                return value["conversations"]
            if isinstance(value, list):
                return value
    return []


def serialize_chat_message(message: Dict[str, Any], index: int) -> str:
    role = str(message.get("role", "unknown")).upper()
    content = str(message.get("content", "")).strip()
    return f"[{index:04d}] {role}: {content}"


def build_cache_key(chat_history_path: Path, include_system_messages: bool, persona_id: str = "") -> str:
    payload = (
        f"{chat_history_path.resolve()}|"
        f"include_system={include_system_messages}|"
        f"persona_id={persona_id}"
    )
    return hashlib.md5(payload.encode("utf-8")).hexdigest()


def build_status_path(cache_dir: Path, cache_key: str) -> Path:
    return cache_dir / f"build_status_{cache_key}.json"


def load_build_status(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def write_persona_build_status(
    status_path: Path,
    cache_key: str,
    next_message_index: int,
    total_messages: int,
    complete: bool,
    **extra_status: Any,
) -> None:
    status = {
        "cache_key": cache_key,
        "next_message_index": next_message_index,
        "total_messages": total_messages,
        "complete": complete,
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }
    status.update(extra_status)
    status_path.write_text(
        json.dumps(status, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def save_persona_build_checkpoint(
    agent: "PersonaRobustAgent",
    memory_cache_file: Path,
    retriever_cache_file: Path,
    retriever_cache_embeddings_file: Path,
    status_path: Path,
    cache_key: str,
    next_message_index: int,
    total_messages: int,
    complete: bool,
) -> None:
    agent.save_cached_state(memory_cache_file, retriever_cache_file, retriever_cache_embeddings_file)
    write_persona_build_status(
        status_path,
        cache_key,
        next_message_index,
        total_messages,
        complete,
    )


def mark_persona_build_failed(
    status_path: Path,
    cache_key: str,
    failed_message_index: int,
    total_messages: int,
    exc: BaseException,
) -> None:
    # Do not save the in-memory agent here; it may have been partially mutated
    # by the failing LLM call. The previous cache remains the clean resume point.
    write_persona_build_status(
        status_path,
        cache_key,
        failed_message_index,
        total_messages,
        False,
        fatal_error=True,
        fatal_error_stage="memory_build",
        failed_message_index=failed_message_index,
        resume_message_index=failed_message_index,
        fatal_error_type=f"{exc.__class__.__module__}.{exc.__class__.__name__}",
        fatal_error_message=str(exc),
    )


def load_existing_output_rows(output_path: Path) -> tuple[List[Dict[str, str]], List[str]]:
    if not output_path.exists():
        return [], []
    with output_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows, list(reader.fieldnames or [])


class PersonaRobustAgent:
    def __init__(
        self,
        model: str,
        backend: str,
        retrieve_k: int,
        answer_temperature: float,
        sglang_host: str = "http://localhost",
        sglang_port: int = 30000,
        api_base: Optional[str] = None,
        preference_aware_level: str = "none",
    ):
        self.memory_system = RobustAgenticMemorySystem(
            model_name="all-MiniLM-L6-v2",
            llm_backend=backend,
            llm_model=model,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
            api_base=api_base,
            preference_aware_level=preference_aware_level,
        )
        self.helper_llm = RobustLLMController(
            backend=backend,
            model=model,
            api_key=None,
            api_base=api_base,
            sglang_host=sglang_host,
            sglang_port=sglang_port,
        )
        self.retrieve_k = retrieve_k
        self.answer_temperature = answer_temperature

    def add_memory(self, content: str, time: Optional[str] = None) -> None:
        self.memory_system.add_note(content, time=time)

    def load_cached_state(
        self,
        memory_cache_file: Path,
        retriever_cache_file: Path,
        retriever_cache_embeddings_file: Path,
    ) -> None:
        with memory_cache_file.open("rb") as f:
            self.memory_system.memories = pickle.load(f)

        if retriever_cache_file.exists() and retriever_cache_embeddings_file.exists():
            self.memory_system.retriever = self.memory_system.retriever.load(
                str(retriever_cache_file), str(retriever_cache_embeddings_file)
            )
        else:
            self.memory_system.retriever = self.memory_system.retriever.load_from_local_memory(
                self.memory_system.memories, "all-MiniLM-L6-v2"
            )

    def save_cached_state(
        self,
        memory_cache_file: Path,
        retriever_cache_file: Path,
        retriever_cache_embeddings_file: Path,
    ) -> None:
        with memory_cache_file.open("wb") as f:
            pickle.dump(self.memory_system.memories, f)
        self.memory_system.retriever.save(
            str(retriever_cache_file), str(retriever_cache_embeddings_file)
        )

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

    def answer_mcq(self, question: str, option_mapping: Dict[str, str]) -> Dict[str, str]:
        keywords = self.generate_query_keywords(question, option_mapping)
        raw_context = self.memory_system.find_related_memories_raw(keywords, k=self.retrieve_k)
        context = raw_context
        options_text = "\n".join(f"{key}. {value}" for key, value in option_mapping.items())

        prompt = f"""You are answering a PersonaMem benchmark question using retrieved conversation memories.

Use only the retrieved memories below as evidence. Focus on the user's demonstrated preferences, priorities, and changes over time. Avoid generic common-sense guessing.

Retrieved memories:
{context}

Question:
{question}

Options:
{options_text}

Think briefly about which option is best supported by the memories. Then end with exactly one line in this format:
Final Answer: <LETTER>"""

        response = self.memory_system.llm_controller.llm.get_completion(
            prompt,
            temperature=self.answer_temperature,
            max_tokens=4096,
        )
        return {
            "response": response,
            "keywords": keywords,
            "raw_context": raw_context,
            "narrowed_context": "",
            "prompt": prompt,
        }

    def answer_openended(self, question: str) -> Dict[str, str]:
        keywords = self.generate_query_keywords(question)
        raw_context = self.memory_system.find_related_memories_raw(keywords, k=self.retrieve_k)
        context = raw_context

        prompt = f"""You are answering a PersonaMem benchmark question using retrieved conversation memories.

Use only the retrieved memories below as evidence. Focus on the user's demonstrated preferences, priorities, and changes over time. Avoid generic common-sense guessing.

Retrieved memories:
{context}

Question:
{question}

Answer the question directly in natural language. Keep the answer grounded in the retrieved memories and avoid mentioning unsupported details."""

        response = self.memory_system.llm_controller.llm.get_completion(
            prompt,
            temperature=self.answer_temperature,
            max_tokens=4096,
        )
        return {
            "response": response,
            "keywords": keywords,
            "raw_context": raw_context,
            "narrowed_context": "",
            "prompt": prompt,
        }


class AgentManager:
    def __init__(
        self,
        model: str,
        backend: str,
        retrieve_k: int,
        answer_temperature: float,
        cache_dir: Path,
        include_system_messages: bool,
        max_live_agents: int,
        sglang_host: str,
        sglang_port: int,
        api_base: Optional[str],
        eval_logger: logging.Logger,
        preference_aware_level: str = "none",
    ):
        self.model = model
        self.backend = backend
        self.retrieve_k = retrieve_k
        self.answer_temperature = answer_temperature
        self.cache_dir = cache_dir
        self.include_system_messages = include_system_messages
        self.max_live_agents = max_live_agents
        self.sglang_host = sglang_host
        self.sglang_port = sglang_port
        self.api_base = api_base
        self.eval_logger = eval_logger
        self.preference_aware_level = preference_aware_level
        self.agent_cache: "OrderedDict[str, PersonaRobustAgent]" = OrderedDict()

    def _new_agent(self) -> PersonaRobustAgent:
        return PersonaRobustAgent(
            model=self.model,
            backend=self.backend,
            retrieve_k=self.retrieve_k,
            answer_temperature=self.answer_temperature,
            sglang_host=self.sglang_host,
            sglang_port=self.sglang_port,
            api_base=self.api_base,
            preference_aware_level=self.preference_aware_level,
        )

    def _cache_paths(self, chat_history_path: Path, persona_id: str) -> Tuple[Path, Path, Path]:
        cache_key = build_cache_key(chat_history_path, self.include_system_messages, persona_id)
        memory_cache = self.cache_dir / f"memory_cache_{cache_key}.pkl"
        retriever_cache = self.cache_dir / f"retriever_cache_{cache_key}.pkl"
        retriever_embeddings = self.cache_dir / f"retriever_cache_embeddings_{cache_key}.npy"
        return memory_cache, retriever_cache, retriever_embeddings

    def get_agent(self, chat_history_path: Path, persona_id: str) -> PersonaRobustAgent:
        cache_key = build_cache_key(chat_history_path, self.include_system_messages, persona_id)
        if cache_key in self.agent_cache:
            agent = self.agent_cache.pop(cache_key)
            self.agent_cache[cache_key] = agent
            return agent

        agent = self._new_agent()
        memory_cache, retriever_cache, retriever_embeddings = self._cache_paths(chat_history_path, persona_id)
        status_path = build_status_path(self.cache_dir, cache_key)
        build_status = load_build_status(status_path)

        if memory_cache.exists():
            self.eval_logger.info("Loading cached A-MEM state for %s", chat_history_path.name)
            agent.load_cached_state(memory_cache, retriever_cache, retriever_embeddings)

        resume_message_index = 0
        if memory_cache.exists() and (build_status.get("complete") or not build_status):
            self.eval_logger.info("Loaded complete cached A-MEM state for %s", chat_history_path.name)
        else:
            messages = load_chat_history_messages(chat_history_path)
            total_messages = len(messages)
            if memory_cache.exists():
                resume_message_index = min(int(build_status.get("next_message_index", 0) or 0), total_messages)
                self.eval_logger.info(
                    "Resuming A-MEM build for %s from message %s/%s",
                    chat_history_path.name,
                    resume_message_index,
                    total_messages,
                )
            else:
                self.eval_logger.info("Building A-MEM state for %s", chat_history_path.name)

            loaded_messages = resume_message_index
            for idx, message in enumerate(
                make_progress(messages[resume_message_index:], desc=f"Build memory {chat_history_path.stem}", leave=False),
                start=resume_message_index,
            ):
                role = str(message.get("role", "")).lower()
                if not self.include_system_messages and role == "system":
                    continue
                content = str(message.get("content", "")).strip()
                if not content:
                    continue
                agent.add_memory(serialize_chat_message(message, idx), time=f"message_{idx:04d}")
                loaded_messages += 1
                save_persona_build_checkpoint(
                    agent,
                    memory_cache,
                    retriever_cache,
                    retriever_embeddings,
                    status_path,
                    cache_key,
                    idx + 1,
                    total_messages,
                    complete=False,
                )
            save_persona_build_checkpoint(
                agent,
                memory_cache,
                retriever_cache,
                retriever_embeddings,
                status_path,
                cache_key,
                total_messages,
                total_messages,
                complete=True,
            )
            self.eval_logger.info(
                "Cached %d messages for %s", loaded_messages, chat_history_path.name
            )

        self.agent_cache[cache_key] = agent
        while len(self.agent_cache) > self.max_live_agents:
            self.agent_cache.popitem(last=False)
        return agent


def resolve_chat_history_path(row: Dict[str, str], size: str, persona_root: Path) -> Path:
    size_column = f"chat_history_{size}_link"
    raw_path = row.get(size_column) or row.get("chat_history_link") or ""
    if not raw_path:
        raise FileNotFoundError(f"No chat history path available for size={size}")

    candidate = Path(raw_path)
    if candidate.is_absolute():
        return candidate

    root_candidate = (persona_root / candidate).resolve()
    if root_candidate.exists():
        return root_candidate

    return candidate.resolve()


def normalize_chain_value(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def build_persona_chain_key(row: Dict[str, str]) -> Tuple[str, ...]:
    """Build a post-hoc preference trajectory key from available benchmark metadata."""
    chain_id = normalize_chain_value(row.get("chain_id", ""))
    if chain_id:
        return ("chain_id", chain_id)

    persona_id = normalize_chain_value(row.get("persona_id", ""))
    for trajectory_col in ("trajectory_id", "full_trajectory", "preference_trajectory"):
        trajectory = normalize_chain_value(row.get(trajectory_col, ""))
        if trajectory:
            dimension = normalize_chain_value(
                row.get("preference_dimension", "")
                or row.get("preference_id", "")
                or row.get("topic_label", "")
                or row.get("topic_preference", "")
            )
            return ("trajectory", persona_id, dimension, trajectory)

    preference = normalize_chain_value(row.get("preference", ""))
    if preference:
        return ("preference", persona_id, preference)

    topic = normalize_chain_value(
        row.get("preference_dimension", "")
        or row.get("preference_id", "")
        or row.get("topic_label", "")
        or row.get("topic_preference", "")
        or row.get("topic_query", "")
    )
    change_family = normalize_chain_value(row.get("change_family", ""))
    return ("fallback", persona_id, topic, change_family)


def compute_metrics_lines(rows: List[Dict[str, str]], size: str) -> List[str]:
    col = f"is_correct_mcq_{size}"
    lines: List[str] = []

    total = sum(1 for row in rows if row.get(col) in ("True", "False"))
    correct = sum(1 for row in rows if row.get(col) == "True")
    lines.append(f"Overall: {correct}/{total} = {correct/total:.3f}" if total else "No data")

    processed_rows = [row for row in rows if row.get(col) in ("True", "False")]

    def collect_by(key_fn):
        stats = defaultdict(lambda: {"c": 0, "t": 0})
        for row in processed_rows:
            key = key_fn(row)
            stats[key]["t"] += 1
            if row[col] == "True":
                stats[key]["c"] += 1
        return stats

    def collect_chains_by_key(chain_rows: List[Dict[str, str]], key_fn):
        chains = defaultdict(list)
        for row in chain_rows:
            chains[key_fn(row)].append(row)
        passed = sum(1 for rows_for_chain in chains.values() if all(row[col] == "True" for row in rows_for_chain))
        return chains, passed

    def collect_chains(chain_rows: List[Dict[str, str]]):
        return collect_chains_by_key(chain_rows, build_persona_chain_key)

    def question_type_key(row: Dict[str, str]) -> str:
        return normalize_chain_value(row.get("ood_type", "") or row.get("topic_query", ""))

    if processed_rows:
        chains, chain_passed = collect_chains(processed_rows)
        chain_total = len(chains)
        chain_sizes = defaultdict(int)
        for rows_for_chain in chains.values():
            chain_sizes[len(rows_for_chain)] += 1
        size_text = ", ".join(f"{size}q:{count}" for size, count in sorted(chain_sizes.items()))
        lines.append("\nPost-hoc Chain Exact Match:")
        lines.append(
            f"  Overall chain acc: {chain_passed}/{chain_total} = {chain_passed/chain_total:.3f}"
        )
        lines.append(
            "  Chain definition: persona_id + explicit trajectory/full_trajectory when present; "
            "else persona_id + preference text; else persona/topic/change_family fallback"
        )
        lines.append(f"  Chain size distribution: {size_text}")

        temporal_rows = [
            row for row in processed_rows
            if normalize_chain_value(row.get("ood_type", "")) == "temporal_trajectory"
            or normalize_chain_value(row.get("topic_query", "")) == "ood_temporal_trajectory"
        ]
        if temporal_rows:
            temporal_chains, temporal_passed = collect_chains(temporal_rows)
            temporal_total = len(temporal_chains)
            lines.append(
                f"  Temporal-trajectory chains: {temporal_passed}/{temporal_total} = {temporal_passed/temporal_total:.3f}"
            )

        lines.append("\nAlternative bucket exact-match metrics:")
        alt_chain_specs = [
            (
                "persona",
                lambda row: (normalize_chain_value(row.get("persona_id", "")),),
            ),
            (
                "persona x question_type",
                lambda row: (normalize_chain_value(row.get("persona_id", "")), question_type_key(row)),
            ),
            (
                "persona x question_type x difficulty",
                lambda row: (
                    normalize_chain_value(row.get("persona_id", "")),
                    question_type_key(row),
                    normalize_chain_value(row.get("ood_difficulty", "")),
                ),
            ),
            (
                "persona x question_type x preference",
                lambda row: (
                    normalize_chain_value(row.get("persona_id", "")),
                    question_type_key(row),
                    normalize_chain_value(row.get("preference", "")),
                ),
            ),
        ]
        for label, key_fn in alt_chain_specs:
            alt_chains, alt_passed = collect_chains_by_key(processed_rows, key_fn)
            alt_total = len(alt_chains)
            alt_sizes = defaultdict(int)
            for rows_for_chain in alt_chains.values():
                alt_sizes[len(rows_for_chain)] += 1
            alt_size_text = ", ".join(f"{size}q:{count}" for size, count in sorted(alt_sizes.items()))
            lines.append(
                f"  {label}: {alt_passed}/{alt_total} = {alt_passed/alt_total:.3f} (sizes: {alt_size_text})"
            )

    by_k = collect_by(lambda row: row.get("change_k", ""))
    lines.append("\nBy k:")
    for key in sorted(by_k.keys()):
        stats = by_k[key]
        lines.append(f"  k={key}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

    by_family = collect_by(lambda row: row.get("change_family", ""))
    lines.append("\nBy family:")
    for key in sorted(by_family.keys()):
        stats = by_family[key]
        lines.append(f"  {key}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

    by_qtype = collect_by(lambda row: "family_specific" if row.get("family_query") else "general")
    lines.append("\nBy question type:")
    for key in sorted(by_qtype.keys()):
        stats = by_qtype[key]
        lines.append(f"  {key}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

    by_cross = collect_by(lambda row: (row.get("change_k", "?"), row.get("change_family", "?")))
    lines.append("\nK x Family:")
    for key in sorted(by_cross.keys()):
        stats = by_cross[key]
        lines.append(f"  k={key[0]}, {key[1]}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

    by_ood = collect_by(lambda row: "OOD" if row.get("ood_type", "") else "In-Distribution")
    lines.append("\nOOD vs Non-OOD:")
    for key in sorted(by_ood.keys()):
        stats = by_ood[key]
        lines.append(f"  {key}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

    ood_rows = [row for row in rows if row.get("ood_type", "")]
    if ood_rows:
        def collect_ood(key_fn):
            stats = defaultdict(lambda: {"c": 0, "t": 0})
            for row in ood_rows:
                if row.get(col) not in ("True", "False"):
                    continue
                key = key_fn(row)
                stats[key]["t"] += 1
                if row[col] == "True":
                    stats[key]["c"] += 1
            return stats

        by_ood_type = collect_ood(lambda row: row.get("ood_type", "unknown"))
        lines.append("\nBy OOD type:")
        for key in sorted(by_ood_type.keys()):
            stats = by_ood_type[key]
            lines.append(f"  {key}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

        by_ood_difficulty = collect_ood(lambda row: row.get("ood_difficulty", "unknown"))
        lines.append("\nBy OOD difficulty:")
        for key in sorted(by_ood_difficulty.keys()):
            stats = by_ood_difficulty[key]
            lines.append(f"  {key}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

        by_ood_cross = collect_ood(lambda row: (row.get("ood_type", "?"), row.get("ood_difficulty", "?")))
        lines.append("\nOOD type x difficulty:")
        for key in sorted(by_ood_cross.keys()):
            stats = by_ood_cross[key]
            lines.append(f"  {key[0]}/{key[1]}: {stats['c']}/{stats['t']} = {stats['c']/stats['t']:.3f}")

    return lines


def write_metrics_file(results_csv_path: Path, rows: List[Dict[str, str]], size: str) -> Path:
    metrics_path = results_csv_path.with_name(f"{results_csv_path.stem}_metrics.txt")
    metrics_lines = compute_metrics_lines(rows, size)
    metrics_path.write_text("\n".join(metrics_lines) + "\n", encoding="utf-8")
    return metrics_path


def write_usage_summary_file(results_csv_path: Path, usage_recorder: LLMUsageRecorder) -> Path:
    usage_summary_path = results_csv_path.with_name(f"{results_csv_path.stem}_llm_usage_summary.json")
    usage_summary_path.write_text(
        json.dumps(usage_recorder.summary(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return usage_summary_path


def build_output_path(
    output_arg: Optional[str],
    persona_root: Path,
    benchmark_file: Path,
    model: str,
    size: str,
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
        f"amem_robust_{size}_bench-{sanitize_filename_part(benchmark_file.stem)}"
        f"_model-{sanitize_filename_part(model)}{worker_suffix}_{timestamp}.csv"
    )
    return results_dir / filename


def evaluate_persona_benchmark(
    benchmark_file: Path,
    model: str,
    backend: str,
    size: str,
    output_path: Path,
    persona_root: Path,
    retrieve_k: int,
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
    preference_aware_level: str = "none",
) -> Path:
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
    worker_suffix = f"_worker{worker_id}" if num_workers > 1 else ""
    log_path = Path(__file__).resolve().parent / "logs" / (
        f"persona_robust_{sanitize_filename_part(model)}_{backend}_{size}{worker_suffix}_{timestamp}.log"
    )
    eval_logger = setup_logger(log_path)

    eval_logger.info("Persona root: %s", persona_root)
    eval_logger.info("Loading benchmark from %s", benchmark_file)
    eval_logger.info("Size: %s", size)
    eval_logger.info("Backend: %s", backend)
    eval_logger.info("Using robust A-MEM memory layer (preference_aware_level=%s)", preference_aware_level)

    fieldnames, rows = load_benchmark_rows(benchmark_file, persona_ids=persona_ids, max_items=max_items)
    eval_logger.info("Loaded %d benchmark rows before worker sharding", len(rows))

    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if worker_id < 0 or worker_id >= num_workers:
        raise ValueError("worker_id must satisfy 0 <= worker_id < num_workers")
    if num_workers > 1:
        rows = shard_rows_by_chat_history(rows, size=size, persona_root=persona_root, num_workers=num_workers, worker_id=worker_id)
        eval_logger.info("Worker %d/%d processing %d rows after chat-history sharding", worker_id, num_workers, len(rows))

    pref_suffix_map = {
        "none": "",
        "patch_only": "_prefaware_patchonly",
        "full": "_prefaware_full",
    }
    pref_suffix = pref_suffix_map.get(preference_aware_level, "")
    cache_dir = Path(__file__).resolve().parent / (
        f"cached_memories_persona_robust_{backend}_{sanitize_filename_part(model)}_{size}{pref_suffix}"
    )
    cache_dir.mkdir(parents=True, exist_ok=True)

    agent_manager = AgentManager(
        model=model,
        backend=backend,
        retrieve_k=retrieve_k,
        answer_temperature=answer_temperature,
        cache_dir=cache_dir,
        include_system_messages=include_system_messages,
        max_live_agents=max_live_agents,
        sglang_host=sglang_host,
        sglang_port=sglang_port,
        api_base=api_base,
        eval_logger=eval_logger,
        preference_aware_level=preference_aware_level,
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
    processed = resume_row_index
    mcq_processed = sum(1 for row in written_rows if row.get(f"is_correct_mcq_{size}") in ("True", "False"))
    usage_log_path = output_path.with_name(f"{output_path.stem}_llm_usage.jsonl")
    if not resume_row_index and usage_log_path.exists():
        usage_log_path.unlink()
    usage_recorder = LLMUsageRecorder(
        usage_log_path,
        metadata={
            "suite": "persona",
            "method": "robust",
            "model": model,
            "backend": backend,
            "size": size,
            "benchmark_file": str(benchmark_file),
            "output_path": str(output_path),
            "worker_id": worker_id,
            "num_workers": num_workers,
        },
    )

    file_mode = "a" if resume_row_index else "w"
    with output_path.open(file_mode, encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=output_fieldnames)
        if not resume_row_index:
            writer.writeheader()

        for idx, row in enumerate(make_progress(rows[resume_row_index:], desc="Persona robust eval"), start=resume_row_index):
            output_row = row.copy()
            try:
                chat_history_path = resolve_chat_history_path(row, size=size, persona_root=persona_root)
                with llm_usage_recorder(usage_recorder), llm_usage_stage("ingestion"):
                    agent = agent_manager.get_agent(chat_history_path, str(row.get("persona_id", "")))
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
                    prediction = extract_final_answer(answer_result["response"])
                    is_correct = check_mcq_correctness(
                        prediction,
                        row.get("correct_answer", ""),
                        option_mapping,
                    )

                    output_row[f"model_response_mcq_{size}"] = answer_result["response"]
                    output_row[f"predicted_answer_mcq_{size}"] = prediction
                    output_row[f"is_correct_mcq_{size}"] = str(is_correct)
                    output_row[f"correct_mcq_option_{size}"] = correct_letter
                    output_row[f"raw_input_prompt_mcq_{size}"] = answer_result["prompt"]

                    if save_debug_columns:
                        output_row[f"retrieval_keywords_{size}"] = answer_result["keywords"]
                        output_row[f"retrieved_context_{size}"] = answer_result["raw_context"]
                        output_row[f"narrowed_context_{size}"] = answer_result["narrowed_context"]
                        output_row[f"amem_prompt_{size}"] = answer_result["prompt"]

                    if is_correct:
                        correct += 1
                    mcq_processed += 1

                if eval_mode in ("generative", "both"):
                    with llm_usage_recorder(usage_recorder), llm_usage_stage("qa.openended"):
                        openended_result = agent.answer_openended(question)
                    output_row[f"model_response_openended_{size}"] = openended_result["response"]
                    output_row[f"is_correct_openended_{size}"] = ""
                    output_row[f"raw_input_prompt_openended_{size}"] = openended_result["prompt"]

                    if save_debug_columns:
                        output_row[f"retrieval_keywords_openended_{size}"] = openended_result["keywords"]
                        output_row[f"retrieved_context_openended_{size}"] = openended_result["raw_context"]
                        output_row[f"narrowed_context_openended_{size}"] = openended_result["narrowed_context"]
                        output_row[f"amem_prompt_openended_{size}"] = openended_result["prompt"]

                processed += 1
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
                processed += 1
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
    if args.preference_aware_level and args.preference_aware_level != "none":
        child_base.extend(["--preference_aware_level", args.preference_aware_level])
    elif args.preference_aware:
        child_base.append("--preference_aware")

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
    parser = argparse.ArgumentParser(description="Run robust A-MEM on Persona-release benchmark")
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
    parser.add_argument("--retrieve_k", type=int, default=12, help="Number of memories to retrieve")
    parser.add_argument(
        "--answer_temperature",
        type=float,
        default=0.0,
        help="Temperature for the final MCQ answer call",
    )
    parser.add_argument("--sglang_host", type=str, default="http://localhost", help="SGLang host")
    parser.add_argument("--sglang_port", type=int, default=30000, help="SGLang port")
    parser.add_argument("--api_base", type=str, default=None,
                        help="Optional OpenAI-compatible API base URL. Useful for custom OpenRouter-compatible endpoints.")
    parser.add_argument(
        "--skip_system_messages",
        action="store_true",
        help="Do not index chat-history system messages into A-MEM",
    )
    parser.add_argument(
        "--save_debug_columns",
        action="store_true",
        help="Write retrieval keywords, contexts, and prompts into the output CSV",
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
    parser.add_argument("--preference_aware", action="store_true", default=False,
                        help="[legacy] Use preference-aware prompts for memory building. "
                             "Equivalent to --preference_aware_level full.")
    parser.add_argument("--preference_aware_level", type=str, default="none",
                        choices=["none", "patch_only", "full"],
                        help="Granular preference-aware mode: "
                             "'none' = original prompts; "
                             "'patch_only' = PREF ANALYZE with base graph prompts; "
                             "'full' = PREF prompts everywhere (legacy --preference_aware behavior).")
    args = parser.parse_args()

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
        preference_aware_level=args.preference_aware_level,
    )


if __name__ == "__main__":
    main()
