import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import test_persona_robust as persona


class FakeAgent:
    def __init__(self):
        self.questions = []
        self.openended_questions = []

    def answer_mcq(self, question, option_mapping):
        self.questions.append((question, dict(option_mapping)))
        correct_letter = next(key for key, value in option_mapping.items() if value == "Green tea")
        return {
            "response": f"Reasoning\nFinal Answer: {correct_letter}",
            "keywords": "tea, coffee",
            "raw_context": "ctx",
            "narrowed_context": "nctx",
            "prompt": "prompt",
        }

    def answer_openended(self, question):
        self.openended_questions.append(question)
        return {
            "response": "Green tea is the best fit.",
            "keywords": "tea, direct answer",
            "raw_context": "open_ctx",
            "narrowed_context": "open_nctx",
            "prompt": "open_prompt",
        }


class FakeAgentManager:
    last_instance = None

    def __init__(self, *args, **kwargs):
        self.agent = FakeAgent()
        FakeAgentManager.last_instance = self

    def get_agent(self, chat_history_path, persona_id=""):
        return self.agent


class PersonaRobustLogicTests(unittest.TestCase):
    def test_parse_user_query_appends_recall_instruction(self):
        raw = "{'role': 'user', 'content': 'What dessert would I like?'}"
        parsed = persona.parse_user_query(raw)
        self.assertEqual(parsed["role"], "user")
        self.assertIn("What dessert would I like?", parsed["content"])
        self.assertIn("Please recall my related preferences", parsed["content"])

    def test_resolve_chat_history_path_uses_persona_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chat = root / "data/chat_history_32k/sample.json"
            chat.parent.mkdir(parents=True, exist_ok=True)
            chat.write_text("[]", encoding="utf-8")
            row = {"chat_history_32k_link": "data/chat_history_32k/sample.json"}
            resolved = persona.resolve_chat_history_path(row, size="32k", persona_root=root)
            self.assertEqual(resolved, chat.resolve())

    def test_compute_metrics_lines_prefers_explicit_chain_id(self):
        rows = [
            {
                "persona_id": "0",
                "preference": "Different text A",
                "chain_id": "chain_shared",
                "is_correct_mcq_32k": "True",
            },
            {
                "persona_id": "0",
                "preference": "Different text B",
                "chain_id": "chain_shared",
                "is_correct_mcq_32k": "False",
            },
        ]

        metrics_text = "\n".join(persona.compute_metrics_lines(rows, "32k"))

        self.assertIn("Overall: 1/2 = 0.500", metrics_text)
        self.assertIn("Overall chain acc: 0/1 = 0.000", metrics_text)
        self.assertIn("Chain size distribution: 2q:1", metrics_text)

    def test_compute_metrics_lines_includes_posthoc_chain_exact_match(self):
        rows = [
            {
                "persona_id": "0",
                "preference": "Likes tea.; Now prefers green tea.",
                "is_correct_mcq_32k": "True",
                "ood_type": "temporal_trajectory",
            },
            {
                "persona_id": "0",
                "preference": "Likes tea.; Now prefers green tea.",
                "is_correct_mcq_32k": "False",
                "ood_type": "temporal_trajectory",
            },
            {
                "persona_id": "0",
                "preference": "Likes quiet cafes.",
                "is_correct_mcq_32k": "True",
                "ood_type": "single_pattern_transfer",
            },
        ]

        metrics_text = "\n".join(persona.compute_metrics_lines(rows, "32k"))

        self.assertIn("Overall: 2/3 = 0.667", metrics_text)
        self.assertIn("Overall chain acc: 1/2 = 0.500", metrics_text)
        self.assertIn("Temporal-trajectory chains: 0/1 = 0.000", metrics_text)
        self.assertIn("Chain size distribution: 1q:1, 2q:1", metrics_text)

    def test_evaluate_persona_benchmark_writes_persona_shaped_outputs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            benchmark = root / "benchmark_v34/text/benchmark_49p_ood_v34.csv"
            benchmark.parent.mkdir(parents=True, exist_ok=True)

            chat = root / "data/chat_history_32k/chat_history_persona0.json"
            chat.parent.mkdir(parents=True, exist_ok=True)
            chat.write_text(json.dumps({
                "chat_history": [
                    {"role": "system", "content": "system seed"},
                    {"role": "user", "content": "I like tea."},
                ]
            }), encoding="utf-8")

            fieldnames = [
                "persona_id",
                "chat_history_32k_link",
                "user_query",
                "correct_answer",
                "incorrect_answers",
                "ood_type",
                "ood_difficulty",
            ]
            with benchmark.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow({
                    "persona_id": "0",
                    "chat_history_32k_link": "data/chat_history_32k/chat_history_persona0.json",
                    "user_query": "{'role': 'user', 'content': 'Which drink would I choose?'}",
                    "correct_answer": "Green tea",
                    "incorrect_answers": json.dumps(["Black coffee", "Orange soda", "Sparkling water"]),
                    "ood_type": "single_pattern_transfer",
                    "ood_difficulty": "L1",
                })

            output_path = root / "results.csv"

            with patch.object(persona, "AgentManager", FakeAgentManager):
                result_path = persona.evaluate_persona_benchmark(
                    benchmark_file=benchmark,
                    model="fake-model",
                    backend="openai",
                    size="32k",
                    output_path=output_path,
                    persona_root=root,
                    retrieve_k=4,
                    answer_temperature=0.0,
                    sglang_host="http://localhost",
                    sglang_port=30000,
                    api_base=None,
                    include_system_messages=True,
                    max_live_agents=1,
                    num_workers=1,
                    worker_id=0,
                    persona_ids=None,
                    max_items=None,
                    save_debug_columns=True,
                    eval_mode="both",
                )

            self.assertEqual(result_path, output_path)
            self.assertTrue(output_path.exists())
            metrics_path = output_path.with_name("results_metrics.txt")
            self.assertTrue(metrics_path.exists())

            with output_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row["is_correct_mcq_32k"], "True")
            self.assertEqual(row["predicted_answer_mcq_32k"], row["correct_mcq_option_32k"])
            self.assertIn(row["correct_mcq_option_32k"], {"A", "B", "C", "D"})
            self.assertEqual(row["retrieval_keywords_32k"], "tea, coffee")
            self.assertEqual(row["retrieved_context_32k"], "ctx")
            self.assertEqual(row["narrowed_context_32k"], "nctx")
            self.assertEqual(row["amem_prompt_32k"], "prompt")
            self.assertEqual(row["model_response_openended_32k"], "Green tea is the best fit.")
            self.assertEqual(row["is_correct_openended_32k"], "")
            self.assertEqual(row["retrieval_keywords_openended_32k"], "tea, direct answer")
            self.assertEqual(row["retrieved_context_openended_32k"], "open_ctx")
            self.assertEqual(row["narrowed_context_openended_32k"], "open_nctx")
            self.assertEqual(row["amem_prompt_openended_32k"], "open_prompt")

            recorded_question, option_mapping = FakeAgentManager.last_instance.agent.questions[0]
            self.assertIn("Which drink would I choose?", recorded_question)
            self.assertIn("Please recall my related preferences", recorded_question)
            self.assertNotIn("Please choose the best answer from the following options", recorded_question)
            self.assertEqual(set(option_mapping.keys()), {"A", "B", "C", "D"})
            self.assertEqual(FakeAgentManager.last_instance.agent.openended_questions, [recorded_question])

            metrics_text = metrics_path.read_text(encoding="utf-8")
            self.assertIn("Overall: 1/1 = 1.000", metrics_text)
            self.assertIn("single_pattern_transfer: 1/1 = 1.000", metrics_text)
            self.assertIn("L1: 1/1 = 1.000", metrics_text)


if __name__ == "__main__":
    unittest.main()
