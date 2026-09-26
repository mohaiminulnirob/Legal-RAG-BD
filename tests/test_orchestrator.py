import unittest

from src.reasoning.orchestrator import OrchestrationStatus, orchestrate


def record(chunk_id="chunk-1", content="Murder is an offence. Whoever intentionally causes death commits murder and shall be punished."):
    return {
        "chunk_id": chunk_id,
        "act_title": "Penal Code",
        "section_id": "300",
        "section_content": content,
        "source_file": "penal-code.json",
        "source_url": "https://example.test/penal-code/300",
    }


class SearchQueue:
    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def __call__(self, query, *, top_k, candidate_k, model):
        self.calls.append({"query": query, "top_k": top_k, "candidate_k": candidate_k})
        index = min(len(self.calls) - 1, len(self.outputs) - 1)
        return self.outputs[index]


class FakeLLM:
    model = "test-model"

    def __init__(self):
        self.calls = []

    def generate(self, system_prompt, user_prompt):
        self.calls.append((system_prompt, user_prompt))
        return "Answer grounded in the supplied legal evidence."


class OrchestratorTests(unittest.TestCase):
    def test_fully_supported_multistep_query_generates_only_after_steps_complete(self):
        search = SearchQueue([[record()]])
        llm = FakeLLM()
        result = orchestrate(
            "What offence and punishment apply for murder?",
            search_fn=search,
            answer_llm=llm,
        )
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(result.answer, "Answer grounded in the supplied legal evidence.")
        self.assertEqual(len(result.state.assessments), 2)
        self.assertEqual([item.step_id for item in result.state.assessments], ["step_1", "step_2"])
        self.assertEqual(len(llm.calls), 1)

    def test_missing_evidence_retries_within_budget_then_acknowledges(self):
        search = SearchQueue([[], []])
        result = orchestrate(
            "What offence applies?",
            search_fn=search,
            max_retrieval_attempts=1,
        )
        self.assertEqual(result.status, OrchestrationStatus.UNCERTAINTY_ACKNOWLEDGED)
        self.assertEqual(len(search.calls), 2)
        self.assertEqual(search.calls[0]["candidate_k"], 20)
        self.assertEqual(search.calls[1]["candidate_k"], 40)
        self.assertEqual(result.state.retrieval_attempts_for("step_1"), 1)

    def test_clarification_pause_then_user_fact_resumes_and_reassesses(self):
        partial_record = record(content="Murder is an offence.")
        search = SearchQueue([[partial_record], [record("chunk-2", "Whoever intentionally causes death.")]])
        first = orchestrate(
            "What offence applies?",
            search_fn=search,
            missing_user_facts={"step_1": ("whether the act was intentional",)},
        )
        self.assertEqual(first.status, OrchestrationStatus.AWAITING_CLARIFICATION)
        self.assertIn("intentional", first.clarification_requests[0].question)

        resumed = orchestrate(
            state=first.state,
            clarification_answers={"whether the act was intentional": "Yes, it was intentional."},
            search_fn=search,
        )
        self.assertEqual(resumed.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(resumed.state.user_facts[0].value, "Yes, it was intentional.")
        self.assertGreaterEqual(len(resumed.state.assessments), 2)
        self.assertIn("Yes, it was intentional.", search.calls[-1]["query"])

    def test_handler_can_collect_clarification_during_one_run(self):
        search = SearchQueue([[record(content="Murder is an offence.")], [record()]])
        result = orchestrate(
            "What offence applies?",
            search_fn=search,
            missing_user_facts={"step_1": ("whether the act was intentional",)},
            clarification_handler=lambda requests: {requests[0].target: "Yes."},
        )
        self.assertEqual(result.status, OrchestrationStatus.COMPLETED)
        self.assertEqual(len(result.state.user_facts), 1)

    def test_persistent_uncertainty_returns_structured_evidence_record(self):
        search = SearchQueue([[record(content="A different procedural rule applies.")]])
        result = orchestrate(
            "What offence applies?",
            search_fn=search,
            max_retrieval_attempts=0,
        )
        self.assertEqual(result.status, OrchestrationStatus.UNCERTAINTY_ACKNOWLEDGED)
        self.assertIsNotNone(result.uncertainty)
        self.assertIn("cannot be established", result.answer)

    def test_uncertainty_preserves_evidence_from_previously_supported_steps(self):
        search = SearchQueue(
            [
                [record()],
                [record("chunk-2", "Punishment may apply in some circumstances.")],
            ]
        )
        result = orchestrate(
            "What offence and punishment apply for murder?",
            search_fn=search,
            max_retrieval_attempts=0,
        )
        self.assertEqual(result.status, OrchestrationStatus.UNCERTAINTY_ACKNOWLEDGED)
        self.assertEqual(result.state.current_step_id, "step_2")
        self.assertEqual(result.uncertainty.supported_evidence[0].reasoning_objective,
                         result.state.plan.steps[0].objective)
        self.assertIn(result.state.plan.steps[0].objective, result.answer)

    def test_unresolved_earlier_step_prevents_later_step_execution(self):
        search = SearchQueue([[]])
        result = orchestrate(
            "What offence and punishment apply for murder?",
            search_fn=search,
            max_retrieval_attempts=0,
        )
        self.assertEqual(result.status, OrchestrationStatus.UNCERTAINTY_ACKNOWLEDGED)
        self.assertEqual(result.state.current_step_id, "step_1")
        self.assertEqual(len(search.calls), 1)
        self.assertEqual([item.step_id for item in result.state.assessments], ["step_1"])

    def test_retrieval_budget_is_zero_or_more_and_never_exceeded(self):
        with self.assertRaises(ValueError):
            orchestrate("What offence applies?", search_fn=SearchQueue([[]]), max_retrieval_attempts=11)

    def test_final_llm_is_not_called_while_waiting_for_clarification(self):
        llm = FakeLLM()
        result = orchestrate(
            "What offence applies?",
            search_fn=SearchQueue([[record(content="Murder is an offence.")]]),
            missing_user_facts={"step_1": ("whether the act was intentional",)},
            answer_llm=llm,
        )
        self.assertEqual(result.status, OrchestrationStatus.AWAITING_CLARIFICATION)
        self.assertEqual(llm.calls, [])


if __name__ == "__main__":
    unittest.main()
