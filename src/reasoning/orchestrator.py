"""Bounded end-to-end orchestration of the deterministic legal reasoning flow."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, replace
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from src.analysis.query_analyzer import analyze_query
from src.analysis.schemas import QueryAnalysis
from src.rag.context import build_context
from src.rag.llm import DEFAULT_MODEL, GroqResponsesLLM, LLMClient
from src.rag.prompt import SYSTEM_PROMPT, build_user_prompt
from src.reasoning.clarification import ClarificationRequest, build_clarification_requests
from src.reasoning.execution_state import ExecutionState
from src.reasoning.next_action import NextAction, NextActionDecision, decide_next_action
from src.reasoning.planner import build_reasoning_plan
from src.reasoning.schemas import ReasoningStep, StepStatus
from src.reasoning.step_retriever import SearchFunction, retrieve_step
from src.reasoning.sufficiency import EvidenceAssessment, assess_step
from src.reasoning.uncertainty import UncertaintyAcknowledgement, build_uncertainty_acknowledgement


MAX_CANDIDATE_K = 100
MAX_RETRIEVAL_ATTEMPTS = 10
MAX_CLARIFICATION_ATTEMPTS = 5


class OrchestrationStatus(StrEnum):
    COMPLETED = "completed"
    NO_REASONING_STEPS = "no_reasoning_steps"
    AWAITING_CLARIFICATION = "awaiting_clarification"
    UNCERTAINTY_ACKNOWLEDGED = "uncertainty_acknowledged"
    STOPPED = "stopped"


ClarificationHandler = Callable[[tuple[ClarificationRequest, ...]], Mapping[str, str] | None]
StopCheck = Callable[[ExecutionState], bool]


@dataclass(frozen=True)
class OrchestrationResult:
    status: OrchestrationStatus
    analysis: QueryAnalysis
    state: ExecutionState
    clarification_requests: tuple[ClarificationRequest, ...] = ()
    uncertainty: UncertaintyAcknowledgement | None = None
    answer: str | None = None
    model: str | None = None
    schema_version: str = "1.0"

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "analysis": self.analysis.to_dict(),
            "state": self.state.to_dict(),
            "clarification_requests": [request.to_dict() for request in self.clarification_requests],
            "uncertainty": self.uncertainty.to_dict() if self.uncertainty else None,
            "answer": self.answer,
            "model": self.model,
            "schema_version": self.schema_version,
        }


def _prepare_search(
    search_fn: SearchFunction | None,
    retrieval_model: Any | None,
    *,
    bm25_index: Path,
    dense_database: Path,
    model_name: str | None,
) -> tuple[SearchFunction, Any | None]:
    if search_fn is not None:
        return search_fn, retrieval_model
    from src.retrieval import dense
    from src.retrieval.hybrid import search as hybrid_search

    if retrieval_model is None:
        retrieval_model = dense.create_embedding_model(model_name or dense.MODEL_NAME)
    return partial(
        hybrid_search,
        bm25_index=bm25_index,
        dense_database=dense_database,
    ), retrieval_model


def _merge_evidence(
    previous: Sequence[dict[str, Any]],
    new: Sequence[dict[str, Any]],
    *,
    maximum: int,
) -> tuple[dict[str, Any], ...]:
    combined: dict[str, dict[str, Any]] = {}
    for item in (*previous, *new):
        chunk_id = item.get("chunk_id")
        if not isinstance(chunk_id, str) or not chunk_id:
            continue
        combined[chunk_id] = {**combined.get(chunk_id, {}), **dict(item)}
    return tuple(list(combined.values())[:maximum])


def _retrieve(
    step: ReasoningStep,
    *,
    search_fn: SearchFunction,
    retrieval_model: Any | None,
    top_k: int,
    candidate_k: int,
    max_evidence: int,
    user_facts: Sequence[Any] = (),
) -> ReasoningStep:
    fact_marker = "\nUser-provided facts:\n"
    base_query = step.retrieval_query.split(fact_marker, 1)[0]
    facts = [item for item in user_facts if item.step_id == step.step_id]
    effective_query = base_query
    if facts:
        effective_query += fact_marker + "\n".join(f"{item.fact}: {item.value}" for item in facts)
    search_step = replace(step, retrieval_query=effective_query)
    retrieved = retrieve_step(
        search_step,
        top_k=top_k,
        candidate_k=candidate_k,
        search_fn=search_fn,
        retrieval_model=retrieval_model,
    )
    return replace(
        retrieved,
        evidence=_merge_evidence(step.evidence, retrieved.evidence, maximum=max_evidence),
    )


def _latest_assessments(state: ExecutionState) -> tuple[EvidenceAssessment, ...]:
    latest: dict[str, EvidenceAssessment] = {}
    for item in state.assessments:
        latest[item.step_id] = item
    return tuple(latest[step.step_id] for step in state.plan.steps if step.step_id in latest)


def _final_evidence(state: ExecutionState, assessments: Sequence[EvidenceAssessment]) -> tuple[dict[str, Any], ...]:
    relevant_ids = {chunk_id for item in assessments for chunk_id in item.relevant_evidence_ids}
    evidence: dict[str, dict[str, Any]] = {}
    for step in state.plan.steps:
        for record in step.evidence:
            chunk_id = record.get("chunk_id")
            if chunk_id in relevant_ids and isinstance(chunk_id, str):
                evidence[chunk_id] = dict(record)
    return tuple(evidence.values())


def _generate_answer(
    query: str,
    state: ExecutionState,
    assessments: Sequence[EvidenceAssessment],
    llm: LLMClient,
    *,
    uncertainty: UncertaintyAcknowledgement | None,
) -> str:
    evidence = _final_evidence(state, assessments)
    context = build_context(evidence)
    step_status = "\n".join(
        f"- {step.objective}: {assessment.status.value}; missing={', '.join(assessment.missing_requirements) or 'none'}"
        for step in state.plan.steps
        for assessment in assessments
        if assessment.step_id == step.step_id
    )
    controls = f"\n\nDeterministic step assessments:\n{step_status}"
    user_facts = [item for item in state.user_facts]
    if user_facts:
        facts_text = "\n".join(
            f"- Step {item.step_id}: {item.fact} = {item.value} (provided by the user)"
            for item in user_facts
        )
        controls += "\n\nUser-provided facts (do not treat as verified legal evidence):\n" + facts_text
    if uncertainty is not None:
        controls += "\n\nDeterministic uncertainty record (preserve these limitations):\n" + uncertainty.render()
        controls += "\nDo not state a definitive legal conclusion that this record says cannot be established."
    user_prompt = build_user_prompt(query, context + controls)
    return llm.generate(SYSTEM_PROMPT, user_prompt)


def orchestrate(
    query: str | None = None,
    *,
    state: ExecutionState | None = None,
    clarification_answers: Mapping[str, str] | None = None,
    missing_user_facts: Mapping[str, Sequence[str]] | None = None,
    clarification_handler: ClarificationHandler | None = None,
    stop_check: StopCheck | None = None,
    answer_llm: LLMClient | None = None,
    search_fn: SearchFunction | None = None,
    retrieval_model: Any | None = None,
    top_k: int = 5,
    candidate_k: int = 20,
    max_candidate_k: int = MAX_CANDIDATE_K,
    max_retrieval_attempts: int = 2,
    max_clarification_attempts: int = 1,
    bm25_index: Path = Path("data/processed/bm25_index.pkl"),
    dense_database: Path = Path("chroma_db"),
    model_name: str | None = None,
) -> OrchestrationResult:
    """Run the bounded reasoning loop, or resume it from a clarification reply.

    Retrieval and answer generation are injectable for offline tests. The
    deterministic layers decide every transition; an optional handler only
    collects clarification answers, and an optional LLM only writes the final
    natural-language response after a terminal result.
    """
    if top_k < 1 or candidate_k < top_k or candidate_k > MAX_CANDIDATE_K:
        raise ValueError("Require 1 <= top_k <= candidate_k <= 100.")
    if not candidate_k <= max_candidate_k <= MAX_CANDIDATE_K:
        raise ValueError("max_candidate_k must be between candidate_k and 100.")
    if not 0 <= max_retrieval_attempts <= MAX_RETRIEVAL_ATTEMPTS:
        raise ValueError("max_retrieval_attempts must be between 0 and 10.")
    if not 0 <= max_clarification_attempts <= MAX_CLARIFICATION_ATTEMPTS:
        raise ValueError("max_clarification_attempts must be between 0 and 5.")

    if state is None:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("A non-empty query is required for a new execution.")
        analysis = analyze_query(query)
        plan = build_reasoning_plan(analysis)
        state = ExecutionState.start(
            plan,
            query=analysis.query,
            max_retrieval_attempts=max_retrieval_attempts,
            max_clarification_attempts=max_clarification_attempts,
        )
        for step_id, facts in (missing_user_facts or {}).items():
            if isinstance(facts, str):
                raise ValueError("Missing user facts must be a sequence of fact descriptions, not one string.")
            state = state.with_pending_user_facts(tuple(facts), step_id=step_id)
    else:
        if query is not None and query != state.query:
            raise ValueError("A resumed execution must use its original query.")
        analysis = analyze_query(state.query)
        if missing_user_facts:
            raise ValueError("Supply missing user facts when starting an execution, not when resuming it.")
        active_pending = dict(state.pending_user_facts).get(state.current_step_id, ())
        last_active_action = next(
            (item for item in reversed(state.selected_actions) if item.step_id == state.current_step_id),
            None,
        )
        if not clarification_answers and active_pending and last_active_action is not None and last_active_action.action == NextAction.CLARIFY:
            requests = build_clarification_requests(last_active_action)
            requests = tuple(item for item in requests if item.target in active_pending)
            return OrchestrationResult(
                OrchestrationStatus.AWAITING_CLARIFICATION,
                analyze_query(state.query),
                state,
                clarification_requests=requests,
            )
        if clarification_answers:
            pending = active_pending
            if any(target not in pending for target in clarification_answers):
                raise ValueError("Clarification answers must match pending targets for the active step.")
            for target, value in clarification_answers.items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Clarification answers must be non-empty strings.")
                state = state.record_user_fact(target, value)
            remaining = set(dict(state.pending_user_facts).get(state.current_step_id, ()))
            if remaining and last_active_action is not None and last_active_action.action == NextAction.CLARIFY:
                requests = tuple(
                    item for item in build_clarification_requests(last_active_action) if item.target in remaining
                )
                return OrchestrationResult(
                    OrchestrationStatus.AWAITING_CLARIFICATION,
                    analyze_query(state.query),
                    state,
                    clarification_requests=requests,
                )

    if not state.plan.steps:
        return OrchestrationResult(OrchestrationStatus.NO_REASONING_STEPS, analysis, state)
    if state.stopped:
        return OrchestrationResult(OrchestrationStatus.STOPPED, analysis, state)

    search, retrieval_model = _prepare_search(
        search_fn,
        retrieval_model,
        bm25_index=bm25_index,
        dense_database=dense_database,
        model_name=model_name,
    )
    transition_limit = len(state.plan.steps) * (
        1 + max_retrieval_attempts + max_clarification_attempts + 2
    )
    transitions = 0
    final_status: OrchestrationStatus | None = None
    terminal_uncertainty: UncertaintyAcknowledgement | None = None
    pending_requests: tuple[ClarificationRequest, ...] = ()

    while state.current_step_id is not None and not state.stopped:
        transitions += 1
        if transitions > transition_limit:
            raise RuntimeError("Orchestration transition bound exceeded; refusing to continue the loop.")
        step = next(item for item in state.plan.steps if item.step_id == state.current_step_id)

        if step.status == StepStatus.PENDING:
            step = _retrieve(
                step,
                search_fn=search,
                retrieval_model=retrieval_model,
                top_k=top_k,
                candidate_k=candidate_k,
                max_evidence=max_candidate_k,
                user_facts=state.user_facts,
            )
            state = state.update_current_step(step)

        assessment = assess_step(step)
        state = state.record_assessment(assessment)
        phase12_state = state.next_action_state()
        if stop_check is not None and stop_check(state):
            phase12_state = replace(phase12_state, stop_requested=True)
        decision = decide_next_action(assessment, phase12_state)
        state = state.record_action(decision)

        if decision.action == NextAction.CONTINUE:
            state = state.advance_after_continue()
            continue

        if decision.action == NextAction.RETRIEVE_MORE:
            state = state.record_retrieval_attempt()
            factor = state.retrieval_attempts_for(step.step_id) + 1
            retry_candidate_k = min(max_candidate_k, candidate_k * factor)
            retry_top_k = min(retry_candidate_k, top_k * factor)
            step = _retrieve(
                step,
                search_fn=search,
                retrieval_model=retrieval_model,
                top_k=retry_top_k,
                candidate_k=retry_candidate_k,
                max_evidence=max_candidate_k,
                user_facts=state.user_facts,
            )
            state = state.update_current_step(step)
            continue

        if decision.action == NextAction.CLARIFY:
            pending_requests = build_clarification_requests(decision)
            state = state.record_clarification_requests(pending_requests)
            if clarification_handler is None:
                return OrchestrationResult(
                    OrchestrationStatus.AWAITING_CLARIFICATION,
                    analysis,
                    state,
                    clarification_requests=pending_requests,
                )
            answers = clarification_handler(pending_requests)
            if answers is None:
                return OrchestrationResult(
                    OrchestrationStatus.AWAITING_CLARIFICATION,
                    analysis,
                    state,
                    clarification_requests=pending_requests,
                )
            allowed_targets = {item.target for item in pending_requests}
            if any(target not in allowed_targets for target in answers):
                raise ValueError("Clarification handler returned an answer for an unrequested target.")
            for target, value in answers.items():
                if not isinstance(value, str) or not value.strip():
                    raise ValueError("Clarification answers must be non-empty strings.")
                state = state.record_user_fact(target, value)
            remaining = set(dict(state.pending_user_facts).get(step.step_id, ()))
            if remaining:
                still_pending = tuple(item for item in pending_requests if item.target in remaining)
                return OrchestrationResult(
                    OrchestrationStatus.AWAITING_CLARIFICATION,
                    analysis,
                    state,
                    clarification_requests=still_pending,
                )
            continue

        if decision.action == NextAction.ACKNOWLEDGE_UNCERTAINTY:
            latest_by_step = {item.step_id: item for item in _latest_assessments(state)}
            current_position = next(
                index for index, item in enumerate(state.plan.steps) if item.step_id == step.step_id
            )
            prior_supported = tuple(
                (prior_step, latest_by_step[prior_step.step_id])
                for prior_step in state.plan.steps[:current_position]
                if prior_step.step_id in latest_by_step
                and latest_by_step[prior_step.step_id].status.value == "supported"
            )
            terminal_uncertainty = build_uncertainty_acknowledgement(
                decision,
                assessment,
                step,
                unresolved_user_facts=dict(state.pending_user_facts).get(step.step_id, ()),
                prior_supported_steps=prior_supported,
            )
            final_status = OrchestrationStatus.UNCERTAINTY_ACKNOWLEDGED
            break

        if decision.action == NextAction.STOP:
            final_status = OrchestrationStatus.STOPPED
            break

    if final_status is None:
        final_status = OrchestrationStatus.COMPLETED
    latest = _latest_assessments(state)
    answer: str | None = terminal_uncertainty.render() if terminal_uncertainty else None
    model: str | None = None
    if answer_llm is not None and final_status in {
        OrchestrationStatus.COMPLETED,
        OrchestrationStatus.UNCERTAINTY_ACKNOWLEDGED,
    }:
        answer = _generate_answer(
            state.query,
            state,
            latest,
            answer_llm,
            uncertainty=terminal_uncertainty,
        )
        model = getattr(answer_llm, "model", None)
    return OrchestrationResult(
        final_status,
        analysis,
        state,
        uncertainty=terminal_uncertainty,
        answer=answer,
        model=model,
    )


def _interactive_clarification(requests: tuple[ClarificationRequest, ...]) -> Mapping[str, str]:
    answers: dict[str, str] = {}
    for request in requests:
        print(request.question)
        answers[request.target] = input("> ").strip()
    return answers


def _parse_missing_fact_values(values: Sequence[str]) -> dict[str, tuple[str, ...]]:
    parsed: dict[str, list[str]] = {}
    for value in values:
        step_id, separator, fact = value.partition("=")
        if not separator or not step_id.strip() or not fact.strip():
            raise ValueError("Missing-fact arguments must use STEP_ID=FACT_DESCRIPTION.")
        parsed.setdefault(step_id.strip(), []).append(fact.strip())
    return {step_id: tuple(facts) for step_id, facts in parsed.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run bounded adaptive legal reasoning orchestration.")
    parser.add_argument("--query", required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--max-retrieval-attempts", type=int, default=2)
    parser.add_argument("--max-clarification-attempts", type=int, default=1)
    parser.add_argument("--max-candidate-k", type=int, default=MAX_CANDIDATE_K)
    parser.add_argument("--model", help="Optional local embedding model override.")
    parser.add_argument("--answer-model", default=DEFAULT_MODEL)
    parser.add_argument("--generate-answer", action="store_true", help="Use Groq to phrase the final evidence-grounded response.")
    parser.add_argument("--interactive-clarification", action="store_true", help="Prompt for explicitly configured missing user facts.")
    parser.add_argument(
        "--missing-user-fact",
        action="append",
        default=[],
        metavar="STEP_ID=FACT",
        help="Explicitly mark a fact for clarification; repeat as needed (for example step_1=whether the act was intentional).",
    )
    parser.add_argument("--bm25-index", type=Path, default=Path("data/processed/bm25_index.pkl"))
    parser.add_argument("--dense-database", type=Path, default=Path("chroma_db"))
    args = parser.parse_args()

    missing_facts = _parse_missing_fact_values(args.missing_user_fact)
    if args.interactive_clarification and not missing_facts:
        parser.error("--interactive-clarification requires at least one --missing-user-fact STEP_ID=FACT.")

    llm = GroqResponsesLLM(args.answer_model) if args.generate_answer else None
    result = orchestrate(
        args.query,
        missing_user_facts=missing_facts,
        clarification_handler=_interactive_clarification if args.interactive_clarification else None,
        answer_llm=llm,
        top_k=args.top_k,
        candidate_k=args.candidate_k,
        max_candidate_k=args.max_candidate_k,
        max_retrieval_attempts=args.max_retrieval_attempts,
        max_clarification_attempts=args.max_clarification_attempts,
        bm25_index=args.bm25_index,
        dense_database=args.dense_database,
        model_name=args.model,
    )
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
