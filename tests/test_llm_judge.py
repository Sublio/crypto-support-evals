"""LLM-as-judge checks: is each substantive answer relevant to the question,
and is every claim in it supported by the retrieved context?

Run: uv run pytest tests/test_llm_judge.py   (needs Ollama + ANTHROPIC_API_KEY)
"""
import json
import os
from datetime import datetime

os.environ.setdefault("DEEPEVAL_TELEMETRY_OPT_OUT", "YES")

import pytest
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from crypto_support_evals.bot import FALLBACK, answer
from crypto_support_evals.judge import ClaudeJudge
from crypto_support_evals.run_evals import EVALS_FILE, RESULTS_DIR

# Starting points; milestone 4 calibrates them against the golden dataset.
RELEVANCY_THRESHOLD = 0.7
FAITHFULNESS_THRESHOLD = 0.7
RECALL_THRESHOLD = 0.7
PRECISION_THRESHOLD = 0.7


def load_cases() -> list:
    cases = []
    for line in EVALS_FILE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        # Same idea as XFAIL in run_evals: a known issue is reported, not a CI failure.
        # raises=AssertionError: only a low metric score counts; API/infra errors still fail.
        marks = [
            pytest.mark.xfail(reason=case["known_issue"], raises=AssertionError, strict=False)
        ] if case.get("known_issue") else []
        cases.append(pytest.param(case, id=case["id"], marks=marks))
    return cases


@pytest.fixture(scope="module")
def judge() -> ClaudeJudge:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")
    return ClaudeJudge()


@pytest.fixture(scope="module")
def score_log():
    rows = []
    yield rows
    run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_file = RESULTS_DIR / f"judge-{run_time}.json"
    out_file.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nSaved to {out_file}")


@pytest.mark.parametrize("case", load_cases())
def test_answer_is_relevant_and_faithful(case: dict, judge: ClaudeJudge,score_log:list) -> None:
    result = answer(case["question"])
    if result["answer"] == FALLBACK:
        pytest.skip("FALLBACK answer: graded exactly by run_evals, nothing for the judge to score")

    test_case = LLMTestCase(
        input=case["question"],
        actual_output=result["answer"],
        retrieval_context=result["contexts"],
        expected_output=case.get("reference"),
    )
    relevancy = AnswerRelevancyMetric(threshold=RELEVANCY_THRESHOLD, model=judge)
    faithfulness = FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=judge)
    relevancy.measure(test_case)
    faithfulness.measure(test_case)
    row = {"id": case["id"], 
           "answer": result["answer"],
           "relevancy": relevancy.score,
           "relevancy_reason": relevancy.reason,
           "faithfulness": faithfulness.score,
           "faithfulness_reason": faithfulness.reason
           }
    if test_case.expected_output:
        recall = ContextualRecallMetric(threshold=RECALL_THRESHOLD, model=judge)
        precision = ContextualPrecisionMetric(threshold=PRECISION_THRESHOLD, model=judge)
        recall.measure(test_case)
        precision.measure(test_case)
        row["recall"] = recall.score
        row["recall_reason"] = recall.reason
        row["precision"] = precision.score
        row["precision_reason"] = precision.reason
    score_log.append(row)
    metrics = [relevancy, faithfulness]
    if test_case.expected_output:
        metrics += [recall, precision]
    assert all(m.is_successful() for m in metrics), row
