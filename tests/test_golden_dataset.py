import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.models import AnthropicModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.policy_bot import ask_policy_bot

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "golden_dataset.json"
DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
CASE_IDS = [case["id"] for case in DATASET]

POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "refund_policy.md"
POLICY_TEXT = POLICY_PATH.read_text(encoding="utf-8")

# Haiku as the judge. The cost values are only for DeepEval's cost tracking.
JUDGE = AnthropicModel(
    model="claude-haiku-5-5",
    cost_per_input_token=1e-7,
    cost_per_output_token=5e-7,
)


@pytest.mark.parametrize("case", DATASET, ids=CASE_IDS)
def test_must_contain(case):
    answer = ask_policy_bot(case["input"])
    for word in case["must_contain"]:
        assert word in answer, f"Missing '{word}' in answer: {answer}"


@pytest.mark.parametrize("case", DATASET, ids=CASE_IDS)
def test_answer_meets_criteria(case):
    metric = GEval(
        name="Policy accuracy",
        criteria=(
            "Check that the actual output meets these criteria: "
            + "; ".join(case["criteria"])
            + ". The output must not invent rules, fees, or programs that are not in the policy."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
            
        ],
        model=JUDGE,
        threshold=0.7,
    )
    test_case = LLMTestCase(
        input=case["input"],
        actual_output=ask_policy_bot(case["input"]),
        expected_output=case["expected_output"],
        retrieval_context=[POLICY_TEXT],
    )
    assert_test(test_case, [metric], run_async=False)