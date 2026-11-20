"""Paid tests: the triage agent on data/agent_dataset.json (calls the API and the judge)."""
import json
from pathlib import Path

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from app.judge import HaikuJudge
from app.triage_agent import POLICY_PATH, triage_ticket

DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "agent_dataset.json"
DATASET = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
CASE_IDS = [case["id"] for case in DATASET]
POLICY_TEXT = POLICY_PATH.read_text(encoding="utf-8")

JUDGE = HaikuJudge()


@pytest.mark.parametrize("case", DATASET, ids=CASE_IDS)
def test_triage_case(case):
    result = triage_ticket(case["input"])

    # 1. the run must finish with a valid category
    assert result.error is None, result.error
    assert result.category == case["expected_category"]

    # 2. tool choice
    called = {c["name"] for c in result.tool_calls}
    for name in case["required_tools"]:
        assert name in called, f"expected tool {name}, called {called}"
    for name in case["forbidden_tools"]:
        assert name not in called, f"tool {name} must not be called"

    # 3. tool arguments: required policy sections were looked up
    looked_up = {c["input"]["section"] for c in result.tool_calls if c["name"] == "lookup_policy"}
    for section in case["expected_sections"]:
        assert section in looked_up, f"policy section {section} not looked up: {looked_up}"

    # 4. escalation decision
    if case["expected_escalated"] is not None:
        assert result.escalated == case["expected_escalated"]

    # 5. reply quality, judged against the policy text
    metric = GEval(
        name="Triage reply accuracy",
        criteria=(
            "Check that the reply meets these criteria: "
            + "; ".join(case["criteria"])
            + ". The reply must not invent rules, fees, or programs that are not in the policy."
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
        actual_output=result.reply,
        expected_output=case["expected_reply"],
        retrieval_context=[POLICY_TEXT],
    )
    assert_test(test_case, [metric], run_async=False)
