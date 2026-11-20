"""Free tests for the triage agent. They use a scripted fake client, so no API calls."""
from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from app.triage_agent import MAX_STEPS, lookup_policy, triage_ticket


class FakeMessages:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeClient:
    def __init__(self, responses):
        self.messages = FakeMessages(responses)


def block(type_, **fields):
    return SimpleNamespace(type=type_, **fields)


def tool_use(name, tool_input, id_):
    return block("tool_use", name=name, input=tool_input, id=id_)


def reply(stop_reason, content):
    return SimpleNamespace(stop_reason=stop_reason, content=content)


def submit(category, text="Reply", id_="s"):
    return reply("tool_use", [tool_use("submit_triage", {"category": category, "reply": text}, id_)])


def test_lookup_policy_returns_section_text():
    assert "30 days" in lookup_policy(1)


def test_lookup_policy_rejects_unknown_section():
    with pytest.raises(ValueError):
        lookup_policy(99)


def test_lookup_then_submit_records_calls_and_category():
    client = FakeClient([
        reply("tool_use", [tool_use("lookup_policy", {"section": 1}, "a")]),
        submit("return_request"),
    ])
    result = triage_ticket("Can I return my headphones?", client=client)

    assert result.error is None
    assert result.category == "return_request"
    assert [c["name"] for c in result.tool_calls] == ["lookup_policy", "submit_triage"]
    # the tool result goes back to Claude with the matching tool_use id
    # messages[2] is the tool_result user turn sent before the second call
    tool_result = client.messages.calls[1]["messages"][2]["content"][0]
    assert tool_result["tool_use_id"] == "a"
    assert "30 days" in tool_result["content"]


def test_bad_section_is_sent_back_as_error_and_loop_continues():
    client = FakeClient([
        reply("tool_use", [tool_use("lookup_policy", {"section": 99}, "a")]),
        submit("needs_more_info"),
    ])
    result = triage_ticket("Question", client=client)

    assert result.error is None
    tool_result = client.messages.calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "No policy section 99" in tool_result["content"]


def test_escalation_sets_flag():
    client = FakeClient([
        reply("tool_use", [tool_use("escalate_to_human", {"reason": "legal threat"}, "e")]),
        submit("out_of_scope"),
    ])
    result = triage_ticket("I will sue you", client=client)

    assert result.escalated is True
    assert result.category == "out_of_scope"


def test_invalid_category_is_reported_as_error():
    client = FakeClient([submit("pizza")])
    result = triage_ticket("Anything", client=client)

    assert result.category is None
    assert "invalid category" in result.error


def test_stops_after_max_steps_without_submit():
    looping = [reply("tool_use", [tool_use("lookup_policy", {"section": 1}, f"t{i}")])
               for i in range(MAX_STEPS)]
    result = triage_ticket("Loop forever", client=FakeClient(looping))

    assert result.category is None
    assert f"no submit_triage after {MAX_STEPS} steps" in result.error


def test_refusal_is_reported_as_error():
    client = FakeClient([reply("refusal", [])])
    result = triage_ticket("Something unsafe", client=client)

    assert result.error == "stopped: refusal"


def test_text_only_answer_without_submit_is_error():
    client = FakeClient([reply("end_turn", [block("text", text="Hello")])])
    result = triage_ticket("Hi", client=client)

    assert result.error == "model finished without calling submit_triage"


def _request():
    return httpx2.Request("POST", "https://api.example.test/v1/messages")


def test_connection_error_is_reported_not_raised():
    client = FakeClient([anthropic.APIConnectionError(request=_request())])
    result = triage_ticket("Anything", client=client)

    assert result.category is None
    assert result.error == "API error: APIConnectionError"


def test_timeout_is_reported_not_raised():
    client = FakeClient([anthropic.APITimeoutError(request=_request())])
    result = triage_ticket("Anything", client=client)

    assert result.error == "API error: APITimeoutError"


def test_empty_reply_is_error():
    client = FakeClient([submit("return_request", text="   ")])
    result = triage_ticket("Anything", client=client)

    assert result.category is None
    assert result.error == "empty reply"


def test_wrong_tool_input_type_is_sent_back_and_loop_continues():
    client = FakeClient([
        reply("tool_use", [tool_use("lookup_policy", {"section": "one"}, "a")]),
        submit("needs_more_info"),
    ])
    result = triage_ticket("Question", client=client)

    assert result.error is None
    tool_result = client.messages.calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True


def test_unknown_tool_is_sent_back_and_loop_continues():
    client = FakeClient([
        reply("tool_use", [tool_use("delete_ticket", {"id": 1}, "a")]),
        submit("needs_more_info"),
    ])
    result = triage_ticket("Question", client=client)

    assert result.error is None
    assert result.category == "needs_more_info"
    tool_result = client.messages.calls[1]["messages"][2]["content"][0]
    assert tool_result["is_error"] is True
    assert "Unknown tool: delete_ticket" in tool_result["content"]
