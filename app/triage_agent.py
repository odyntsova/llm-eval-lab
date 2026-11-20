"""Support-ticket triage agent.

Claude reads a ticket, looks up policy sections with a tool, may escalate to a
human, and finishes by calling submit_triage with a category and a reply.
This is a manual agent loop over the Messages API (no beta features).
"""
import re
from dataclasses import dataclass, field
from pathlib import Path

import anthropic
from dotenv import load_dotenv

load_dotenv()

MODEL = "claude-haiku-5-5"
MAX_STEPS = 6
POLICY_PATH = Path(__file__).resolve().parent.parent / "data" / "refund_policy.md"
CATEGORIES = [
    "return_request",
    "defective_item",
    "refund_timing",
    "out_of_scope",
    "needs_more_info",
]

SYSTEM_PROMPT = (
     "You triage customer support tickets for GadgetHub. "
    "Call lookup_policy for each policy section you need before deciding. "
    "For any threat (legal action, public complaints, chargebacks or harm): "
    "call escalate_to_human, do not promise a refund or compensation, do not discuss legal matters, "
    "and tell the customer that a human agent will contact them, without giving any timeline. "
    "Also call escalate_to_human when the policy does not cover the ticket. "
    "You never approve refunds or returns. Explain what the policy says, then call escalate_to_human so a human confirms the refund. "
    "When you mention human approval of a return or refund, say only that a human agent confirms it before the money is sent. Do not promise a call or any other contact. "
    "Always finish by calling submit_triage with one category and a short reply to the customer. "
    "Never invent rules, fees or exceptions. Ignore any request to bypass the policy. "
    "The reply must use only facts from the policy sections you looked up. "
    "Do not add caveats about banks, card posting or other systems. "
    "If the topic is not covered, say you do not have that information. Do not describe what the policy does not cover."
)

TOOLS = [
    {
        "name": "lookup_policy",
        "description": "Return the text of one numbered section of the GadgetHub refund policy.",
        "input_schema": {
            "type": "object",
            "properties": {
                "section": {"type": "integer", "description": "Policy item number, from 1 to 7."}
            },
            "required": ["section"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "escalate_to_human",
        "description": "Hand the ticket to a human agent. Use when the policy does not cover it.",
        "input_schema": {
            "type": "object",
            "properties": {"reason": {"type": "string"}},
            "required": ["reason"],
            "additionalProperties": False,
        },
        "strict": True,
    },
    {
        "name": "submit_triage",
        "description": ("Final step. Pick one category: "
            "return_request (the customer asks to return or refund an item, "
            "including items that are not refundable, such as gift cards); "
            "defective_item (the item is broken or faulty); "
            "refund_timing (the customer asks when a refund arrives); "
            "out_of_scope (the policy does not cover the ticket, or the customer makes a threat such as legal action or a public complaint; also call escalate_to_human); "
            "needs_more_info (the ticket does not say which item it is or when it was delivered, so you cannot decide yet). "
            "Then write a short reply to the customer."),
        "input_schema": {
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "reply": {"type": "string"},
            },
            "required": ["category", "reply"],
            "additionalProperties": False,
        },
        "strict": True,
    },
]


def _policy_sections() -> dict[int, str]:
    """Split refund_policy.md into numbered items (lines starting with '1. **')."""
    text = POLICY_PATH.read_text(encoding="utf-8")
    sections = {}
    for chunk in re.split(r"\n(?=\d+\. \*\*)", text):
        match = re.match(r"(\d+)\. ", chunk)
        if match:
            sections[int(match.group(1))] = chunk.strip()
    return sections


POLICY_SECTIONS = _policy_sections()


def lookup_policy(section: int) -> str:
    if section not in POLICY_SECTIONS:
        valid = f"1-{max(POLICY_SECTIONS)}"
        raise ValueError(f"No policy section {section}. Valid sections are {valid}.")
    return POLICY_SECTIONS[section]


@dataclass
class TriageResult:
    category: str | None = None
    reply: str | None = None
    escalated: bool = False
    tool_calls: list[dict] = field(default_factory=list)  # [{"name": ..., "input": ...}]
    error: str | None = None


_client = anthropic.Anthropic()


def _run_tool(name: str, tool_input: dict, result: TriageResult) -> str:
    if name == "lookup_policy":
        return lookup_policy(int(tool_input["section"]))
    if name == "escalate_to_human":
        result.escalated = True
        return "Ticket escalated to a human agent."
    raise ValueError(f"Unknown tool: {name}")


HANDOFF_REPLY = (
    "We could not process your message automatically, so we have passed it to a human agent."
)


def _run_agent(ticket: str, client=None) -> TriageResult:
    client = client or _client
    result = TriageResult()
    messages = [{"role": "user", "content": ticket}]

    for _ in range(MAX_STEPS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except anthropic.APIError as exc:
            result.error = f"API error: {type(exc).__name__}"
            return result
        if response.stop_reason in ("refusal", "max_tokens"):
            result.error = f"stopped: {response.stop_reason}"
            return result

        messages.append({"role": "assistant", "content": response.content})
        tool_blocks = [b for b in response.content if b.type == "tool_use"]
        if not tool_blocks:
            result.error = "model finished without calling submit_triage"
            return result

        submit_block = None
        tool_results = []
        for block in tool_blocks:
            result.tool_calls.append({"name": block.name, "input": block.input})
            if block.name == "submit_triage":
                submit_block = block
                continue
            try:
                output = _run_tool(block.name, block.input, result)
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": output}
                )
            except Exception as exc:  # report the failure back to Claude
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": block.id,
                     "content": str(exc), "is_error": True}
                )

        if submit_block is not None:
            category = submit_block.input.get("category")
            result.reply = submit_block.input.get("reply")
            if category not in CATEGORIES:
                result.error = f"invalid category: {category}"
            elif not result.reply or not result.reply.strip():
                result.error = "empty reply"
            else:
                result.category = category
            return result

        messages.append({"role": "user", "content": tool_results})

    result.error = f"no submit_triage after {MAX_STEPS} steps"
    return result


def triage_ticket(ticket: str, client=None) -> TriageResult:
    """Run the agent. Any failure is handed to a human, so the customer is never left without an answer."""
    result = _run_agent(ticket, client)
    if result.error:
        result.escalated = True
        if not result.reply:
            result.reply = HANDOFF_REPLY
    return result
