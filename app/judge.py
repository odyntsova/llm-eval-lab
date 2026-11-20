"""Claude Haiku as a DeepEval judge.

DeepEval's built-in AnthropicModel fails on this setup: it expects an API
method that does not exist and sometimes gets judge output without a `score`.
This judge calls the Messages API directly, parses the JSON itself, and retries
when the output is malformed.
"""
import json
import re

import anthropic
from deepeval.models import DeepEvalBaseLLM

MODEL = "claude-haiku-5-5"
MAX_ATTEMPTS = 3


def _extract_json(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        raise ValueError("no JSON object in judge output")
    return json.loads(match.group(0))


class HaikuJudge(DeepEvalBaseLLM):
    def __init__(self):
        super().__init__(MODEL)

    def load_model(self):
        return anthropic.Anthropic()

    def _call(self, prompt: str) -> str:
        response = self.model.messages.create(
            model=MODEL,
            max_tokens=4000,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(block.text for block in response.content if block.type == "text")

    def generate(self, prompt: str, schema=None):
        if schema is None:
            return self._call(prompt)
        last_error = None
        for _ in range(MAX_ATTEMPTS):
            text = self._call(prompt)
            try:
                return schema.model_validate(_extract_json(text))
            except ValueError as exc:  # pydantic's ValidationError is a ValueError
                last_error = exc
        raise last_error

    async def a_generate(self, prompt: str, schema=None):
        # The calls are synchronous; the test suite is small enough for that.
        return self.generate(prompt, schema=schema)

    def get_model_name(self) -> str:
        return MODEL
