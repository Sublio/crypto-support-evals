import anthropic
from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

JUDGE_MODEL = "claude-haiku-5-5"


class ClaudeJudge(DeepEvalBaseLLM):
    """DeepEval judge backed by Claude via the official anthropic SDK.

    DeepEval's metrics pass a Pydantic schema; messages.parse() returns an
    instance of it, so the metric never has to parse free-text JSON.
    No temperature: Claude Haiku 5.5 rejects non-default sampling params."""

    def __init__(self, model: str = JUDGE_MODEL):
        self.client = anthropic.Anthropic()
        self.async_client = anthropic.AsyncAnthropic()
        super().__init__(model)

    def load_model(self):
        return self.client

    def get_model_name(self) -> str:
        return self.name

    def _request(self, prompt: str, schema: type[BaseModel] | None) -> dict:
        request = {
            "model": self.name,
            "max_tokens": 16000,
            "messages": [{"role": "user", "content": prompt}],
            # Grading is classification-shaped; low effort keeps runs fast and cheap.
            "output_config": {"effort": "low"},
        }
        if schema is not None:
            request["output_format"] = schema
        return request

    def _result(self, response, schema: type[BaseModel] | None):
        # Haiku has no server-side refusal fallback; fail loudly rather than score garbage.
        if response.stop_reason == "refusal":
            raise RuntimeError(f"Judge refused: {response.stop_details}")
        if schema is not None:
            return response.parsed_output
        return next(block.text for block in response.content if block.type == "text")

    def generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is not None:
            response = self.client.messages.parse(**self._request(prompt, schema))
        else:
            response = self.client.messages.create(**self._request(prompt, schema))
        return self._result(response, schema)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None):
        if schema is not None:
            response = await self.async_client.messages.parse(**self._request(prompt, schema))
        else:
            response = await self.async_client.messages.create(**self._request(prompt, schema))
        return self._result(response, schema)
