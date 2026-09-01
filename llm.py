"""The only provider-specific code. No tools, browsing, or persistent threads."""

from typing import Protocol

from openai import AsyncOpenAI

from config import Settings


class LLM(Protocol):
    async def generate(self, system_prompt: str, context: str) -> str: ...

    async def close(self) -> None: ...


class OpenAILLM:
    def __init__(self, api_key: str, model: str, reasoning_effort: str | None = None):
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.client = AsyncOpenAI(api_key=api_key, timeout=30.0, max_retries=0)

    async def generate(self, system_prompt: str, context: str) -> str:
        request = dict(
            model=self.model,
            instructions=system_prompt,
            input=context,
            max_output_tokens=400,
            store=False,
        )
        if self.reasoning_effort is not None:
            request["reasoning"] = {"effort": self.reasoning_effort}
        response = await self.client.responses.create(**request)
        if response.status != "completed":
            # Don't post a partial answer (including a truncated SILENT decision).
            raise RuntimeError("LLM response did not complete.")
        if not response.output_text.strip():
            raise RuntimeError("LLM returned no text.")
        return response.output_text

    async def close(self) -> None:
        await self.client.close()


def create_llm(settings: Settings) -> LLM:
    if settings.llm_provider == "openai":
        return OpenAILLM(
            settings.llm_api_key, settings.llm_model, settings.llm_reasoning_effort
        )
    raise ValueError("Unsupported LLM provider.")
