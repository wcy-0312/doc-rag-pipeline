import abc
import json
import os
import re
from typing import Optional


def _parse_json_response(raw: str) -> dict:
    # Strip markdown code fences that some models wrap around JSON output
    stripped = raw.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)
    return json.loads(stripped)


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, system: str, user: str) -> dict:
        ...

    def generate_multimodal(self, messages: list) -> dict:
        """Default: extract text from messages and fallback to generate()."""
        system = next((m["content"] for m in messages if m["role"] == "system"), "")
        user_parts = next((m["content"] for m in messages if m["role"] == "user"), "")
        if isinstance(user_parts, list):
            user = " ".join(p["text"] for p in user_parts if p.get("type") == "text")
        else:
            user = user_parts
        return self.generate(system, user)

    def generate_with_tools(self, messages: list, tools: list) -> tuple:
        """Override in subclasses that support function calling.

        Returns:
            (tool_calls, None)  — LLM wants to call tools
            ([], content_str)   — LLM gives final answer

        tool_calls: list of {"id": str, "name": str, "arguments": dict}
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support tool calling. Use GPT41Client."
        )


class _StubLLMClient(LLMClient):
    def generate(self, system: str, user: str) -> dict:
        return {
            "answer": "stub",
            "claims": [{"text": "stub claim", "citations": ["E1"]}],
            "abstain": False,
            "abstain_reason": None,
        }

    def generate_with_tools(self, messages: list, tools: list) -> tuple:
        content = json.dumps({
            "answer": "stub",
            "claims": [{"text": "stub claim", "citations": ["E1"]}],
            "abstain": False,
            "abstain_reason": None,
        })
        return ([], content)


class Gemma3Client(LLMClient):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key="not-needed",
            base_url="http://172.31.6.3:8080/gemma3/v1",
        )

    def generate(self, system: str, user: str) -> dict:
        response = self._client.chat.completions.create(
            model="/model",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        return _parse_json_response(raw)


class GPT41Client(LLMClient):
    def __init__(self):
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        from openai import AzureOpenAI
        credential = DefaultAzureCredential()
        token_provider = get_bearer_token_provider(
            credential, "https://cognitiveservices.azure.com/.default"
        )
        self._client = AzureOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT", ""),
            api_version="2024-12-01-preview",
        )
        self._deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    def generate(self, system: str, user: str) -> dict:
        response = self._client.chat.completions.create(
            model=self._deployment,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        return _parse_json_response(raw)

    def generate_with_tools(self, messages: list, tools: list) -> tuple:
        kwargs = {
            "model": self._deployment,
            "messages": messages,
            "temperature": 0.0,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"
        response = self._client.chat.completions.create(**kwargs)
        msg = response.choices[0].message
        if msg.tool_calls:
            tool_calls = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": json.loads(tc.function.arguments),
                }
                for tc in msg.tool_calls
            ]
            return (tool_calls, None)
        return ([], msg.content)


class Gemma4Client(LLMClient):
    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(
            api_key="not-needed",
            base_url="http://172.31.6.3:8080/gemma4/v1",
        )

    def generate(self, system: str, user: str) -> dict:
        response = self._client.chat.completions.create(
            model="/model",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.0,
        )
        raw = response.choices[0].message.content
        return _parse_json_response(raw)


def get_llm_client() -> LLMClient:
    backend = os.environ.get("GENERATION_LLM_BACKEND", "gemma3").lower()
    if backend == "gemma3":
        return Gemma3Client()
    elif backend == "gpt41":
        return GPT41Client()
    elif backend == "gemma4":
        return Gemma4Client()
    elif backend == "stub":
        return _StubLLMClient()
    else:
        raise ValueError(f"Unknown GENERATION_LLM_BACKEND: {backend}")
