import abc
import base64
import json
import os
import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]


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

    @abc.abstractmethod
    def generate_text(self, user: str, system: str = "") -> str:
        """Plain text generation without JSON parsing."""
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

    def generate_text_multimodal(self, user_text: str, images: list[bytes], system: str = "") -> str:
        """Vision-capable text generation. Default: silently drop images, call generate_text."""
        return self.generate_text(user_text, system)

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

    def generate_text(self, user: str, system: str = "") -> str:
        return "stub summary"


class Gemma3Client(LLMClient):
    def __init__(self):
        if OpenAI is None:
            raise ImportError("openai package is required: pip install openai")
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

    def generate_text(self, user: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat.completions.create(
            model="/model",
            messages=messages,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()

    def generate_text_multimodal(self, user_text: str, images: list[bytes], system: str = "") -> str:
        if not images:
            return self.generate_text(user_text, system)
        user_content: list[dict] = [{"type": "text", "text": user_text}]
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        response = self._client.chat.completions.create(
            model="/model",
            messages=messages,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()


class GPT41Client(LLMClient):
    def __init__(self):
        from azure.identity import DefaultAzureCredential, get_bearer_token_provider
        self._token_provider = get_bearer_token_provider(
            DefaultAzureCredential(), "https://ai.azure.com/.default"
        )
        self._endpoint = os.environ.get(
            "AZURE_OPENAI_ENDPOINT",
            "https://aif-futago-dev-eus2-01.services.ai.azure.com/openai/v1",
        )
        self._deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT", "gpt-4.1")

    def _make_client(self):
        if OpenAI is None:
            raise ImportError("openai package is required: pip install openai")
        return OpenAI(base_url=self._endpoint, api_key=self._token_provider())

    def generate(self, system: str, user: str) -> dict:
        response = self._make_client().chat.completions.create(
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
        response = self._make_client().chat.completions.create(**kwargs)
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

    def generate_text(self, user: str, system: str = "") -> str:
        _, text = self.generate_with_tools(
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": "user", "content": user}],
            [],
        )
        return (text or "").strip()

    def generate_text_multimodal(self, user_text: str, images: list[bytes], system: str = "") -> str:
        if not images:
            return self.generate_text(user_text, system)
        user_content: list[dict] = [{"type": "text", "text": user_text}]
        for img_bytes in images:
            b64 = base64.b64encode(img_bytes).decode()
            user_content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
            })
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user_content})
        response = self._make_client().chat.completions.create(
            model=self._deployment,
            messages=messages,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()


class Gemma4Client(LLMClient):
    def __init__(self):
        if OpenAI is None:
            raise ImportError("openai package is required: pip install openai")
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

    def generate_text(self, user: str, system: str = "") -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        response = self._client.chat.completions.create(
            model="/model",
            messages=messages,
            temperature=0.0,
        )
        return (response.choices[0].message.content or "").strip()


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
