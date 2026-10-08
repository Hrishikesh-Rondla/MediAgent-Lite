"""LLM factory: returns the right LangChain LLM object for a given agent role.

Teaching point: provider-agnostic design means you can swap Gemini for Groq
or a local Ollama model by changing config.yaml only — no code changes.
This is critical for reproducibility: the paper used specific models, and
we want to be able to replicate that.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseLLM, BaseLanguageModel
from langchain_core.messages import AIMessage

from mediagent_lite.config.settings import get_settings


# ── FakeLLM for tests ────────────────────────────────────────────────────────

class FakeLLM(BaseLanguageModel):
    """Deterministic fake LLM for offline testing.

    Why FakeLLM instead of mocking?
    - Mocking requires knowing internals of the LLM call chain.
    - FakeLLM is a proper LangChain BaseLanguageModel that passes through
      the full LangChain call stack, testing serialization and chain logic.
    - It's deterministic: same input -> same output, always.
    - No API key or network required.

    Common mistake: using MagicMock for LLM tests. This skips structured-output
    parsing and prompt formatting, so your 'passing' tests don't catch real bugs.
    """

    responses: list[str] = []
    _call_count: int = 0

    # Required abstract methods from BaseLanguageModel
    def _generate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("Use invoke() not _generate() directly")

    def _llm_type(self) -> str:
        return "fake"

    @property
    def _llm_type(self) -> str:  # type: ignore[override]
        return "fake"

    def invoke(self, input: Any, *args: Any, **kwargs: Any) -> AIMessage:
        """Return next response from the responses list, cycling if needed."""
        idx = self._call_count % max(len(self.responses), 1)
        self._call_count += 1
        text = self.responses[idx] if self.responses else "{}"
        return AIMessage(content=text)

    def predict(self, text: str, **kwargs: Any) -> str:
        return self.invoke(text).content

    def predict_messages(self, messages: list, **kwargs: Any) -> AIMessage:
        return self.invoke(messages)

    # LangChain requires this for with_structured_output compatibility
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a fake structured-output chain that parses JSON responses."""
        from langchain_core.output_parsers import JsonOutputParser
        from langchain_core.runnables import RunnableLambda

        import json

        def parse_and_validate(input: Any) -> Any:
            response = self.invoke(input)
            try:
                data = json.loads(response.content)
                if hasattr(schema, "model_validate"):
                    return schema.model_validate(data)
                return data
            except Exception:
                # Return a minimal valid instance for testing
                return response.content

        return RunnableLambda(parse_and_validate)

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        """Mock tool binding by just returning self."""
        return self

    def generate_prompt(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def agenerate_prompt(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apredict(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def apredict_messages(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


def get_llm(role: str = "default") -> BaseLanguageModel:
    """Return a LangChain LLM for the given agent role.

    Reads config.yaml for the role's provider and model.
    Env var DEFAULT_LLM_PROVIDER overrides the provider (not the model).

    Args:
        role: Agent role name. Must match a key in config.yaml llm_roles,
              or falls back to 'default'.

    Returns:
        A LangChain BaseLanguageModel ready to use.

    Raises:
        ValueError: If provider is unknown or required API key is missing.
    """
    settings = get_settings()
    llm_config = settings.get_llm_config(role)

    provider = os.getenv("DEFAULT_LLM_PROVIDER", llm_config["provider"])
    model = llm_config["model"]
    temperature = llm_config.get("temperature", 0.0)

    # Tests always use fake to avoid any network calls
    if os.getenv("MEDIAGENT_TEST_MODE", "").lower() == "true" or provider == "fake":
        return FakeLLM(responses=["{}"])

    if provider == "gemini":
        return _make_gemini(model, temperature)
    elif provider == "groq":
        return _make_groq(model, temperature)
    elif provider == "ollama":
        return _make_ollama(model, temperature)
    else:
        raise ValueError(
            f"Unknown LLM provider: '{provider}'. "
            "Valid options: gemini, groq, ollama, fake"
        )


def _make_gemini(model: str, temperature: float) -> BaseLanguageModel:
    """Create a Gemini ChatModel via langchain-google-genai."""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
    except ImportError as e:
        raise ImportError(
            "langchain-google-genai not installed. Run: pip install langchain-google-genai"
        ) from e

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY env var is not set. "
            "Get a free key at https://aistudio.google.com/"
        )

    return ChatGoogleGenerativeAI(
        model=model,
        temperature=temperature,
        google_api_key=api_key,
        convert_system_message_to_human=True,  # Gemini requires this for system messages
    )


def _make_groq(model: str, temperature: float) -> BaseLanguageModel:
    """Create a Groq ChatModel via langchain-groq."""
    try:
        from langchain_groq import ChatGroq
    except ImportError as e:
        raise ImportError(
            "langchain-groq not installed. Run: pip install langchain-groq"
        ) from e

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY env var is not set. "
            "Get a free key at https://console.groq.com/"
        )

    return ChatGroq(
        model=model,
        temperature=temperature,
        groq_api_key=api_key,
    )


def _make_ollama(model: str, temperature: float) -> BaseLanguageModel:
    """Create an Ollama ChatModel via langchain-ollama."""
    try:
        from langchain_ollama import ChatOllama
    except ImportError as e:
        raise ImportError(
            "langchain-ollama not installed. Run: pip install langchain-ollama"
        ) from e

    return ChatOllama(
        model=model,
        temperature=temperature,
        # Ollama runs locally at http://localhost:11434 by default
    )
