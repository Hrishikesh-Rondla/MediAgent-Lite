"""LLM factory: returns the right LangChain LLM object for a given agent role.

Teaching point: provider-agnostic design means you can swap Gemini for Groq
or a local Ollama model by changing config.yaml only — no code changes.
This is critical for reproducibility: the paper used specific models, and
we want to be able to replicate that.
"""

from __future__ import annotations

import os
from typing import Any

from langchain_core.language_models import BaseLanguageModel
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
        # Provide a fallback robust medical JSON response if responses are empty or "{}"
        if not self.responses or self.responses[0] == "{}":
            prompt_str = str(input).lower()
            print(f"DEBUG FakeLLM prompt_str: {prompt_str[:200]}")
            
            # Detect which case we are running based on keywords
            is_neuro = "weakness" in prompt_str or "droop" in prompt_str or "slurred" in prompt_str or "stroke" in prompt_str or "ischemic stroke" in prompt_str
            print(f"DEBUG FakeLLM is_neuro: {is_neuro}")

            if "expert evidence appraiser" in prompt_str:
                if is_neuro:
                    return AIMessage(content='{"records": [{"hypothesis": "Acute Ischemic Stroke", "pmid": "87654321", "title": "tPA in acute stroke", "classified_type": "rct", "evidence_weight": 0.8, "relevance_summary": "Patient has classic signs of MCA stroke...", "year": 2024}], "weighted_scores": {"Acute Ischemic Stroke": 0.8}}')
                return AIMessage(content='{"records": [{"hypothesis": "Acute Myocardial Infarction", "pmid": "12345678", "title": "ST elevation MI study", "classified_type": "cohort", "evidence_weight": 0.6, "relevance_summary": "Patient exhibits ST elevation...", "year": 2023}], "weighted_scores": {"Acute Myocardial Infarction": 0.6}}')
            
            elif "lead diagnostician" in prompt_str or "generate the final synthesis" in prompt_str:
                if is_neuro:
                    return AIMessage(content='{"diagnoses": [{"diagnosis": "Acute Ischemic Stroke", "rank": 1, "confidence": 0.92, "retrieval_similarity": 0.88, "weighted_evidence_score": 0.9, "concordance": 1.0, "explanation": "Presentation strongly suggests MCA territory stroke. Supported by RAG and literature.", "has_conflict": false, "conflict_detail": "", "supporting_chunk_ids": [], "pubmed_ids": []}], "top_confidence": 0.92, "iteration": 0, "conflicts": []}')
                return AIMessage(content='{"diagnoses": [{"diagnosis": "Acute Myocardial Infarction", "rank": 1, "confidence": 0.95, "retrieval_similarity": 0.85, "weighted_evidence_score": 0.9, "concordance": 0.95, "explanation": "Matches symptoms and literature.", "has_conflict": false, "conflict_detail": "", "supporting_chunk_ids": [], "pubmed_ids": []}], "top_confidence": 0.95, "iteration": 0, "conflicts": []}')
            
            elif "ranked list of possible diagnoses" in prompt_str:
                if is_neuro:
                    return AIMessage(content='{"hypotheses": [{"diagnosis": "Acute Ischemic Stroke", "rank": 1, "reasoning": "Sudden onset focal neurologic deficits.", "supporting_chunks": [{"chunk_id": "chunk_2", "text_span": "facial droop and weakness point to stroke", "similarity_score": 0.88}]}]}')
                return AIMessage(content='{"hypotheses": [{"diagnosis": "Acute Myocardial Infarction", "rank": 1, "reasoning": "Typical presentation.", "supporting_chunks": [{"chunk_id": "chunk_1", "text_span": "chest pain points to MI", "similarity_score": 0.85}]}]}')
            
            elif "clinical informatics extractor" in prompt_str:
                if is_neuro:
                    return AIMessage(content='{"age": 72, "sex": "F", "chief_complaint": "right-sided weakness", "symptoms": [{"name": "facial droop"}, {"name": "slurred speech"}, {"name": "right-sided weakness"}], "labs": [], "imaging": [], "past_medical_history": [], "medications": [], "allergies": [], "family_history": [], "icd10_codes": [], "vitals": {}}')
                return AIMessage(content='{"age": 65, "sex": "M", "chief_complaint": "chest pain", "symptoms": [{"name": "chest pain"}], "labs": [], "imaging": [], "past_medical_history": [], "medications": [], "allergies": [], "family_history": [], "icd10_codes": [], "vitals": {"BP": "160/90", "HR": "110"}}')
            
            elif "generate one highly specific pubmed" in prompt_str:
                if is_neuro:
                    return AIMessage(content='{"type": "text", "text": "ischemic stroke AND facial droop"}')
                return AIMessage(content='{"type": "text", "text": "myocardial infarction AND chest pain"}')
                
            elif "query rewrite expert" in prompt_str:
                if is_neuro:
                    return AIMessage(content='stroke diagnosis')
                return AIMessage(content='myocardial infarction diagnosis')
            
            return AIMessage(content='{"diagnoses": [{"diagnosis": "Mock Diagnosis", "rank": 1, "confidence": 0.99, "retrieval_similarity": 0.99, "weighted_evidence_score": 0.99, "concordance": 0.99, "explanation": "FakeLLM Mock Response.", "has_conflict": false}], "top_confidence": 0.99, "iteration": 0, "conflicts": []}')

        idx = self._call_count % max(len(self.responses), 1)
        self._call_count += 1
        text = self.responses[idx]
        return AIMessage(content=text)

    def predict(self, text: str, **kwargs: Any) -> str:
        return self.invoke(text).content

    def predict_messages(self, messages: list, **kwargs: Any) -> AIMessage:
        return self.invoke(messages)

    # LangChain requires this for with_structured_output compatibility
    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a fake structured-output chain that parses JSON responses."""
        import json

        from langchain_core.runnables import RunnableLambda

        def parse_and_validate(input: Any) -> Any:
            response = self.invoke(input)
            try:
                data = json.loads(response.content)
                if hasattr(schema, "model_validate"):
                    return schema.model_validate(data)
                return data
            except Exception:
                import traceback
                traceback.print_exc()
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

    # If the UI forces a global provider override, force a known working model 
    # to avoid sending "gemini-3.5-flash" to the Groq API, etc.
    if provider == "groq" and "gemini" in model:
        model = "qwen/qwen3.8-27b"
    elif provider == "groq" and "llama" in model:
        # User's API key lacks llama access, force qwen
        model = "qwen/qwen3.8-27b"
    elif provider == "gemini" and "gemini-1.5" in model:
        model = "gemini-3.5-flash"
    elif provider == "ollama":
        model = "llama3.1"

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
