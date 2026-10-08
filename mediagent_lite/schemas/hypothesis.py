"""Schemas for RAG-derived diagnosis hypotheses."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class SupportingChunk(BaseModel):
    """A RAG chunk that supports a hypothesis.

    Why store the chunk ID and text span?
    Traceability: every hypothesis must be grounded in retrieved text.
    This prevents LLM hallucination from propagating as a 'supported' diagnosis.
    """

    chunk_id: str = Field(description="Unique chunk identifier from Qdrant")
    text_span: str = Field(
        description="The specific excerpt from the chunk that is relevant"
    )
    similarity_score: float = Field(
        ge=0.0, le=1.0, description="Cosine similarity score"
    )


class DiagnosisHypothesis(BaseModel):
    """A candidate diagnosis proposed by the RAG Analyzer.

    Key design choice: a hypothesis without a supporting chunk is rejected.
    This is enforced by the RAG Analyzer agent, not here (validation would
    block the schema, but rejection logic belongs in the agent).
    """

    diagnosis: str = Field(description="Diagnosis name")
    icd10_code: str | None = Field(
        default=None, description="ICD-10 code if known"
    )
    rank: int = Field(ge=1, description="Rank by likelihood (1 = most likely)")
    reasoning: str = Field(
        description="Brief reasoning for this hypothesis based on retrieved evidence"
    )
    supporting_chunks: list[SupportingChunk] = Field(
        default_factory=list,
        description="RAG chunks supporting this hypothesis",
    )

    @field_validator("rank")
    @classmethod
    def rank_must_be_positive(cls, v: int) -> int:
        if v < 1:
            raise ValueError("Rank must be >= 1")
        return v


class HypothesisList(BaseModel):
    """Ordered list of hypotheses from RAG Analyzer."""

    hypotheses: list[DiagnosisHypothesis] = Field(
        description="Ranked hypotheses, index 0 = most likely"
    )

    @field_validator("hypotheses")
    @classmethod
    def must_have_at_least_one(cls, v: list) -> list:
        if not v:
            raise ValueError("At least one hypothesis is required")
        return v
