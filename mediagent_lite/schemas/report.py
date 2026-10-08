"""Schemas for the fused clinical report."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class ConflictFlag(BaseModel):
    """Flags a conflict between RAG and web evidence for a diagnosis."""

    diagnosis: str
    rag_stance: str = Field(description="RAG evidence stance on this diagnosis")
    web_stance: str = Field(description="Web evidence stance on this diagnosis")
    description: str = Field(description="Human-readable conflict description")


class DiagnosisReport(BaseModel):
    """Fused report for a single diagnosis candidate."""

    diagnosis: str
    icd10_code: Optional[str] = None
    rank: int = Field(ge=1)

    # The deterministic confidence score — NOT from LLM self-assessment
    confidence: float = Field(
        default=0.0,
        description=(
            "Deterministic score: w1*retrieval_sim + w2*evidence_score + w3*concordance. "
            "See config.yaml confidence section."
        ),
    )

    # Component scores for transparency
    retrieval_similarity: float = Field(default=0.0)
    weighted_evidence_score: float = Field(default=0.0)
    concordance: float = Field(
        default=0.0,
        description="1=agree, 0.5=neutral, 0=conflict between RAG and web",
    )

    # Sources (traceable)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    pubmed_ids: list[str] = Field(default_factory=list)

    # LLM explains the score — but does NOT set it
    explanation: Optional[str] = Field(
        default=None,
        description="LLM-written explanation of the evidence. Does not affect score.",
    )
    has_conflict: bool = False
    conflict_detail: Optional[str] = None

    @field_validator("confidence", mode="before")
    @classmethod
    def confidence_is_bounded(cls, v: float) -> float:
        """Clamp to [0, 1] in case of floating-point drift."""
        return max(0.0, min(1.0, v))


class FusedReport(BaseModel):
    """Complete fused report, output of ClinicalDataFusion."""

    diagnoses: list[DiagnosisReport] = Field(
        description="Ranked list of diagnoses with confidence scores"
    )
    conflicts: list[ConflictFlag] = Field(default_factory=list)
    top_confidence: float = Field(
        default=0.0,
        description="Confidence of the top-ranked diagnosis",
    )
    iteration: int = Field(ge=0, description="Which loop iteration produced this report")
    summary: Optional[str] = Field(
        default=None,
        description="LLM-generated overall summary of findings",
    )

    @field_validator("diagnoses")
    @classmethod
    def must_have_diagnoses(cls, v: list) -> list:
        if not v:
            raise ValueError("FusedReport must contain at least one diagnosis")
        return v
