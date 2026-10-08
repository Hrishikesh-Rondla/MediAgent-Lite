"""Schemas for PubMed evidence records."""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class PublicationType(str, Enum):
    """Evidence hierarchy classification.

    Based on Oxford Centre for Evidence-Based Medicine (CEBM) levels.
    Higher = stronger evidence. Used to compute weighted evidence scores.
    """

    GUIDELINE = "guideline"
    SYSTEMATIC_REVIEW = "systematic_review"
    META_ANALYSIS = "meta_analysis"
    RCT = "rct"
    COHORT = "cohort"
    CASE_CONTROL = "case_control"
    CASE_REPORT = "case_report"
    OTHER = "other"


class PubMedAbstract(BaseModel):
    """A parsed PubMed abstract."""

    pmid: str = Field(description="PubMed ID")
    title: str
    abstract: str
    authors: list[str] = Field(default_factory=list)
    year: Optional[int] = Field(default=None)
    publication_types: list[str] = Field(
        default_factory=list,
        description="Raw publication type strings from PubMed XML",
    )
    classified_type: PublicationType = Field(
        default=PublicationType.OTHER,
        description="Classified evidence type (used for weighting)",
    )
    journal: Optional[str] = Field(default=None)


class EvidenceRecord(BaseModel):
    """Evidence for a single hypothesis from one PubMed abstract."""

    hypothesis: str = Field(description="Diagnosis this evidence pertains to")
    pmid: str
    title: str
    evidence_weight: float = Field(
        ge=0.0,
        le=1.0,
        description="Weight based on publication type hierarchy",
    )
    classified_type: PublicationType
    relevance_summary: Optional[str] = Field(
        default=None,
        description="LLM-generated 2-3 sentence summary of relevance",
    )
    year: Optional[int] = Field(default=None)


class EvidenceBundle(BaseModel):
    """All evidence records for all hypotheses, output of EvidenceScanner."""

    records: list[EvidenceRecord] = Field(default_factory=list)
    weighted_scores: dict[str, float] = Field(
        default_factory=dict,
        description="Per-hypothesis weighted evidence score (0-1)",
    )
