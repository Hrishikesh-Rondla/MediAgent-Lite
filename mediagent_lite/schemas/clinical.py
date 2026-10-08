"""Pydantic v2 schemas for structured clinical case representation.

Why Pydantic for agent I/O?
- Structured output from LLMs is unreliable; Pydantic catches bad outputs
  at the boundary so bad data never propagates downstream.
- Enables JSON schema generation for LangChain's with_structured_output().
"""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, model_validator


class Symptom(BaseModel):
    """A single symptom with optional anatomical and temporal context."""

    name: str = Field(description="Symptom name, e.g. 'chest pain'")
    anatomical_region: Optional[str] = Field(
        default=None,
        description="Body region, e.g. 'left lower quadrant'",
    )
    duration: Optional[str] = Field(
        default=None,
        description="Duration, e.g. '3 days', '2 hours'",
    )
    severity: Optional[str] = Field(
        default=None,
        description="Severity descriptor: mild / moderate / severe",
    )
    onset: Optional[str] = Field(
        default=None,
        description="Onset pattern: sudden / gradual / intermittent",
    )


class Lab(BaseModel):
    """A lab result with optional reference range and interpretation."""

    name: str = Field(description="Lab test name, e.g. 'WBC'")
    value: str = Field(description="Result value with units, e.g. '12.5 K/uL'")
    interpretation: Optional[str] = Field(
        default=None,
        description="high / low / normal / critical",
    )


class Imaging(BaseModel):
    """An imaging study result."""

    modality: str = Field(description="e.g. 'CT chest', 'X-ray abdomen'")
    finding: str = Field(description="Key radiological finding")


class ClinicalCase(BaseModel):
    """Raw clinical case as entered by the user. Intentionally loose schema."""

    text: str = Field(description="Raw clinical case text")


class StructuredCase(BaseModel):
    """Structured representation extracted from raw clinical text.

    Produced by ClinicalTextClarifier. All fields are optional because
    real clinical notes are incomplete — we never hallucinate missing data.
    """

    # Patient demographics
    age: Optional[int] = Field(default=None, description="Patient age in years")
    sex: Optional[str] = Field(default=None, description="Patient sex: M / F / other")
    chief_complaint: Optional[str] = Field(
        default=None, description="Primary reason for visit"
    )

    # Clinical findings
    symptoms: list[Symptom] = Field(default_factory=list)
    labs: list[Lab] = Field(default_factory=list)
    imaging: list[Imaging] = Field(default_factory=list)

    # History
    past_medical_history: list[str] = Field(
        default_factory=list,
        description="Relevant past diagnoses",
    )
    medications: list[str] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    family_history: list[str] = Field(default_factory=list)
    social_history: Optional[str] = Field(default=None)

    # Extracted codes (ICD-10 only if found in local table)
    icd10_codes: list[str] = Field(
        default_factory=list,
        description="ICD-10 codes matched from local lookup table. Empty if no match.",
    )

    # Vital signs
    vitals: dict[str, str] = Field(
        default_factory=dict,
        description="e.g. {'BP': '140/90', 'HR': '95', 'Temp': '38.5C'}",
    )

    @model_validator(mode="after")
    def must_have_some_clinical_data(self) -> StructuredCase:
        """Require at least a chief complaint or one symptom.

        Why: an empty structured case will produce meaningless hypotheses
        and waste API calls. Fail early at the boundary.
        """
        if not self.chief_complaint and not self.symptoms:
            raise ValueError(
                "StructuredCase must have at least a chief_complaint or one symptom. "
                "The input may be too vague or non-clinical."
            )
        return self
