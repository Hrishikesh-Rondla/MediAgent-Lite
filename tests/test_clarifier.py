"""Tests for ClinicalTextClarifier."""

import pytest
from unittest.mock import patch

from mediagent_lite.agents.clarifier import ClinicalTextClarifier
from mediagent_lite.schemas.clinical import StructuredCase

# We need a FakeLLM that returns a valid JSON matching StructuredCase
FAKE_CLARIFIER_RESPONSE = """
{
    "age": 45,
    "sex": "M",
    "chief_complaint": "chest pain",
    "symptoms": [
        {
            "name": "chest pain",
            "anatomical_region": "left chest",
            "severity": "severe",
            "onset": "sudden"
        }
    ],
    "labs": [],
    "imaging": [],
    "past_medical_history": ["Hypertension"],
    "medications": [],
    "allergies": [],
    "family_history": [],
    "vitals": {},
    "icd10_codes": []
}
"""

def test_clarifier_initialization():
    """Ensure it initializes and loads ICD-10 table."""
    agent = ClinicalTextClarifier()
    assert isinstance(agent.icd10_table, dict)
    
    # Check if table loaded properly (depends on data/icd10_codes.csv being present)
    if "chest pain, unspecified" in agent.icd10_table:
        assert agent.icd10_table["chest pain, unspecified"] == "R07.9"

@patch("mediagent_lite.llm_factory.FakeLLM.invoke")
def test_clarifier_process(mock_invoke):
    """Test text processing flow."""
    from langchain_core.messages import AIMessage
    mock_invoke.return_value = AIMessage(content=FAKE_CLARIFIER_RESPONSE)
    
    agent = ClinicalTextClarifier()
    # Mock the lookup table for the test
    agent.icd10_table = {"chest pain": "R07.9", "hypertension": "I10"}
    
    result = agent.process("45yo M c/o sudden severe left chest pain. PMH: Hypertension.")
    
    assert isinstance(result, StructuredCase)
    assert result.age == 45
    assert result.chief_complaint == "chest pain"
    assert len(result.symptoms) == 1
    assert result.symptoms[0].severity == "severe"
    
    # Check ICD-10 lookup
    assert "R07.9" in result.icd10_codes
    assert "I10" in result.icd10_codes
