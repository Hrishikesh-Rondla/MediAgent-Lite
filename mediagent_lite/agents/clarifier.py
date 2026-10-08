"""Clinical Text Clarifier Agent.

Responsibility: Take unstructured clinical text and extract a structured
case representation (demographics, symptoms, labs). Optional ICD-10 lookup.
"""

import csv
from pathlib import Path
from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from mediagent_lite.llm_factory import get_llm
from mediagent_lite.schemas.clinical import StructuredCase


class ClinicalTextClarifier:
    """Extracts structured data from raw clinical text."""

    def __init__(self):
        self.llm = get_llm("clarifier")
        
        # We use with_structured_output to force the LLM to return our Pydantic schema
        # This is a major teaching point: boundary validation prevents downstream errors.
        self.chain = self._build_prompt() | self.llm.with_structured_output(StructuredCase)
        
        self.icd10_table = self._load_icd10_table()

    def _build_prompt(self) -> ChatPromptTemplate:
        return ChatPromptTemplate.from_messages([
            ("system", """You are an expert clinical informatics extractor.
Your job is to read raw clinical notes or patient complaints and extract structured data.

Rules:
1. ONLY extract information explicitly stated in the text.
2. Do NOT hallucinate missing values. Leave them null/empty.
3. If a symptom has an anatomical region, duration, severity, or onset pattern, extract those into the proper fields.
4. If lab values or imaging findings are present, extract them.
"""),
            ("human", "Clinical text to process:\n{raw_text}")
        ])

    def _load_icd10_table(self) -> dict[str, str]:
        """Load local ICD-10 lookup table."""
        table = {}
        csv_path = Path(__file__).parent.parent.parent / "data" / "icd10_codes.csv"
        if csv_path.exists():
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    table[row["description"].lower()] = row["code"]
        return table

    def _lookup_icd10(self, case: StructuredCase) -> list[str]:
        """Simple exact-match ICD-10 lookup.
        
        Note for teaching: Real SNOMED/ICD-10 mapping requires a dedicated entity
        linking model (like MedCAT or UMLS). This is a simplified proxy.
        """
        codes = []
        
        # Check chief complaint
        if case.chief_complaint and case.chief_complaint.lower() in self.icd10_table:
            codes.append(self.icd10_table[case.chief_complaint.lower()])
            
        # Check past medical history
        for dx in case.past_medical_history:
            if dx.lower() in self.icd10_table:
                codes.append(self.icd10_table[dx.lower()])
                
        return list(set(codes))

    def process(self, raw_text: str, config: Optional[RunnableConfig] = None) -> StructuredCase:
        """Process raw text into a StructuredCase."""
        result = self.chain.invoke({"raw_text": raw_text}, config=config)
        
        # We know result is a StructuredCase because with_structured_output guarantees it
        # (or raises a validation error, which we let bubble up)
        case: StructuredCase = result
        
        # Attach any ICD-10 codes we can find
        case.icd10_codes = self._lookup_icd10(case)
        
        return case
