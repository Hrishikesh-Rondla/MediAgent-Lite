"""Evidence Scanner Agent.

Responsibility: Takes a SINGLE DiagnosisHypothesis, uses the PubMedTool
to search the web, and returns an EvidenceBundle containing findings
that either support or refute the hypothesis.
"""

from typing import Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from mediagent_lite.llm_factory import get_llm
from mediagent_lite.schemas.clinical import StructuredCase
from mediagent_lite.schemas.hypothesis import DiagnosisHypothesis
from mediagent_lite.schemas.evidence import EvidenceBundle, EvidenceRecord, PublicationType
from mediagent_lite.tools.pubmed_tool import PubMedTool
from mediagent_lite.config.settings import get_settings


class EvidenceScanner:
    """Agent that searches the web for evidence regarding a specific hypothesis."""

    def __init__(self):
        # We use the 'evidence_scanner' role, which is configured for Groq
        # by default because we will map over hypotheses concurrently.
        self.llm = get_llm("evidence_scanner")
        self.settings = get_settings()
        
        # Bind the tool to the LLM
        self.tool = PubMedTool()
        self.llm_with_tools = self.llm.bind_tools([self.tool])
        
        # We use a two-step process:
        # 1. ReAct loop (handled natively by LangChain agent executor, or we can write a simple loop)
        # 2. Structured output extraction
        
        # For MediAgent-Lite, to keep it simple and robust, we do a 1-shot tool call:
        # We ask the LLM to generate the query, we run the tool, then we ask it to extract the bundle.
        self.query_chain = ChatPromptTemplate.from_messages([
            ("system", "You are a medical researcher. Generate ONE highly specific PubMed search query to find evidence validating or refuting the proposed diagnosis for the given patient. Return ONLY the raw query string, nothing else."),
            ("human", "Case:\n{case_json}\n\nHypothesis: {diagnosis}")
        ]) | self.llm
        
        self.extract_chain = ChatPromptTemplate.from_messages([
            ("system", """You are an expert evidence appraiser.
Review the retrieved PubMed abstracts. Extract up to 3 of the most relevant pieces of evidence. NEVER EXTRACT MORE THAN 3.
Be strict. If the text does not explicitly mention the condition or symptoms, do not use it.

CRITICAL: You MUST respond with valid JSON matching this exact structure:
{{
  "records": [
    {{
      "hypothesis": "Diagnosis Name",
      "pmid": "123456",
      "title": "Article Title",
      "classified_type": "cohort",
      "evidence_weight": 0.6,
      "relevance_summary": "Short explanation",
      "year": 2024
    }}
  ]
}}
"""),
            ("human", "Case:\n{case_json}\n\nHypothesis: {diagnosis}\n\nRetrieved Literature:\n{literature}")
        ]) | self.llm.with_structured_output(EvidenceBundle)

    def process(
        self, 
        case: StructuredCase, 
        hypothesis: DiagnosisHypothesis, 
        config: Optional[RunnableConfig] = None
    ) -> EvidenceBundle:
        """Scan for evidence for a single hypothesis."""
        
        # 1. Generate Query
        case_json = case.model_dump_json(exclude_none=True)
        
        # OPTIMIZATION FOR LOCAL MODELS: 
        # Small models (8B) often hallucinate conversation rather than raw search strings.
        # We programmatically construct the keyword query to guarantee a valid PubMed search.
        import re
        
        # Clean diagnosis string (e.g. "Myocardial Infarction/Unstable Angina" -> "Myocardial Infarction Unstable Angina")
        clean_dx = re.sub(r'[^a-zA-Z0-9\s]', ' ', hypothesis.diagnosis).strip()
        
        # Use short symptom names instead of a massive chief complaint sentence
        sym_list = [sym.name for sym in case.symptoms][:2]
        sym_str = " ".join(sym_list) if sym_list else "diagnosis"
        
        # We DO NOT use quotes. PubMed keyword matching handles raw terms much better.
        query = f"{clean_dx} {sym_str}"
        
        # 2. Execute Tool
        # We bypass the LLM routing and just call the tool directly to ensure stability
        # and prevent infinite loops in the Lite version.
        literature = self.tool._run(query=query, retmax=3)
        
        if "No results found" in literature or "Error" in literature:
            return EvidenceBundle(
                records=[]
            )
            
        # 3. Extract Structured Evidence
        bundle: EvidenceBundle = self.extract_chain.invoke({
            "case_json": case_json,
            "diagnosis": hypothesis.diagnosis,
            "literature": literature
        }, config=config)
        
        # Inject weights based on publication type and compute scores
        weights = self.settings.evidence_weights
        total_weight = 0.0
        for record in bundle.records:
            record.hypothesis = hypothesis.diagnosis
            record.evidence_weight = weights.get(record.classified_type.value, 0.2)
            total_weight += record.evidence_weight
            
        bundle.weighted_scores[hypothesis.diagnosis] = max(0.0, min(1.0, total_weight))
            
        return bundle
